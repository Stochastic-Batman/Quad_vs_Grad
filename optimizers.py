import logging
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s  (%(levelname)s): %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(Path(__file__).stem + "Logger")

_EPS = 1e-7


class TinyMLP(nn.Module):
    """
    Two-layer MLP for binary XOR classification with exactly two trainable parameters.

    Architecture:
    - Input:   x in R^2
    - Hidden:  h = tanh(W_fix * x + b_fix) in R^2
                 W_fix = [[1, 1], [1, -1]],  b_fix = 0  <- FROZEN at init
    - Output:  y_pred = sigma(v * h) in (0, 1)
                 v = (v1, v2) in R^2  <- the only TRAINABLE parameters (no bias)
    - Loss:    Binary cross-entropy
                 L(v) = -1/N sum_i [y_i log y_pred_i + (1 - y_i) log(1 - y_pred_i)]
    """

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.tensor([[1.0, 1.0], [1.0, -1.0]]))
            self.hidden.bias.zero_()
        for param in self.hidden.parameters():
            param.requires_grad_(False)

        self.output = nn.Linear(2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.hidden(x))
        return torch.sigmoid(self.output(h)).squeeze(-1)


def make_model(x0: float, y0: float) -> TinyMLP:
    model = TinyMLP()
    with torch.no_grad():
        model.output.weight.copy_(torch.tensor([[x0, y0]]))
    return model


def get_params(model: TinyMLP) -> np.ndarray:
    return model.output.weight.detach().cpu().numpy().flatten().copy()


def set_params(model: TinyMLP, params: np.ndarray) -> None:
    with torch.no_grad():
        model.output.weight.copy_(torch.tensor(params, dtype=torch.float32).unsqueeze(0))


def get_hidden_activations(X: torch.Tensor) -> np.ndarray:
    """Return frozen hidden-layer activations H in R^{N x 2} for the dataset X."""
    model = TinyMLP()
    with torch.no_grad():
        return torch.tanh(model.hidden(X)).numpy()


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(preds: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(preds, _EPS, 1.0 - _EPS)
    return -float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def grad_bce(H: np.ndarray, preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    return H.T @ (preds - y) / len(y)


def run_sgd(X: torch.Tensor, y: torch.Tensor, x0: float, y0: float, lr: float, steps: int, tol: float) -> list[np.ndarray]:
    """
    Stochastic Gradient Descent (full-batch variant):
        v_{t+1} = v_t - eta * nabla L(v_t)

    - v_t:     trainable parameter vector at step t
    - eta:     learning rate (step size)
    - nabla L: gradient of the binary cross-entropy loss w.r.t. v_t
    """
    model = make_model(x0, y0)
    path = [get_params(model)]
    optimizer = torch.optim.SGD(model.output.parameters(), lr=lr)

    for step in range(steps):
        optimizer.zero_grad()

        loss = nn.functional.binary_cross_entropy(model(X), y)
        loss.backward()
        optimizer.step()

        path.append(get_params(model))
        if np.linalg.norm(path[-1] - path[-2]) < tol:
            logger.info("SGD converged at step %d  (loss=%.4f)", step + 1, loss.item())
            break
    else:
        logger.info("SGD reached max steps (%d)", steps)

    return path


def run_adam(X: torch.Tensor, y: torch.Tensor, x0: float, y0: float, lr: float, steps: int, tol: float) -> list[np.ndarray]:
    """
    Adam (Adaptive Moment Estimation):
        m_t = beta1 * m_{t-1} + (1 - beta1) * nabla L(v_t)
        s_t = beta2 * s_{t-1} + (1 - beta2) * [nabla L(v_t)]^2
        m_hat_t = m_t / (1 - beta1^t),   s_hat_t = s_t / (1 - beta2^t)
        v_{t+1} = v_t - eta * m_hat_t / (sqrt(s_hat_t) + epsilon)

    - v_t:              parameter vector at step t
    - m_t:              first moment (exponentially weighted mean of gradients)
    - s_t:              second moment (exponentially weighted mean of squared gradients)
    - m_hat_t, s_hat_t: bias-corrected moment estimates
    - eta:              learning rate
    - beta1, beta2:     decay rates (defaults: 0.9, 0.999)
    - epsilon:          small constant for numerical stability (default: 1e-8)
    """
    model = make_model(x0, y0)
    path = [get_params(model)]
    optimizer = torch.optim.Adam(model.output.parameters(), lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy(model(X), y)
        loss.backward()
        optimizer.step()
        path.append(get_params(model))
        if np.linalg.norm(path[-1] - path[-2]) < tol:
            logger.info("Adam converged at step %d  (loss=%.4f)", step + 1, loss.item())
            break
    else:
        logger.info("Adam reached max steps (%d)", steps)

    return path


def run_newton(X: torch.Tensor, y: torch.Tensor, x0: float, y0: float, steps: int, tol: float, damping: float) -> list[np.ndarray]:
    """
    Damped Newton's method:
        v_{t+1} = v_t - (H(v_t) + lambda * I)^{-1} * nabla L(v_t)

    - v_t:     parameter vector at step t
    - H(v_t):  exact Hessian of L w.r.t. v_t, computed via double backpropagation
    - lambda:  damping (Levenberg-Marquardt) - keeps H + lambda * I positive-definite
    - I:       2x2 identity matrix
    - nabla L: gradient of binary cross-entropy w.r.t. v_t
    """
    H_fix = torch.tanh(TinyMLP().hidden(X)).detach()

    def loss_fn(p: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(H_fix @ p)
        return nn.functional.binary_cross_entropy(pred, y)

    params = np.array([x0, y0], dtype=np.float32)
    path = [params.copy()]

    for step in range(steps):  # I had never written this before
        p_t = torch.tensor(params, dtype=torch.float32)
        grad = torch.autograd.functional.jacobian(loss_fn, p_t).numpy()
        hess = torch.autograd.functional.hessian(loss_fn, p_t).numpy()

        H_damp = hess + damping * np.eye(2)
        delta = np.linalg.solve(H_damp, grad)
        params = params - delta
        path.append(params.copy())

        if np.linalg.norm(delta) < tol:
            logger.info("Newton converged at step %d", step + 1)
            break
    else:
        logger.info("Newton reached max steps (%d)", steps)

    return path


def run_bfgs(X: torch.Tensor, y: torch.Tensor, x0: float, y0: float, steps: int, tol: float) -> list[np.ndarray]:
    """
    BFGS (Broyden-Fletcher-Goldfarb-Shanno) quasi-Newton method.
    Builds a rank-2 update approximation B_t approx H(v_t)^{-1}:
        s_t = v_t - v_{t-1},   q_t = nabla L(v_t) - nabla L(v_{t-1})
        B_{t+1} = (I - rho_t * s_t * q_t^T) * B_t * (I - rho_t * q_t * s_t^T) + rho_t * s_t * s_t^T
        v_{t+1} = v_t - B_{t+1} * nabla L(v_t),   rho_t = 1 / (q_t^T * s_t)

    - B_t:     running inverse-Hessian approximation (2x2 dense matrix)
    - s_t:     parameter step vector
    - q_t:     gradient difference vector
    - rho_t:   curvature scaling factor
    - nabla L: gradient of binary cross-entropy w.r.t. v_t
    """
    H_np = get_hidden_activations(X)
    y_np = y.numpy()

    path = [np.array([x0, y0], dtype=np.float64)]

    def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        preds = sigmoid(H_np @ params)
        loss = bce_loss(preds, y_np)
        grad = grad_bce(H_np, preds, y_np)
        return float(loss), grad.astype(np.float64)

    def callback(xk: np.ndarray) -> None:
        path.append(xk.copy())

    result = minimize(loss_and_grad, np.array([x0, y0], dtype=np.float64), method='BFGS', jac=True, callback=callback, options={'maxiter': steps, 'gtol': tol})
    logger.info("BFGS: %s  (final loss=%.4f)", result.message, result.fun)
    return path


def run_lbfgs(X: torch.Tensor, y: torch.Tensor, x0: float, y0: float, steps: int, tol: float) -> list[np.ndarray]:
    """
    L-BFGS (Limited-memory BFGS): stores only the m most recent (s, q) pairs
    instead of the full dense inverse-Hessian B_t, reducing memory from O(D^2)
    to O(m * D).  The implicit two-loop recursion recovers the Newton direction:
        v_{t+1} = v_t - H_tilde_t^{-1} * nabla L(v_t)

    - H_tilde_t^{-1}: low-rank inverse-Hessian approximation built from m curvature pairs
    - s_k = v_{k+1} - v_k:                       parameter step (stored for k = t-m, ..., t-1)
    - q_k = nabla L(v_{k+1}) - nabla L(v_k):     gradient difference (stored alongside s_k)
    - m:       memory size (number of pairs kept; SciPy default: 10)
    - nabla L: gradient of binary cross-entropy w.r.t. v_t
    """
    H_np = get_hidden_activations(X)
    y_np = y.numpy()

    path = [np.array([x0, y0], dtype=np.float64)]

    def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        preds = sigmoid(H_np @ params)
        loss = bce_loss(preds, y_np)
        grad = grad_bce(H_np, preds, y_np)
        return float(loss), grad.astype(np.float64)

    def callback(xk: np.ndarray) -> None:
        path.append(xk.copy())

    result = minimize(loss_and_grad, np.array([x0, y0], dtype=np.float64), method='L-BFGS-B', jac=True, callback=callback, options={'maxiter': steps, 'gtol': tol})
    logger.info("L-BFGS: %s  (final loss=%.4f)", result.message, result.fun)
    return path


def run_ggn(X: torch.Tensor, y: torch.Tensor, x0: float, y0: float, steps: int, tol: float, damping: float) -> list[np.ndarray]:
    """
    Generalized Gauss-Newton (GGN) method:
        G(v_t) = 1/N * sum_i fi * (1 - fi) * hi * hi^T
        v_{t+1} = v_t - (G(v_t) + lambda * I)^{-1} * nabla L(v_t)

    - G(v_t):    GGN curvature matrix - always positive semi-definite (PSD)
    - fi:        model output sigma(v_t^T * hi) for sample i
    - hi:        hidden activation tanh(W_fix * xi) for sample i, shape (2,)
    - fi*(1-fi): per-sample output curvature (Bernoulli variance)
    - lambda:    damping factor that ensures G + lambda * I is strictly positive-definite
    - nabla L:   gradient of binary cross-entropy: 1/N * sum_i (fi - yi) * hi
    """
    H_np = get_hidden_activations(X)
    y_np = y.numpy()

    params = np.array([x0, y0], dtype=np.float64)
    path = [params.copy()]

    for step in range(steps):
        preds = sigmoid(H_np @ params)
        grad = grad_bce(H_np, preds, y_np)

        weights = preds * (1.0 - preds)
        G = (H_np * weights[:, None]).T @ H_np / len(y_np)
        G_damp = G + damping * np.eye(2)
        delta = np.linalg.solve(G_damp, grad)
        params = params - delta
        path.append(params.copy())

        if np.linalg.norm(delta) < tol:
            logger.info("GGN converged at step %d", step + 1)
            break
    else:
        logger.info("GGN reached max steps (%d)", steps)

    return path


def run_kfac(X: torch.Tensor, y: torch.Tensor, x0: float, y0: float, steps: int, tol: float, damping: float) -> list[np.ndarray]:
    """
    Kronecker-Factored Approximate Curvature (K-FAC):
        A      = 1/N * sum_i hi * hi^T       (input covariance, shape 2x2)
        Gamma  = 1/N * sum_i fi * (1 - fi)   (mean output curvature, scalar)
        G_kfac = Gamma * A                    (Kronecker product = scalar * matrix for 1-output)
        v_{t+1} = v_t - (G_kfac + lambda * I)^{-1} * nabla L(v_t)

    - A:       activation covariance - captures input geometry of the output layer
    - Gamma:   mean curvature of the loss w.r.t. the pre-activation output
    - G_kfac:  approximates GGN by assuming Gamma and hi*hi^T are statistically independent;
               exact GGN would be 1/N * sum_i fi*(1-fi) * hi*hi^T  (correlated product)
    - lambda:  damping factor
    - nabla L: gradient of binary cross-entropy: 1/N * sum_i (fi - yi) * hi
    """
    H_np = get_hidden_activations(X)
    y_np = y.numpy()

    params = np.array([x0, y0], dtype=np.float64)
    path = [params.copy()]

    A = H_np.T @ H_np / len(y_np)

    for step in range(steps):
        preds = sigmoid(H_np @ params)
        grad = grad_bce(H_np, preds, y_np)

        gamma = float(np.mean(preds * (1.0 - preds)))
        G_kfac = gamma * A
        G_damp = G_kfac + damping * np.eye(2)
        delta = np.linalg.solve(G_damp, grad)
        params = params - delta
        path.append(params.copy())

        if np.linalg.norm(delta) < tol:
            logger.info("K-FAC converged at step %d", step + 1)
            break
    else:
        logger.info("K-FAC reached max steps (%d)", steps)

    return path