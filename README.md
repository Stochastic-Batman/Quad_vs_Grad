# Quad_vs_Grad
First-Order vs Second-Order Optimization Methods for Deep Learning

This repository demonstrates and visualizes the differences between first-order (e.g., Gradient Descent) and second-order (e.g., Newton's method) optimization techniques. It uses PyTorch, NumPy, SciPy, and Matplotlib to compare their paths on a non-convex loss landscape. I optimize the [Himmelblau function](https://en.wikipedia.org/wiki/Himmelblau%27s_function) - a classic test function with multiple local minima that mimics the complex, "mountainous" loss surfaces found in neural networks. This makes it ideal for illustrating how second-order methods can navigate curvature better than first-order ones, often converging faster or avoiding poor paths.

The content and insights here are inspired by the University of Tübingen's "Numerics of Machine Learning" lecture slides (2022/23), available at the [Numerics of ML repo](https://github.com/philipphennig/NumericsOfML/tree/main), specifically the second-order optimization slide: [12_SecondOrderOptimization.pdf](https://github.com/philipphennig/NumericsOfML/blob/main/slides/12_SecondOrderOptimization.pdf). **Grok 4.1 Fast** was used for grammar correction of this `README.md`.

Citation for the lecture notes:
```
@techreport{NoML22,
     title = {Numerics of Machine Learning},
     author = {N. Bosch and J. Grosse and P. Hennig and A. Kristiadi and
               M. Pförtner and J. Schmidt and F. Schneider and L. Tatzel and J. Wenger},
     series = {Lecture Notes in Machine Learning},
     year = {2022},
     institution = {Tübingen AI Center},
}
```

## Advantages of Second-Order Methods
Second-order optimization methods, like Newton's method, offer several appealing benefits, particularly when compared to first-order (gradient-based) approaches:
- They are robust to ill-conditioned problems, handling steep or flat landscapes effectively.
- They converge quickly near the optimum, often achieving quadratic convergence rates.
- Newton steps inherently have a "natural length", avoiding the need for manual step-size tuning common in gradient descent.
- They have a proven track record and are the default in many scientific domains outside deep learning.

Despite these strengths, their application in deep learning is **limited** due to three core challenges: *non-convexity*, *computational cost*, and *stochasticity*. Below is a summary of each problem, why it's an issue in deep learning contexts, and the proposed solutions or workarounds from the University of Tübingen's "Numerics of ML" slides.

### 1. Non-Convexity
**Problem**: Deep learning objectives are typically non-convex, meaning the quadratic approximation (used in Newton's method) can be poor or misleading far from the optimum. Even in convex cases, the model might fail, leading to unreliable updates or divergence.

**Why It's a Challenge in DL**: Neural networks have high-dimensional, non-convex loss landscapes with saddle points and local minima. Standard Newton steps assume positive definite curvature, but negative curvature can point toward worse solutions.

**Answers/Solutions**:
- **Positive Semi-Definite Curvature Matrices (e.g., GGN and FIM)**: Use proxies like the Generalized Gauss-Newton (GGN) matrix or Fisher Information Matrix (FIM) for "meaningful" curvature approximations. These are always positive semi-definite, and often GGN equals FIM. They provide unbiased estimates even on finite datasets.
- **Damping**: Add regularization to control update conservatism. Heuristics like Levenberg-Marquardt damping work well in practice by blending Newton-like steps with safer gradient descent behaviors.

### 2. Cost (Accessing and Inverting the Curvature Matrix)
**Problem**: Computing a Newton step requires the Hessian (curvature) matrix, which is massive in deep learning (storage scales as **O(D^2)**, inversion as **O(D^3)**, where **D** is the number of parameters, often millions or billions).

**Why It's a Challenge in DL**: Direct computation is infeasible due to memory and time constraints. We need efficient ways to approximate, access, and "invert" this matrix without full materialization.

**Answers/Solutions**:
- **BFGS & L-BFGS**: Quasi-Newton methods that build a dynamic low-rank approximation of the Hessian using gradient history. L-BFGS is memory-efficient and the default for small deterministic problems (e.g., in **SciPy**'s `optimize.minimize`).
- **HF (Hessian-Free Optimization)**: Approximates Newton steps without storing the full Hessian, using iterative methods like conjugate gradients. It's close to true Newton behavior, requires little memory, but involves more sequential computations and can be problematic with layers like BatchNorm.
- **K-FAC (Kronecker-Factored Approximate Curvature)**: A lightweight approximation of the FIM using Kronecker products for block-diagonal structure. Widely used in uncertainty quantification (e.g., Bayesian DL), though it relies heavily on heuristic damping for stability.

These approaches are pragmatic, with pros (e.g., efficiency) and cons (e.g., approximations may not always capture full curvature).

### 3. Stochasticity
**Problem**: Deep learning uses stochastic gradients (e.g., from mini-batches) rather than exact ones, so we only have noisy estimates Ĥ and ĝ instead of the true Hessian H and gradient g. This violates the assumptions of deterministic second-order methods.

**Why It's a Challenge in DL**: Finite data introduces variance, making updates unreliable. Exact computations would require full-dataset passes, which is impractical for large-scale training.

**Answers/Solutions**:
- Fundamentally unsolved.

## Project Overview
The "deep learning task" here is a simplified proxy: minimizing the Himmelblau function, defined as **f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2**. This 2D function has four local minima, creating a visually engaging landscape that represents the non-convex challenges in training neural networks (e.g., avoiding saddles or suboptimal minima). The project runs Gradient Descent (first-order) and Newton's method (second-order), records their optimization paths, and visualizes them on contour plots - both as static images and animated GIFs. This highlights how second-order methods use curvature information for more efficient "jumps," while first-order methods take smaller, gradient-directed steps.

- **Dependencies**: Matplotlib (plots and animations), NumPy (arrays), SciPy (optimization helpers), Pillow (for saving GIFs), PyTorch (for autograd and Hessians).
- **Outputs**: `paths.png` (static overlay of paths) and `journey.gif` (animated paths).

## Project Structure
```
Quad_vs_Grad/
├── main.py             # Entry point: Runs optimizations and generates visuals
├── optimizers.py       # Implements Gradient Descent and Newton's method
├── visualize.py        # Functions for static plots and animations
├── requirements.txt    # Dependencies: matplotlib, numpy, pillow, scipy, torch
└── README.md           
```

After running, you'll also see generated files like `paths.png` and `journey.gif` in the root.

## How to Run
1. Clone the repo: `git clone https://github.com/yourusername/Quad_vs_Grad.git`
2. Navigate to the directory: `cd Quad_vs_Grad`
3. Create and activate a virtual environment: `python -m venv qvg_env` then `source qvg_env/bin/activate` (Linux/macOS) or `qvg_env\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the main script: `python main.py`
   - Customize via command-line arguments: Provide the initial point (e.g., `python main.py --x0 2.7 --y0 -3.14` to start at **[2.7, -3.14]**; defaults to **[0.0, 0.0]**) and major parameters for optimizers, such as `--lr 0.001` (learning rate for Gradient Descent), `--steps 1000` (max steps for both), `--tol 1e-6` (tolerance for convergence), and others like damping for Newton's method if implemented.
6. View outputs: Open `paths.png` for the static plot or `journey.gif` for the animation.
