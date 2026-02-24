import argparse
import logging

from data import generate_xor_data
from optimizers import (
    get_hidden_activations,
    run_sgd,
    run_adam,
    run_newton,
    run_bfgs,
    run_lbfgs,
    run_ggn,
    run_kfac,
)
from pathlib import Path
from visualize import save_static_plot, save_animation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s  (%(levelname)s): %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(Path(__file__).stem + "Logger")


_FIRST_ORDER_FNS = {'sgd': run_sgd, 'adam': run_adam}
_SECOND_ORDER_FNS = {'newton': run_newton, 'bfgs': run_bfgs, 'lbfgs': run_lbfgs, 'ggn': run_ggn, 'kfac': run_kfac}
_DEFAULT_LR = {'sgd': 0.8, 'adam': 0.05}
_COLORS = {  # some random colors (slightly adjusted to look better than pure random colors (some looked similar)
    'sgd': '#003366',
    'adam': '#00FFFF',
    'newton': '#FF0000',
    'bfgs': '#FF00FF',
    'lbfgs': '#800080',
    'ggn': '#FFA500',
    'kfac': '#5C4033',
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare first-order vs second-order optimization on a tiny XOR MLP.')
    parser.add_argument('--x0', type=float, default=2.0, help='Initial value of output weight v₁  (default: 2.0)')
    parser.add_argument('--y0', type=float, default=-2.0, help='Initial value of output weight v₂  (default: -2.0)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate for the first-order method  (default: per-optimizer)')
    parser.add_argument('--steps', type=int, default=500, help='Maximum optimisation steps  (default: 500)')
    parser.add_argument('--tol',type=float, default=1e-5, help='Convergence tolerance on parameter step norm  (default: 1e-5)')
    parser.add_argument('--damping', type=float, default=0.1, help='Damping λ for Newton / GGN / K-FAC  (default: 0.1)')
    return parser.parse_args()


def _run_first_order(name: str, X, y, args) -> list:
    lr = args.lr if args.lr is not None else _DEFAULT_LR[name]
    logger.info("Running %s  (lr=%.4f, max_steps=%d)", name.upper(), lr, args.steps)
    return _FIRST_ORDER_FNS[name](X=X, y=y, x0=args.x0, y0=args.y0, lr=lr, steps=args.steps, tol=args.tol)


def _run_second_order(name: str, X, y, args) -> list:
    logger.info("Running %s  (max_steps=%d)", name.upper(), args.steps)
    common = dict(X=X, y=y, x0=args.x0, y0=args.y0, steps=args.steps, tol=args.tol)
    if name in ('newton', 'ggn', 'kfac'):  # BFGS & L-BFGS do not support damping
        return _SECOND_ORDER_FNS[name](**common, damping=args.damping)
    return _SECOND_ORDER_FNS[name](**common)


def main() -> None:
    args = parse_args()
    X, y = generate_xor_data(n_points=200, noise=0.2)
    H_fixed = get_hidden_activations(X)
    y_np = y.numpy()

    all_names = list(_FIRST_ORDER_FNS.keys()) + list(_SECOND_ORDER_FNS.keys())
    paths, names, colors = [], [], []

    for name in all_names:
        if name in _FIRST_ORDER_FNS:
            path = _run_first_order(name, X, y, args)
        else:
            path = _run_second_order(name, X, y, args)

        paths.append(path)
        names.append(name.upper())
        colors.append(_COLORS.get(name, '#000000'))

    save_static_plot(H_fixed, y_np, paths, names, colors, output_path='paths.png')
    logger.info('Started generating GIF... This took 2 minutes to run on my moderately powered machine:(')
    save_animation(H_fixed, y_np, paths, names, colors, output_path='journey.gif')

    logger.info("Comparison complete. Check paths.png and journey.gif")


if __name__ == '__main__':
    main()