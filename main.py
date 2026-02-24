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

logging.basicConfig(level=logging.INFO, format='%(asctime)s- %(name)s  (%(levelname)s): %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(Path(__file__).stem + "Logger")


_FIRST_ORDER_FNS = {'sgd': run_sgd, 'adam': run_adam}
_SECOND_ORDER_FNS = {'newton': run_newton, 'bfgs': run_bfgs, 'lbfgs': run_lbfgs, 'ggn': run_ggn, 'kfac': run_kfac}
_DEFAULT_LR = {'sgd': 0.8, 'adam': 0.05}
_COLORS = {  # some random colors (slightly adjusted to look better than pure random colors (some looked similar)
    'sgd':    '#4C9BE8',
    'adam':   '#5DADE2',
    'newton': '#E8784C',
    'bfgs':   '#E84CA0',
    'lbfgs':  '#A04CE8',
    'ggn':    '#E8C34C',
    'kfac':   '#4CE878',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare first-order vs second-order optimization on a tiny XOR MLP.')
    parser.add_argument('--x0', type=float, default=2.0, help='Initial value of output weight v₁  (default: 2.0)')
    parser.add_argument('--y0', type=float, default=-2.0, help='Initial value of output weight v₂  (default: -2.0)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate for the first-order method  (default: per-optimizer)')
    parser.add_argument('--steps', type=int, default=500, help='Maximum optimisation steps  (default: 500)')
    parser.add_argument('--tol',type=float, default=1e-5, help='Convergence tolerance on parameter step norm  (default: 1e-5)')
    parser.add_argument('--first_order', type=str, default='sgd', choices=list(_FIRST_ORDER_FNS), help='First-order optimizer  (default: sgd)')
    parser.add_argument('--second_order', type=str, default='lbfgs', choices=list(_SECOND_ORDER_FNS), help='Second-order optimizer  (default: lbfgs)')
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
    logger.info("Config -> first_order=%s, second_order=%s, start=(%.2f, %.2f), damping=%.3f", args.first_order, args.second_order, args.x0, args.y0, args.damping)

    X, y = generate_xor_data(n_points=200, noise=0.1)
    H_fixed = get_hidden_activations(X)
    y_np = y.numpy()

    path_fo = _run_first_order(args.first_order, X, y, args)
    path_so = _run_second_order(args.second_order, X, y, args)

    paths = [path_fo, path_so]
    names = [args.first_order.upper(), args.second_order.upper()]
    colors = [_COLORS[args.first_order], _COLORS[args.second_order]]

    save_static_plot(H_fixed, y_np, paths, names, colors, output_path='paths.png')
    save_animation(H_fixed, y_np, paths, names, colors, output_path='journey.gif')

    logger.info("All done.  Outputs: paths.png  |  journey.gif")


if __name__ == '__main__':
    main()