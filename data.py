import logging
import numpy as np
import torch

from pathlib import Path


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s  (%(levelname)s): %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(Path(__file__).stem + "Logger")


def generate_xor_data(n_points: int = 1000, noise: float = 0.08, seed: int = 95) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)  # for reproducibility

    xy = rng.uniform(-1.0, 1.0, (n_points, 2)).astype(np.float32)
    labels = ((xy[:, 0] > 0) ^ (xy[:, 1] > 0)).astype(np.float32)

    xy += rng.normal(0.0, noise, xy.shape).astype(np.float32)
    np.clip(xy, -1.0, 1.0, out=xy)

    logger.info("Generated XOR dataset: %d points, noise=%.3f, seed=%d",n_points, noise, seed)
    return torch.from_numpy(xy), torch.from_numpy(labels)


if __name__ == "__main__":
    print(generate_xor_data())