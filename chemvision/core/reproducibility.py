"""Global seed propagation for full reproducibility.

Call ``set_global_seed(42)`` once at startup to seed Python, NumPy,
PyTorch, and RDKit simultaneously.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Seed all RNG sources for deterministic behaviour.

    Seeds: Python random, numpy, torch (if available), environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except ImportError:
        pass
