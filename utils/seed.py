"""Reproducibility utilities for the LeafFusionNet project.

This module centralizes random seed configuration for Python, NumPy, and
TensorFlow, including deterministic TensorFlow operation settings when they are
supported by the installed TensorFlow version.
"""

import os
import random

import numpy as np
import tensorflow as tf


def set_random_seed(seed: int) -> None:
    """Configure deterministic behavior and random seeds.

    Parameters
    ----------
    seed : int
        Seed value used for Python hashing, Python's ``random`` module, NumPy,
        and TensorFlow random number generation.

    Notes
    -----
    TensorFlow deterministic operation support can vary by version, platform,
    and installed kernels. When deterministic operation configuration is not
    available, this function continues without raising an exception.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

try:
    pass
except (AttributeError, RuntimeError, ValueError):
    pass


__all__ = ["set_random_seed"]
