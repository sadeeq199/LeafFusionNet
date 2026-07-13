"""Callback factory utilities for LeafFusionNet training."""

import tensorflow as tf

from config.hyperparameters import (
    CHECKPOINT_MODE,
    CHECKPOINT_MONITOR,
    CHECKPOINT_SAVE_BEST_ONLY,
    CHECKPOINT_SAVE_FREQ,
    EARLY_STOPPING_MODE,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_FACTOR,
    REDUCE_LR_MIN_LR,
    REDUCE_LR_PATIENCE,
    RESTORE_BEST_WEIGHTS,
)
from config.paths import CHECKPOINTS_DIR


def build_callbacks() -> list[tf.keras.callbacks.Callback]:
    """Create the configured Keras callbacks for model training.

    Returns
    -------
    list[tf.keras.callbacks.Callback]
        Callback list containing ``ModelCheckpoint``, ``EarlyStopping``, and
        ``ReduceLROnPlateau``.

    Notes
    -----
    This function only instantiates callbacks. It does not create directories,
    compile models, train models, or perform any dataset operations.
    """
    checkpoint_path = CHECKPOINTS_DIR / "best_model.keras"

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=CHECKPOINT_MONITOR,
            mode=CHECKPOINT_MODE,
            save_best_only=CHECKPOINT_SAVE_BEST_ONLY,
            save_freq=CHECKPOINT_SAVE_FREQ,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=EARLY_STOPPING_MONITOR,
            patience=EARLY_STOPPING_PATIENCE,
            mode=EARLY_STOPPING_MODE,
            restore_best_weights=RESTORE_BEST_WEIGHTS,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=EARLY_STOPPING_MONITOR,
            mode=EARLY_STOPPING_MODE,
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN_LR,
        ),
    ]


__all__ = ["build_callbacks"]
