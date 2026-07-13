"""Centralized hyperparameter configuration for the LeafFusionNet project.

This module is the single source of truth for configurable model, dataset,
training, callback, regularization, and reproducibility values used throughout
the project. It intentionally contains constants only and has no runtime
dependencies on deep learning or numerical libraries.
"""

# =============================================================================
# Image Configuration
# =============================================================================
IMG_SIZE: int = 224
IMG_CHANNELS: int = 3
INPUT_SHAPE: tuple[int, int, int] = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
NUM_CLASSES: int = 33

# =============================================================================
# Stem Configuration
# =============================================================================
PADDING: str = "same"
STEM_FILTERS: int = 32
STEM_KERNEL_SIZE: tuple[int, int] = (3, 3)
STEM_STRIDES: tuple[int, int] = (2, 2)

# =============================================================================
# Branch A Configuration
# =============================================================================
BRANCH_A_FILTERS: tuple[int, int] = (32, 64)
BRANCH_A_KERNEL_SIZES: tuple[tuple[int, int], tuple[int, int]] = (
    (3, 3),
    (3, 3),
)

# =============================================================================
# Branch B Configuration
# =============================================================================
BRANCH_B_FILTERS: tuple[int, int] = (32, 64)
BRANCH_B_KERNEL_SIZES: tuple[tuple[int, int], tuple[int, int]] = (
    (5, 5),
    (3, 3),
)
BRANCH_B_DILATION_RATE: int = 3

# =============================================================================
# Feature Fusion Configuration
# =============================================================================
FUSION_UNITS: int = 256
FUSION_DROPOUT_1: float = 0.20
FUSION_DROPOUT_2: float = 0.30

# =============================================================================
# Classifier Configuration
# =============================================================================
CLASSIFIER_UNITS: int = NUM_CLASSES
CLASSIFIER_ACTIVATION: str = "softmax"

# =============================================================================
# Training Configuration
# =============================================================================
BATCH_SIZE: int = 32
INITIAL_EPOCHS: int = 50
INITIAL_LEARNING_RATE: float = 1e-3
OPTIMIZER: str = "adam"
LOSS_FUNCTION: str = "sparse_categorical_crossentropy"

# =============================================================================
# Regularization Configuration
# =============================================================================
L2_REGULARIZATION: float = 1e-4

# =============================================================================
# Callback Configuration
# =============================================================================
EARLY_STOPPING_MONITOR: str = "val_loss"
EARLY_STOPPING_PATIENCE: int = 10
EARLY_STOPPING_MODE: str = "min"
RESTORE_BEST_WEIGHTS: bool = True

CHECKPOINT_MONITOR: str = "val_loss"
CHECKPOINT_MODE: str = "min"
CHECKPOINT_SAVE_BEST_ONLY: bool = True
CHECKPOINT_SAVE_FREQ: str = "epoch"

REDUCE_LR_FACTOR: float = 0.2
REDUCE_LR_PATIENCE: int = 5
REDUCE_LR_MIN_LR: float = 1e-6

# =============================================================================
# Dataset Configuration
# =============================================================================
SHUFFLE_DATASET: bool = True
CACHE_DATASET: bool = True
PREFETCH_DATASET: bool = True

# =============================================================================
# Reproducibility Configuration
# =============================================================================
RANDOM_SEED: int = 42

__all__ = [
    "IMG_SIZE",
    "IMG_CHANNELS",
    "INPUT_SHAPE",
    "NUM_CLASSES",
    "PADDING",
    "STEM_FILTERS",
    "STEM_KERNEL_SIZE",
    "STEM_STRIDES",
    "BRANCH_A_FILTERS",
    "BRANCH_A_KERNEL_SIZES",
    "BRANCH_B_FILTERS",
    "BRANCH_B_KERNEL_SIZES",
    "BRANCH_B_DILATION_RATE",
    "FUSION_UNITS",
    "FUSION_DROPOUT_1",
    "FUSION_DROPOUT_2",
    "CLASSIFIER_UNITS",
    "CLASSIFIER_ACTIVATION",
    "BATCH_SIZE",
    "INITIAL_EPOCHS",
    "INITIAL_LEARNING_RATE",
    "OPTIMIZER",
    "LOSS_FUNCTION",
    "L2_REGULARIZATION",
    "EARLY_STOPPING_MONITOR",
    "EARLY_STOPPING_PATIENCE",
    "EARLY_STOPPING_MODE",
    "RESTORE_BEST_WEIGHTS",
    "CHECKPOINT_MONITOR",
    "CHECKPOINT_MODE",
    "CHECKPOINT_SAVE_BEST_ONLY",
    "CHECKPOINT_SAVE_FREQ",
    "REDUCE_LR_FACTOR",
    "REDUCE_LR_PATIENCE",
    "REDUCE_LR_MIN_LR",
    "SHUFFLE_DATASET",
    "CACHE_DATASET",
    "PREFETCH_DATASET",
    "RANDOM_SEED",
]
