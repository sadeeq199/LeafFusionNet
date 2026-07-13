"""TensorFlow dataset construction for the LeafFusionNet project."""

import tensorflow as tf

from config.hyperparameters import (
    BATCH_SIZE,
    CACHE_DATASET,
    IMG_SIZE,
    PREFETCH_DATASET,
    RANDOM_SEED,
    SHUFFLE_DATASET,
)
from config.paths import DATA_DIR, TEST_DIR, TRAIN_DIR, VALID_DIR
from data.augmentation import augment_image
from data.preprocessing import preprocess_image


def build_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Build training and validation datasets from project directories.

    Returns
    -------
    tuple[tf.data.Dataset, tf.data.Dataset]
        Training and validation datasets with preprocessing applied to both
        datasets and augmentation applied only to the training dataset.

    Raises
    ------
    FileNotFoundError
        If ``DATA_DIR``, ``TRAIN_DIR``, or ``VALID_DIR`` does not exist.

    Notes
    -----
    Labels are inferred automatically from class subdirectory names by
    ``tf.keras.utils.image_dataset_from_directory``. This function does not
    implement preprocessing or augmentation logic directly; it only composes
    the existing project utilities into TensorFlow dataset pipelines.
    """
    if not DATA_DIR.is_dir():
        msg = f"Dataset directory does not exist: {DATA_DIR}"
        raise FileNotFoundError(msg)
    if not TRAIN_DIR.is_dir():
        msg = f"Training directory does not exist: {TRAIN_DIR}"
        raise FileNotFoundError(msg)
    if not VALID_DIR.is_dir():
        msg = f"Validation directory does not exist: {VALID_DIR}"
        raise FileNotFoundError(msg)

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_DATASET,
        seed=RANDOM_SEED,
    )
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    train_dataset = train_dataset.map(
        lambda images, labels: (preprocess_image(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_dataset = train_dataset.map(
        lambda images, labels: (augment_image(images, training=True), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    validation_dataset = validation_dataset.map(
        lambda images, labels: (preprocess_image(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if CACHE_DATASET:
        train_dataset = train_dataset.cache()
        validation_dataset = validation_dataset.cache()

    if PREFETCH_DATASET:
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, validation_dataset


def build_test_dataset() -> tf.data.Dataset:
    """Build the independent, non-augmented test dataset.

    The test dataset uses the same image size, batch size, integer labels, and
    preprocessing routine as the training and validation datasets. Its file
    order is deterministic so predictions can be matched to image paths.

    Returns:
        A cached and prefetched TensorFlow dataset yielding normalized image
        batches and integer class labels.

    Raises:
        FileNotFoundError: If the project data directory or test directory is
            missing.
    """
    if not DATA_DIR.is_dir():
        msg = f"Dataset directory does not exist: {DATA_DIR}"
        raise FileNotFoundError(msg)
    if not TEST_DIR.is_dir():
        msg = f"Test directory does not exist: {TEST_DIR}"
        raise FileNotFoundError(msg)

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_dataset = test_dataset.map(
        lambda images, labels: (preprocess_image(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if CACHE_DATASET:
        test_dataset = test_dataset.cache()
    if PREFETCH_DATASET:
        test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    return test_dataset


__all__ = ["build_datasets", "build_test_dataset"]
