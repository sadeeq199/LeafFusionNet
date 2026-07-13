"""Model training orchestration for the LeafFusionNet project.

This module provides a trainer class that coordinates compilation, fitting, and
saving of an already defined LeafFusionNet architecture using already built
TensorFlow datasets.
"""

from pathlib import Path
import time

import tensorflow as tf

from config.hyperparameters import (
    INITIAL_EPOCHS,
    INITIAL_LEARNING_RATE,
    LOSS_FUNCTION,
    OPTIMIZER,
)
from config.paths import MODELS_DIR
from models.leaffusionnet import build_leaffusionnet
from training.callbacks import build_callbacks
from training.history import (
    plot_accuracy_curve,
    plot_learning_rate_curve,
    plot_loss_curve,
    save_training_history,
    save_training_summary,
)
from training.metadata import (
    generate_class_names,
    generate_dataset_summary,
    generate_experiment_metadata,
    generate_model_metadata,
)


class LeafFusionNetTrainer:
    """Trainer for compiling, fitting, and saving LeafFusionNet.

    Parameters
    ----------
    train_dataset : tf.data.Dataset
        Prepared training dataset.
    validation_dataset : tf.data.Dataset
        Prepared validation dataset.
    """

    def __init__(
        self,
        train_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
    ) -> None:
        """Initialize the trainer with datasets and an uncompiled model.

        Parameters
        ----------
        train_dataset : tf.data.Dataset
            Prepared training dataset.
        validation_dataset : tf.data.Dataset
            Prepared validation dataset.
        """
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.model = build_leaffusionnet()
        self._training_completed = False

    def compile(self) -> None:
        """Compile the LeafFusionNet model.

        Returns
        -------
        None
            This method configures the trainer model in place.
        """
        optimizer = tf.keras.optimizers.get(OPTIMIZER)
        optimizer.learning_rate = INITIAL_LEARNING_RATE

        self.model.compile(
            optimizer=optimizer,
            loss=LOSS_FUNCTION,
            metrics=["accuracy"],
        )

    def train(self) -> tf.keras.callbacks.History:
        """Train the compiled LeafFusionNet model.

        Returns
        -------
        tf.keras.callbacks.History
            Keras history object returned by ``model.fit``.
        """
        start_time = time.perf_counter()
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.validation_dataset,
            epochs=INITIAL_EPOCHS,
            callbacks=build_callbacks(),
        )
        training_time_seconds = time.perf_counter() - start_time

        save_training_history(history)
        save_training_summary(history, training_time_seconds)
        plot_accuracy_curve(history)
        plot_loss_curve(history)
        plot_learning_rate_curve(history)

        self._training_completed = True

        return history

    def save(self) -> Path:
        """Save the trained LeafFusionNet model.

        Returns
        -------
        Path
            Path to the saved model file.
        """
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "LeafFusionNet.keras"
        self.model.save(model_path)
        if self._training_completed:
            generate_class_names()
            generate_model_metadata(self.model, model_path)
            generate_dataset_summary()
            generate_experiment_metadata()
        return model_path


__all__ = ["LeafFusionNetTrainer"]
