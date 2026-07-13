"""Project workflow entry point for LeafFusionNet.

This module coordinates the complete project workflow by delegating directory
setup, reproducibility, dataset construction, training, saving, and evaluation
to the existing project components.
"""

from config.hyperparameters import RANDOM_SEED
from config.paths import create_project_directories
from data.dataset import build_datasets
from evaluation.evaluator import evaluate_model
from training.trainer import LeafFusionNetTrainer
from utils.seed import set_random_seed


def main() -> None:
    """Run the complete LeafFusionNet training and evaluation workflow.

    Returns
    -------
    None
        This function coordinates the workflow and prints final evaluation
        metrics to standard output.
    """
    create_project_directories()
    set_random_seed(RANDOM_SEED)

    train_dataset, validation_dataset = build_datasets()

    trainer = LeafFusionNetTrainer(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    trainer.compile()
    trainer.train()
    trainer.save()

    metrics = evaluate_model(
        model=trainer.model,
        dataset=validation_dataset,
    )

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
