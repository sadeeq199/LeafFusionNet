"""Filesystem path configuration for the LeafFusionNet project."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

CONFIG_DIR: Path = PROJECT_ROOT / "config"

DATA_DIR: Path = PROJECT_ROOT / "data"
TRAIN_DIR: Path = DATA_DIR / "train"
VALID_DIR: Path = DATA_DIR / "valid"
TEST_DIR: Path = DATA_DIR / "test"

MODELS_DIR: Path = PROJECT_ROOT / "models"
CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
REPORTS_DIR: Path = RESULTS_DIR / "reports"


def create_project_directories() -> None:
    """Create required generated-output directories.

    Dataset directories are intentionally excluded because this module must not
    create or alter dataset storage locations.
    """
    output_directories: tuple[Path, ...] = (
        MODELS_DIR,
        CHECKPOINTS_DIR,
        LOGS_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
    )

    for directory in output_directories:
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "TRAIN_DIR",
    "VALID_DIR",
    "TEST_DIR",
    "MODELS_DIR",
    "CHECKPOINTS_DIR",
    "LOGS_DIR",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "REPORTS_DIR",
    "CONFIG_DIR",
    "create_project_directories",
]
