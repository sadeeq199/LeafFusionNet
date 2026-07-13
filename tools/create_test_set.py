"""Create a reproducible, independent test dataset from LeafFusionNet training data.

Run from the project root with ``python tools/create_test_set.py``. The script
copies a stratified 15 percent sample from each training class and never alters
the training or validation datasets.
"""

from __future__ import annotations

import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.hyperparameters import RANDOM_SEED
from config.paths import REPORTS_DIR, TEST_DIR, TRAIN_DIR


SPLIT_RATIO: float = 0.15
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
REPORT_FILENAME = "test_dataset_report.txt"


@dataclass(frozen=True)
class ClassSplitResult:
    """Summary of one class processed by the test-set creation utility.

    Attributes:
        class_name: Name of the class directory.
        train_images: Number of supported image files in the training class.
        test_images: Number of supported image files in the resulting test class.
        target_images: Desired test-image count for the configured split ratio.
        copied_images: Number of new images copied during this execution.
        skipped_images: Number of images already present in the test class.
    """

    class_name: str
    train_images: int
    test_images: int
    target_images: int
    copied_images: int
    skipped_images: int


def is_supported_image(path: Path) -> bool:
    """Return whether a path is a supported image file.

    Args:
        path: Filesystem path to inspect.

    Returns:
        ``True`` when the path is a file with a supported image extension.
    """
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def list_images(directory: Path) -> list[Path]:
    """List supported images below a directory in deterministic order.

    Args:
        directory: Root directory to search recursively.

    Returns:
        Supported image paths sorted by their POSIX-style relative paths.

    Raises:
        FileNotFoundError: If ``directory`` does not exist.
        NotADirectoryError: If ``directory`` is not a directory.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a directory: {directory}")
    return sorted(
        (path for path in directory.rglob("*") if is_supported_image(path)),
        key=lambda path: path.relative_to(directory).as_posix().lower(),
    )


def get_class_directories(train_dir: Path) -> list[Path]:
    """Return the training class directories in deterministic order.

    Args:
        train_dir: Root directory containing one directory per class.

    Returns:
        Alphabetically sorted class directories.

    Raises:
        FileNotFoundError: If the train directory does not exist.
        NotADirectoryError: If the train path is not a directory.
        ValueError: If no class directories are present.
    """
    if not train_dir.exists():
        raise FileNotFoundError(f"Training dataset directory does not exist: {train_dir}")
    if not train_dir.is_dir():
        raise NotADirectoryError(f"Training dataset path is not a directory: {train_dir}")

    class_directories = sorted(
        (path for path in train_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name.lower(),
    )
    if not class_directories:
        raise ValueError(f"No class directories found in training dataset: {train_dir}")
    return class_directories


def copy_class_split(class_dir: Path, test_dir: Path, rng: random.Random) -> ClassSplitResult:
    """Copy a reproducible stratified sample for one training class.

    Existing test filenames are preserved and never overwritten. If a prior run
    already copied files for the class, only enough additional unique filenames
    are sampled to meet the 15 percent target where possible.

    Args:
        class_dir: Source class directory under the training dataset.
        test_dir: Root destination directory for the independent test dataset.
        rng: Seeded random-number generator used for reproducible sampling.

    Returns:
        A summary of the copied and resulting images for the class.

    Raises:
        ValueError: If the source class contains no supported images.
        OSError: If a required destination directory cannot be created or a
            selected image cannot be copied.
    """
    train_images = list_images(class_dir)
    if not train_images:
        raise ValueError(f"Training class contains no supported images: {class_dir}")

    destination_dir = test_dir / class_dir.name
    destination_dir.mkdir(parents=True, exist_ok=True)
    existing_images = list_images(destination_dir)
    skipped_images = len(existing_images)
    existing_filenames = {path.name.casefold() for path in existing_images}
    target_images = max(1, math.ceil(len(train_images) * SPLIT_RATIO))
    required_images = max(0, target_images - len(existing_images))

    candidates = [
        image for image in train_images if image.name.casefold() not in existing_filenames
    ]
    selected_images = rng.sample(candidates, k=min(required_images, len(candidates)))
    copied_images = 0
    for image in selected_images:
        destination = destination_dir / image.name
        if destination.name.casefold() in existing_filenames:
            continue
        shutil.copy2(image, destination)
        existing_filenames.add(destination.name.casefold())
        copied_images += 1

    test_images = len(list_images(destination_dir))
    return ClassSplitResult(
        class_name=class_dir.name,
        train_images=len(train_images),
        test_images=test_images,
        target_images=target_images,
        copied_images=copied_images,
        skipped_images=skipped_images,
    )


def build_report(
    results: Iterable[ClassSplitResult],
    execution_time: float,
) -> str:
    """Build the human-readable test dataset creation report.

    Args:
        results: Per-class split results in display order.
        execution_time: Total utility execution time in seconds.

    Returns:
        Complete report text suitable for console output and file storage.
    """
    split_results = list(results)
    total_train = sum(result.train_images for result in split_results)
    total_copied = sum(result.copied_images for result in split_results)
    total_skipped = sum(result.skipped_images for result in split_results)
    total_test = sum(result.test_images for result in split_results)
    separator = "=" * 36
    subsection = "-" * 36
    lines = [
        separator,
        "LeafFusionNet Test Dataset Report",
        separator,
        "",
        f"Classes: {len(split_results)}",
        f"Split Ratio: {SPLIT_RATIO:.0%}",
        "",
    ]
    for result in split_results:
        lines.extend(
            [
                subsection,
                result.class_name,
                f"Train Images : {result.train_images}",
                f"Copied Images : {result.copied_images}",
                f"Skipped Images : {result.skipped_images}",
                f"Test Images : {result.test_images}",
            ]
        )
    lines.extend(
        [
            subsection,
            f"TOTAL TRAIN IMAGES : {total_train}",
            f"TOTAL COPIED IMAGES : {total_copied}",
            f"TOTAL SKIPPED IMAGES : {total_skipped}",
            f"TOTAL TEST IMAGES : {total_test}",
            f"Execution Time : {execution_time:.2f} seconds",
            separator,
            "",
        ]
    )
    return "\n".join(lines)


def save_report(report: str, reports_dir: Path = REPORTS_DIR) -> Path:
    """Save the test dataset report to the configured results directory.

    Args:
        report: Complete report text to save.
        reports_dir: Directory in which the report file will be written.

    Returns:
        Path to the saved report.

    Raises:
        OSError: If the report directory or report file cannot be created.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / REPORT_FILENAME
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    """Create the independent test dataset and save its distribution report.

    Raises:
        FileNotFoundError: If the configured training dataset directory is absent.
        NotADirectoryError: If a configured dataset path is not a directory.
        ValueError: If a class has no supported images or no classes exist.
        OSError: If images or the report cannot be written.
    """
    start_time = time.perf_counter()
    class_directories = get_class_directories(TRAIN_DIR)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)
    results: list[ClassSplitResult] = []

    for class_dir in class_directories:
        print(f"Processing: {class_dir.name}")
        result = copy_class_split(class_dir, TEST_DIR, rng)
        results.append(result)
        print(f"Copied : {result.copied_images}")
        print(f"Skipped: {result.skipped_images}")
        print(f"Target : {result.target_images}")

    execution_time = time.perf_counter() - start_time
    report = build_report(results, execution_time)
    report_path = save_report(report)
    print(f"\n{report}")
    print(f"Execution Time : {execution_time:.2f} seconds")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
