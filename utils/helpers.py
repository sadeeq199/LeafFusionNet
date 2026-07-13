"""Generic filesystem helper utilities for the LeafFusionNet project."""

from pathlib import Path


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist.

    Parameters
    ----------
    path : Path
        Directory path to create.

    Returns
    -------
    Path
        The created or existing directory path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_directory(path: Path) -> Path:
    """Validate that a directory exists.

    Parameters
    ----------
    path : Path
        Directory path to validate.

    Returns
    -------
    Path
        The validated directory path.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    """
    if not path.is_dir():
        msg = f"Directory does not exist: {path}"
        raise FileNotFoundError(msg)

    return path


def validate_file(path: Path) -> Path:
    """Validate that a file exists.

    Parameters
    ----------
    path : Path
        File path to validate.

    Returns
    -------
    Path
        The validated file path.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not path.is_file():
        msg = f"File does not exist: {path}"
        raise FileNotFoundError(msg)

    return path


__all__ = ["ensure_directory", "validate_directory", "validate_file"]
