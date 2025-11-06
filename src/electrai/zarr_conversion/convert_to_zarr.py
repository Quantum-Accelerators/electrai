"""
Convert Materials Project CHGCAR data from JSON.gz format to Zarr format.

This module provides functions to convert CHGCAR charge density data stored in
compressed JSON files to the Zarr format for efficient storage and access.
"""

from __future__ import annotations

import gzip
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .zarr_writer import write_chgcar_to_zarr

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_chgcar_from_json(json_gz_path: Path) -> dict[str, Any]:
    """
    Load CHGCAR data from a compressed JSON file.

    Parameters
    ----------
    json_gz_path : Path
        Path to the .json.gz file containing CHGCAR data

    Returns
    -------
    dict[str, Any]
        Dictionary containing the CHGCAR data structure
    """
    try:
        with gzip.open(json_gz_path, "rt") as f:
            data = json.load(f)
        logger.debug(f"Successfully loaded {json_gz_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading {json_gz_path}: {e}")
        raise


def convert_chgcar_to_zarr(json_gz_path: Path, zarr_path: Path) -> None:
    """
    Convert a single CHGCAR JSON.gz file to Zarr format.

    Parameters
    ----------
    json_gz_path : Path
        Path to the input .json.gz file
    zarr_path : Path
        Path to the output .zarr directory (local filesystem only)

    Notes
    -----
    The Zarr store will contain:
    - /charge_density/total : 3D array of total charge density
    - /charge_density/diff : 3D array of charge density difference (spin polarized)
    - /structure : JSON metadata containing structure information
    - /metadata : Additional metadata (task_id, fs_id, etc.)

    For S3 support, use write_chgcar_to_zarr() directly from zarr_writer module.
    """
    logger.info(f"Converting {json_gz_path} to {zarr_path}")

    # Load the JSON data
    data = load_chgcar_from_json(json_gz_path)

    # Write to zarr using the writer module
    write_chgcar_to_zarr(data, zarr_path)


def convert_directory_to_zarr(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.json.gz",
    max_workers: int | None = None,
) -> tuple[int, int]:
    """
    Convert all CHGCAR JSON.gz files in a directory to Zarr format.

    Parameters
    ----------
    input_dir : Path
        Directory containing .json.gz files
    output_dir : Path
        Directory where .zarr directories will be created
    pattern : str, optional
        Glob pattern to match input files (default: "*.json.gz")
    max_workers : int | None, optional
        Maximum number of parallel workers. If None, uses the number of CPU cores.

    Returns
    -------
    tuple[int, int]
        Number of successfully converted files and number of failed conversions
    """
    input_dir = Path(input_dir).expanduser()
    output_dir = Path(output_dir).expanduser()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all matching files
    input_files = list(input_dir.glob(pattern))
    logger.info(f"Found {len(input_files)} files to convert in {input_dir}")

    if not input_files:
        return 0, 0

    success_count = 0
    failed_count = 0

    # Prepare arguments for parallel processing
    conversion_args = [
        (input_file, output_dir / (input_file.stem.replace(".json", "") + ".zarr"))
        for input_file in input_files
    ]

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(convert_chgcar_to_zarr, input_file, output_path): input_file
            for input_file, output_path in conversion_args
        }

        # Process completed tasks
        for future in as_completed(future_to_file):
            try:
                future.result()
                success_count += 1
            except Exception as e:
                input_file = future_to_file[future]
                logger.error(f"Failed to convert {input_file}: {e}")
                failed_count += 1

    logger.info(
        f"Conversion complete: {success_count} successful, {failed_count} failed"
    )
    return success_count, failed_count


if __name__ == "__main__":
    import fire

    fire.Fire(
        {"convert": convert_chgcar_to_zarr, "convert_dir": convert_directory_to_zarr}
    )
