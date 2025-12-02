"""
Convert Materials Project CHGCAR data from JSON.gz or native CHGCAR files to Zarr.

This module provides functions to convert CHGCAR charge density data stored in
compressed JSON exports or raw .CHGCAR files to the Zarr format for efficient
storage and access.
"""

from __future__ import annotations

import gzip
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pymatgen.io.vasp.outputs import Chgcar

from .zarr_writer import write_chgcar_to_zarr

logger = logging.getLogger(__name__)


def _derive_task_id(path: Path) -> str:
    name = path.name
    if name.endswith(".json.gz"):
        return name[: -len(".json.gz")]
    if name.lower().endswith(".chgcar"):
        return name[: -len(".chgcar")]
    return path.stem


def _apply_default_ids(chgcar: Chgcar, source_path: Path) -> None:
    derived_id = _derive_task_id(source_path)
    if not getattr(chgcar, "task_id", None):
        chgcar.task_id = derived_id


def load_chgcar_from_json(json_gz_path: Path) -> Chgcar:
    """
    Load CHGCAR data from a compressed JSON file.

    Parameters
    ----------
    json_gz_path : Path
        Path to the .json.gz file containing CHGCAR data

    Returns
    -------
    Chgcar
        Pymatgen Chgcar object reconstructed from the JSON payload.
    """
    try:
        with gzip.open(json_gz_path, "rt") as f:
            payload = json.load(f)
        logger.debug(f"Successfully loaded {json_gz_path}")
    except Exception as exc:
        logger.error(f"Error loading {json_gz_path}: {exc}")
        raise

    chgcar_dict: dict[str, Any]
    if isinstance(payload, dict):
        # Check for 'data' field first (common in Materials Project exports)
        if "data" in payload and isinstance(payload["data"], dict):
            chgcar_dict = payload["data"]
        # Check for 'chgcar' field (alternative structure)
        elif "chgcar" in payload:
            chgcar_dict = payload["chgcar"]
        # If neither exists, assume the payload itself is the chgcar dict
        # but filter out non-Chgcar fields that might cause errors
        else:
            # Filter out metadata fields that are not part of Chgcar serialization
            # Chgcar serialization typically has @module, @class, @version, poscar, data, data_aug
            excluded_fields = {"fs_id", "maggma_store_type", "compression", "task_id"}
            chgcar_dict = {k: v for k, v in payload.items() if k not in excluded_fields}
    else:
        raise ValueError(f"Unexpected JSON structure in {json_gz_path}")

    chgcar = Chgcar.from_dict(chgcar_dict)
    _apply_default_ids(chgcar, json_gz_path)
    return chgcar


def load_chgcar(chgcar_path: Path) -> Chgcar:
    """
    Load CHGCAR data from either a native .CHGCAR file or a .json.gz export.
    """
    chgcar_path = Path(chgcar_path).expanduser()
    suffixes = chgcar_path.suffixes

    if len(suffixes) >= 2 and suffixes[-2:] == [".json", ".gz"]:
        chgcar = load_chgcar_from_json(chgcar_path)
    elif chgcar_path.suffix.lower() == ".chgcar":
        chgcar = Chgcar.from_file(str(chgcar_path))
        _apply_default_ids(chgcar, chgcar_path)
    else:
        raise ValueError(
            f"Unsupported file extension for {chgcar_path}. Expected '.json.gz' or '.CHGCAR'."
        )

    return chgcar


def convert_chgcar_to_zarr(
    input_path: Path, zarr_path: Path, write_diff: bool = False
) -> None:
    """
    Convert a single CHGCAR file (JSON.gz export or native CHGCAR) to Zarr format.

    Parameters
    ----------
    input_path : Path
        Path to the input .json.gz or .CHGCAR file
    zarr_path : Path
        Path to the output .zarr directory (local filesystem only)
    write_diff : bool, optional
        Whether to write diff charge density data. If False, only total charge
        density will be written. Default: False

    Notes
    -----
    The Zarr store will contain:
    - /charge_density_total : 3D array of total charge density
    - /charge_density_diff : 3D array of charge density difference (spin polarized, if write_diff=True)
    - /structure : JSON metadata containing structure information
    - /metadata : Additional metadata (task_id, version, etc.)

    For S3 support, use write_chgcar_to_zarr() directly from zarr_writer module.
    """
    logger.info(f"Converting {input_path} to {zarr_path}")

    # Load the CHGCAR data
    chgcar = load_chgcar(input_path)

    # Write to zarr using the writer module
    write_chgcar_to_zarr(chgcar, zarr_path, write_diff=write_diff)


def convert_directory_to_zarr(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.json.gz",
    max_workers: int | None = None,
    write_diff: bool = False,
) -> tuple[int, int]:
    """
    Convert all CHGCAR files in a directory to Zarr format.

    Parameters
    ----------
    input_dir : Path
        Directory containing .json.gz or .CHGCAR files
    output_dir : Path
        Directory where .zarr directories will be created
    pattern : str, optional
        Glob pattern to match input files (default: "*.json.gz")
    max_workers : int | None, optional
        Maximum number of parallel workers. If None, uses the number of CPU cores.
    write_diff : bool, optional
        Whether to write diff charge density data. If False, only total charge
        density will be written. Default: False

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
        (input_file, output_dir / f"{_derive_task_id(input_file)}.zarr", write_diff)
        for input_file in input_files
    ]

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                convert_chgcar_to_zarr, input_file, output_path, write_diff
            ): input_file
            for input_file, output_path, write_diff in conversion_args
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
