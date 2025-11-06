"""
Write CHGCAR data to Zarr format (S3 or local filesystem).

This module provides functionality to write already-loaded CHGCAR data
to Zarr format, supporting both local filesystem and S3 storage.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def write_chgcar_to_zarr(
    chgcar_data: dict[str, Any],
    zarr_path: str | Path,
    s3_kwargs: dict[str, Any] | None = None,
    chunks: tuple[int, int, int] = (16, 16, 16),
    write_diff: bool = True,
) -> None:
    """
    Write CHGCAR data to Zarr format (S3 or local filesystem).

    Parameters
    ----------
    chgcar_data : dict[str, Any]
        Dictionary containing CHGCAR data structure. Expected format:
        {
            "data": {
                "data": {
                    "total": {"data": [...]},
                    "diff": {"data": [...]} or None
                },
                "poscar": {
                    "structure": {...}
                },
                "@version": str
            },
            "task_id": str,
            "fs_id": str,
            "maggma_store_type": str
        }
    zarr_path : str | Path
        Path to the output zarr store. Can be:
        - Local path: "/path/to/output.zarr" or Path("/path/to/output.zarr")
        - S3 path: "s3://bucket/prefix/output.zarr"
    s3_kwargs : dict[str, Any] | None, optional
        Additional kwargs for S3 filesystem (e.g., anon=True, profile='default').
        Only used if zarr_path is an S3 path. Default: None
    chunks : tuple[int, int, int], optional
        Chunk size for zarr arrays. Default: (16, 16, 16)
    write_diff : bool, optional
        Whether to write diff charge density data. If False, only total charge
        density will be written. Default: True

    Notes
    -----
    The Zarr store will contain:
    - /charge_density_total : 3D array of total charge density (float32)
    - /charge_density_diff : 3D array of charge density difference (float32, if present and write_diff=True)
    - /attrs/structure : JSON string containing structure information
    - /attrs/metadata : JSON string with task_id, fs_id, and version information

    Examples
    --------
    >>> # Write to local filesystem
    >>> write_chgcar_to_zarr(data, "/path/to/output.zarr")
    >>>
    >>> # Write to S3
    >>> write_chgcar_to_zarr(
    ...     data, "s3://bucket/prefix/output.zarr", s3_kwargs={"anon": True}
    ... )
    >>>
    >>> # Write only total charge density (skip diff)
    >>> write_chgcar_to_zarr(data, "/path/to/output.zarr", write_diff=False)
    """
    zarr_path_str = str(zarr_path)
    use_s3 = zarr_path_str.startswith("s3://")

    logger.info(f"Writing CHGCAR data to {zarr_path_str}")

    # Open zarr group (create new store)
    if use_s3:
        try:
            import s3fs
        except ImportError as e:
            raise ImportError(
                "s3fs is required for S3 access. Install with: pip install s3fs"
            ) from e

        s3_kwargs = s3_kwargs or {}
        s3fs_instance = s3fs.S3FileSystem(**s3_kwargs)
        store = s3fs.S3Map(root=zarr_path_str, s3=s3fs_instance, check=False)
        root = zarr.open_group(store=store, mode="w")
    else:
        # Local filesystem
        root = zarr.open_group(str(zarr_path), mode="w")

    try:
        # Extract charge density data
        chgcar_data_inner = chgcar_data["data"]["data"]

        # Store total charge density
        total_density = np.array(chgcar_data_inner["total"]["data"], dtype=np.float32)
        root.create(name="charge_density_total", data=total_density, chunks=chunks)
        logger.debug(f"Stored total charge density with shape {total_density.shape}")

        # Store diff charge density (if present and write_diff is True)
        if write_diff and chgcar_data_inner.get("diff") is not None:
            diff_density = np.array(chgcar_data_inner["diff"]["data"], dtype=np.float32)
            root.create(name="charge_density_diff", data=diff_density, chunks=chunks)
            logger.debug(f"Stored diff charge density with shape {diff_density.shape}")

        # Store structure information as JSON
        structure_data = chgcar_data["data"]["poscar"]["structure"]
        root.attrs["structure"] = json.dumps(structure_data)

        # Store metadata
        metadata = {
            "task_id": chgcar_data.get("task_id", ""),
            "fs_id": chgcar_data.get("fs_id", ""),
            "maggma_store_type": chgcar_data.get("maggma_store_type", ""),
            "pymatgen_version": chgcar_data["data"].get("@version", ""),
        }
        root.attrs["metadata"] = json.dumps(metadata)

        logger.info(f"Successfully wrote CHGCAR data to {zarr_path_str}")

    except Exception as e:
        logger.error(f"Error writing to {zarr_path_str}: {e}")
        raise
