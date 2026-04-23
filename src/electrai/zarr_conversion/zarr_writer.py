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

    from pymatgen.io.vasp.outputs import Chgcar

logger = logging.getLogger(__name__)

# Keys we recognize in chgcar_data.data. "total" is handled separately; the
# rest each get their own zarr array chunked independently of total, since
# downstream training typically loads either total or a diff component per
# batch rather than both together.
_DIFF_KEYS: tuple[str, ...] = ("diff", "diff_x", "diff_y", "diff_z")


def _array_name(density_key: str) -> str:
    return f"charge_density_{density_key}"


def _normalize_data_aug(data_aug: dict[str, Any] | None) -> dict[str, list[str]]:
    """Coerce pymatgen's data_aug (dict of list[str] or None) to JSON-safe form."""
    if not data_aug:
        return {}
    normalized: dict[str, list[str]] = {}
    for key, value in data_aug.items():
        if value is None:
            continue
        normalized[key] = [str(line) for line in value]
    return normalized


def write_chgcar_to_zarr(
    chgcar_data: Chgcar,
    zarr_path: str | Path,
    s3_kwargs: dict[str, Any] | None = None,
    chunks: tuple[int, int, int] = (16, 16, 16),
    chunks_diff: tuple[int, int, int] | None = None,
) -> None:
    """
    Write CHGCAR data to Zarr format (S3 or local filesystem).

    All CHGCAR content is preserved: total charge density, spin-polarized or
    non-collinear diff components, PAW augmentation occupancies, the POSCAR
    comment line, and the structure.

    Parameters
    ----------
    chgcar_data : Chgcar
        Pymatgen Chgcar object containing CHGCAR data.
    zarr_path : str | Path
        Path to the output zarr store. Can be:
        - Local path: "/path/to/output.zarr" or Path("/path/to/output.zarr")
        - S3 path: "s3://bucket/prefix/output.zarr"
    s3_kwargs : dict[str, Any] | None, optional
        Additional kwargs for S3 filesystem (e.g., anon=True, profile='default').
        Only used if zarr_path is an S3 path. Default: None
    chunks : tuple[int, int, int], optional
        Chunk size for the total charge density array. Default: (16, 16, 16)
    chunks_diff : tuple[int, int, int] | None, optional
        Chunk size for diff / diff_x / diff_y / diff_z arrays. Stored under
        independent chunks from total because downstream training typically
        loads only one of total or diff per batch. Defaults to ``chunks``.

    Notes
    -----
    The Zarr store will contain:
    - /charge_density_total : 3D float32 array of total charge density
    - /charge_density_diff : 3D float32 array of magnetization density
      (spin-polarized calculations only)
    - /charge_density_diff_x, _y, _z : 3D float32 arrays of non-collinear
      magnetization components (SOC calculations only)
    - /attrs/structure : JSON-encoded pymatgen Structure
    - /attrs/metadata : JSON-encoded task_id and version info
    - /attrs/data_aug : JSON-encoded dict of PAW augmentation occupancy lines,
      keyed by density component ("total", "diff", ...)
    - /attrs/poscar_comment : POSCAR header/comment string (may be null)
    - /attrs/is_spin_polarized, /attrs/is_soc : bool flags
    """
    zarr_path_str = str(zarr_path)
    use_s3 = zarr_path_str.startswith("s3://")

    logger.info(f"Writing CHGCAR data to {zarr_path_str}")

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
        root = zarr.open_group(str(zarr_path), mode="w")

    diff_chunks = chunks_diff if chunks_diff is not None else chunks

    try:
        charge_data = chgcar_data.data

        total_density = np.asarray(charge_data["total"], dtype=np.float32)
        root.create(name=_array_name("total"), data=total_density, chunks=chunks)
        logger.debug(f"Stored total charge density with shape {total_density.shape}")

        for diff_key in _DIFF_KEYS:
            diff_raw = charge_data.get(diff_key)
            if diff_raw is None:
                continue
            diff_density = np.asarray(diff_raw, dtype=np.float32)
            root.create(
                name=_array_name(diff_key), data=diff_density, chunks=diff_chunks
            )
            logger.debug(
                f"Stored {diff_key} charge density with shape {diff_density.shape}"
            )

        root.attrs["structure"] = json.dumps(chgcar_data.structure.as_dict())

        metadata = {
            "task_id": getattr(chgcar_data, "task_id", ""),
            "pymatgen_version": getattr(chgcar_data, "source_version", ""),
        }
        root.attrs["metadata"] = json.dumps(metadata)

        root.attrs["data_aug"] = json.dumps(
            _normalize_data_aug(getattr(chgcar_data, "data_aug", None))
        )
        root.attrs["poscar_comment"] = getattr(chgcar_data, "name", None)
        root.attrs["is_spin_polarized"] = bool(
            getattr(chgcar_data, "is_spin_polarized", "diff" in charge_data)
        )
        root.attrs["is_soc"] = bool(getattr(chgcar_data, "is_soc", False))

        logger.info(f"Successfully wrote CHGCAR data to {zarr_path_str}")

    except Exception as e:
        logger.error(f"Error writing to {zarr_path_str}: {e}")
        raise
