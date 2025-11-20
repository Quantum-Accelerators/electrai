from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from sklearn.model_selection import train_test_split

from .registry import register_data

logger = logging.getLogger(__name__)


class ZarrS3Reader:
    """
    Reader for CHGCAR data stored in Zarr format on S3 or local filesystem.

    This reader supports lazy loading - it stores references to zarr stores
    rather than loading all data into memory at once.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing input zarr stores (can be S3 path like 's3://bucket/prefix')
    label_dir : str or Path
        Directory containing label zarr stores
    map_dir : str or Path
        Path to JSON file mapping functional to list of task_ids
    functional : str
        Functional type: 'GGA', 'GGA+U', 'PBEsol', 'SCAN', 'r2SCAN'
    normalize : bool
        Whether to normalize the charge density data
    train_fraction : float
        Fraction of data to use for training (0 to 1)
    random_state : int, optional
        Random seed for train/test splitting (default: 42)
    s3_kwargs : dict, optional
        Additional kwargs for S3 filesystem (e.g., anon=True, profile='default')
    """

    def __init__(
        self,
        data_dir: str | Path,
        label_dir: str | Path,
        map_dir: str | Path,
        functional: str,
        normalize: bool,
        train_fraction: float,
        random_state: int = 42,
        density_type: str = "total",
        s3_kwargs: dict[str, Any] | None = None,
    ):
        self.data_dir = (
            Path(data_dir) if not str(data_dir).startswith("s3://") else str(data_dir)
        )
        self.label_dir = (
            Path(label_dir)
            if not str(label_dir).startswith("s3://")
            else str(label_dir)
        )
        self.map_dir = Path(map_dir)
        self.functional = functional
        self.normalize = normalize
        self.tf = train_fraction
        self.rs = random_state
        self.s3_kwargs = s3_kwargs or {}
        self.density_type = density_type
        # Determine if we're using S3
        self.use_s3 = str(data_dir).startswith("s3://") or str(label_dir).startswith(
            "s3://"
        )

        if self.use_s3:
            try:
                import s3fs
            except ImportError as e:
                raise ImportError(
                    "s3fs is required for S3 access. Install with: pip install s3fs"
                ) from e

            self.s3fs = s3fs.S3FileSystem(**self.s3_kwargs)
            logger.info("Initialized S3 filesystem for zarr stores")

    def _get_zarr_path(self, base_dir: str, task_id: str) -> str:
        """
        Construct the path to a zarr store for a given task_id.

        Parameters
        ----------
        base_dir : str
            Base directory (local or S3)
        task_id : str
            Materials Project task ID

        Returns
        -------
        str
            Full path to the zarr store
        """
        return f"{base_dir}/{task_id}.zarr"

    def read_zarr_store(self, zarr_path: str) -> zarr.Group:
        """
        Open a zarr store from S3 or local filesystem.

        Parameters
        ----------
        zarr_path : str
            Path to the zarr store

        Returns
        -------
        zarr.Group
            Opened zarr group (lazy - data not loaded into memory)
        """
        if self.use_s3 and zarr_path.startswith("s3://"):
            # Open from S3 using s3fs
            import s3fs

            store = s3fs.S3Map(root=zarr_path, s3=self.s3fs, check=False)
            return zarr.open_group(store=store, mode="r")
        else:
            # Open from local filesystem
            return zarr.open_group(zarr_path, mode="r")

    def read_charge_density(
        self, zarr_store: zarr.Group
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        """
        Extract charge density array from zarr store.

        Parameters
        ----------
        zarr_store : zarr.Group
            Opened zarr group

        Returns
        -------
        tuple[np.ndarray, tuple[int, int, int]]
            Flattened charge density array and gridsize tuple
        """
        array_name = f"charge_density_{self.density_type}"

        if array_name not in zarr_store:
            raise ValueError(
                f"'{array_name}' not found in zarr store. "
                f"Available arrays: {list(zarr_store.array_keys())}"
            )

        # Access the array (still lazy - not loaded yet)
        charge_array = zarr_store[array_name]
        gridsize = charge_array.shape

        # Load the actual data into memory
        charge_data = np.array(charge_array, dtype=np.float32)

        # Normalize if requested
        if self.normalize:
            charge_data /= np.prod(gridsize)

        return charge_data.flatten(), gridsize

    def get_metadata(self, zarr_store: zarr.Group) -> dict[str, Any]:
        """
        Extract metadata from zarr store.

        Parameters
        ----------
        zarr_store : zarr.Group
            Opened zarr group

        Returns
        -------
        dict[str, Any]
            Metadata dictionary containing structure and task information
        """
        import json

        metadata = {}

        # Get metadata from attributes
        if "metadata" in zarr_store.attrs:
            metadata.update(json.loads(zarr_store.attrs["metadata"]))

        if "structure" in zarr_store.attrs:
            metadata["structure"] = json.loads(zarr_store.attrs["structure"])

        return metadata

    def data_split(
        self,
    ) -> tuple[tuple[list, list, list, list], tuple[list, list, list, list]]:
        """
        Load dataset and split into train/test sets.

        The returned lists contain actual numpy arrays (loaded into memory) to
        maintain compatibility with the existing RhoData dataset class. For
        lazy loading, should be updated to use ZarrDataset class instead.

        Parameters
        ----------
        Returns
        -------
        tuple[tuple[list, list, list, list], tuple[list, list, list, list]]
            (train_sets, test_sets) where each set is:
            (data_list, label_list, gridsize_data_list, gridsize_label_list)
        """
        from monty.serialization import loadfn

        # Load the task_id mapping
        mapping = loadfn(self.map_dir)

        if self.functional not in mapping:
            raise ValueError(
                f"Functional '{self.functional}' not found in mapping. "
                f"Available: {list(mapping.keys())}"
            )

        task_ids = mapping[self.functional]
        num_tasks = len(task_ids)
        logger.info(f"Found {num_tasks} task_ids for functional {self.functional}")

        data_list = []
        label_list = []
        gs_data_list = []
        gs_label_list = []

        # Load data for each task_id
        for i, task_id in enumerate(task_ids):
            try:
                # Construct paths
                data_path = self._get_zarr_path(str(self.data_dir), task_id)
                label_path = self._get_zarr_path(str(self.label_dir), task_id)

                # Open zarr stores and read data
                data_store = self.read_zarr_store(data_path)
                label_store = self.read_zarr_store(label_path)

                # Extract charge densities
                data, gs_data = self.read_charge_density(data_store)
                label, gs_label = self.read_charge_density(label_store)

                data_list.append(data)
                label_list.append(label)
                gs_data_list.append(gs_data)
                gs_label_list.append(gs_label)

                if (i + 1) % 100 == 0:
                    logger.info(f"Loaded {i + 1}/{num_tasks} samples")

            except Exception as e:
                logger.warning(f"Failed to load task_id {task_id}: {e}")
                continue

        logger.info(f"Successfully loaded {len(data_list)} samples")

        # Split into train/test
        splits = train_test_split(
            data_list,
            label_list,
            gs_data_list,
            gs_label_list,
            train_size=self.tf,
            random_state=self.rs,
        )

        train_sets = splits[::2]
        test_sets = splits[1::2]

        return train_sets, test_sets


@register_data("mp_zarr_s3_data")
def load_data(cfg):
    """
    Load CHGCAR data from Zarr format (S3 or local).

    Parameters
    ----------
    cfg : SimpleNamespace or dict
        Configuration object with the following attributes:
        - data_dir: Directory containing input zarr stores
        - label_dir: Directory containing label zarr stores
        - map_dir: Path to task_id mapping JSON
        - functional: Functional type ('GGA', 'GGA+U', etc.)
        - normalize_data: Whether to normalize charge density
        - train_fraction: Fraction for training (0-1)
        - random_state: Random seed
        - density_type: (optional) 'total' or 'diff', defaults to 'total'
        - s3_kwargs: (optional) Dict of S3 connection parameters

    Returns
    -------
    tuple
        (train_sets, test_sets) compatible with RhoData dataset
    """
    reader = ZarrS3Reader(
        data_dir=cfg.data_dir,
        label_dir=cfg.label_dir,
        map_dir=cfg.map_dir,
        functional=cfg.functional,
        normalize=cfg.normalize_data,
        train_fraction=cfg.train_fraction,
        random_state=cfg.random_state,
        density_type=getattr(cfg, "density_type", "total"),
        s3_kwargs=getattr(cfg, "s3_kwargs", None),
    )

    return reader.data_split()
