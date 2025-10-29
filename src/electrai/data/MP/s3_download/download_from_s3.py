"""
Script to read task IDs from map_sample.json.gz and fetch corresponding files from S3.

This script:
1. Reads the map_sample.json.gz file to get the list of task IDs for a specified key
2. Downloads the corresponding .json.gz files from s3://materialsproject-parsed/chgcars/
3. Saves them to a local directory

Note: No AWS credentials required - the Materials Project S3 bucket is public.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import fire
from botocore import UNSIGNED
from botocore.config import Config
from mp_api.client import MPRester

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_MP_VERSION = "2025.09.25"


def load_map_sample(file_path: str) -> dict[str, Any]:
    """Load the map_sample.json.gz file and return the data."""
    try:
        with gzip.open(file_path, "rt") as f:
            data = json.load(f)
        logger.info(f"Successfully loaded map_sample.json.gz from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading map_sample.json.gz: {e}")
        raise


def get_task_ids(map_data: dict[str, Any], key: str) -> list[str]:
    """Extract task IDs from the map data for the specified key."""
    if key not in map_data:
        available_keys = list(map_data.keys())
        raise KeyError(
            f"'{key}' key not found in map_sample.json.gz. Available keys: {available_keys}"
        )

    task_ids = map_data[key]
    logger.info(f"Found {len(task_ids)} task IDs for key '{key}'")
    return task_ids


def download_single_file(
    s3_client, task_id: str, bucket_name: str, s3_prefix: str, local_dir: Path
) -> tuple[str, bool]:
    """
    Download a single file from S3.

    Args:
        s3_client: Boto3 S3 client
        task_id: Task ID to download
        bucket_name: S3 bucket name
        s3_prefix: S3 prefix (folder path)
        local_dir: Local directory to save file

    Returns:
        Tuple of (task_id, success_flag)
    """
    s3_key = f"{s3_prefix}/{task_id}.json.gz"
    local_file_path = local_dir / f"{task_id}.json.gz"

    try:
        logger.info(f"Downloading {s3_key} to {local_file_path}")
        s3_client.download_file(bucket_name, s3_key, str(local_file_path))
        logger.info(f"Successfully downloaded {task_id}.json.gz")
        return task_id, True
    except Exception as e:
        logger.error(f"Failed to download {s3_key}: {e}")
        return task_id, False


def download_from_s3(
    task_ids: list[str],
    bucket_name: str,
    s3_prefix: str,
    local_dir: str,
    max_workers: int = 10,
) -> tuple[int, int]:
    """
    Download files from S3 for the given task IDs using parallel processing.

    Args:
        task_ids: List of task IDs to download
        bucket_name: S3 bucket name
        s3_prefix: S3 prefix (folder path)
        local_dir: Local directory to save files
        max_workers: Maximum number of worker threads (default: 10)

    Returns:
        Tuple of (downloaded_count, failed_count)
    """
    local_dir_path = Path(local_dir).expanduser()

    # Create local directory if it doesn't exist
    local_dir_path.mkdir(parents=True, exist_ok=True)

    # Initialize S3 client with no-sign-request for public bucket
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    downloaded_count = 0
    failed_count = 0

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_task = {
            executor.submit(
                download_single_file,
                s3_client,
                task_id,
                bucket_name,
                s3_prefix,
                local_dir_path,
            ): task_id
            for task_id in task_ids
        }

        # Process completed downloads
        for future in as_completed(future_to_task):
            _, success = future.result()
            if success:
                downloaded_count += 1
            else:
                failed_count += 1

    logger.info(
        f"Download complete: {downloaded_count} successful, {failed_count} failed"
    )
    return downloaded_count, failed_count


def lookup_mp_version() -> str:
    """Lookup the Materials Project version."""

    MP_API_KEY = os.environ.get("MP_API_KEY", None)

    if not MP_API_KEY:
        logger.warning(
            f"MP_API_KEY is not set, using default version {DEFAULT_MP_VERSION}"
        )
        return DEFAULT_MP_VERSION

    with MPRester(MP_API_KEY) as mpr:
        return mpr.get_database_version() or DEFAULT_MP_VERSION


def write_metadata(
    local_dir: Path,
    mp_version: str,
    key: str,
    bucket: str,
    prefix: str,
    task_ids: list[str],
    downloaded_count: int,
    failed_count: int,
) -> None:
    """
    Write metadata about the download to a JSON file.

    Args:
        local_dir: Directory where files were downloaded
        mp_version: Materials Project database version
        key: The functional key (e.g., "GGA")
        bucket: S3 bucket name
        prefix: S3 prefix
        task_ids: List of task IDs that were requested
        downloaded_count: Number of successfully downloaded files
        failed_count: Number of failed downloads
    """
    metadata = {
        "mp_version": mp_version,
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "key": key,
        "s3_bucket": bucket,
        "s3_prefix": prefix,
        "total_task_ids": len(task_ids),
        "downloaded_count": downloaded_count,
        "failed_count": failed_count,
        "task_ids": task_ids,
    }

    metadata_path = local_dir / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata written to {metadata_path}")


class S3Downloader:
    """Download Materials Project task files from S3 based on map_sample.json.gz."""

    def download(
        self,
        key: str = "GGA",
        map_file: str = "../map/chgcars_functional_to_task_ids.json.gz",
        output_dir: str = "~/data/MP/downloaded_chgcars",
        bucket: str = "materialsproject-parsed",
        prefix: str = "chgcars",
        max_workers: int = 10,
    ):
        """
        Download files from S3 for a specified key.

        Args:
            key: Key to extract from map_sample.json.gz (default: GGA)
            map_file: Path to chgcars_functional_to_task_ids.json.gz (default: ../map/chgcars_functional_to_task_ids.json.gz)
            output_dir: Local directory to save downloaded files (default: ~/data/MP/downloaded_chgcars)
            bucket: S3 bucket name (default: materialsproject-parsed)
            prefix: S3 prefix/folder path (default: chgcars)
            max_workers: Maximum number of worker threads for parallel downloads (default: 10)
        """
        try:
            # Load the map sample data
            logger.info(f"Loading {map_file}...")
            map_data = load_map_sample(map_file)

            # Extract task IDs for the specified key
            logger.info(f"Extracting task IDs for key '{key}'...")
            task_ids = get_task_ids(map_data, key)

            # Print the task IDs for verification
            logger.info(f"Task IDs for '{key}': {task_ids}")

            version = lookup_mp_version()
            logger.info(f"Materials Project version: {version}")

            # Download files from S3
            download_dir = f"{output_dir}/{key}/{version}"
            logger.info(f"Starting download from s3://{bucket}/{prefix}/...")
            downloaded_count, failed_count = download_from_s3(
                task_ids, bucket, prefix, download_dir, max_workers
            )

            # Write metadata
            local_dir_path = Path(download_dir).expanduser()
            write_metadata(
                local_dir_path,
                version,
                key,
                bucket,
                prefix,
                task_ids,
                downloaded_count,
                failed_count,
            )

            logger.info("Script completed successfully!")

        except Exception as e:
            logger.error(f"Script failed: {e}")
            raise

    def list_keys(
        self, map_file: str = "../map/chgcars_functional_to_task_ids.json.gz"
    ):
        """
        List available keys in map_sample.json.gz.

        Args:
            map_file: Path to chgcars_functional_to_task_ids.json.gz (default: ../map/chgcars_functional_to_task_ids.json.gz)
        """
        try:
            logger.info(f"Loading {map_file}...")
            map_data = load_map_sample(map_file)

            logger.info("Available keys in map_sample.json.gz:")
            for key, task_ids in map_data.items():
                logger.info(f"  {key}: {len(task_ids)} task IDs")
                if len(task_ids) <= 10:
                    logger.info(f"    Task IDs: {task_ids}")
                else:
                    logger.info(
                        f"    Task IDs: {task_ids[:10]}... (and {len(task_ids) - 10} more)"
                    )

        except Exception as e:
            logger.error(f"Failed to list keys: {e}")
            raise


if __name__ == "__main__":
    fire.Fire(S3Downloader)
