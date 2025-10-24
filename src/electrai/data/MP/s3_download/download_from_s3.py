#!/usr/bin/env python3
"""
Script to read task IDs from map_sample.json.gz and fetch corresponding files from S3.

This script:
1. Reads the map_sample.json.gz file to get the list of task IDs for a specified key
2. Downloads the corresponding .json.gz files from s3://materialsproject-parsed/chgcars/
3. Saves them to a local directory

Note: No AWS credentials required - the Materials Project S3 bucket is public.
"""

import json
import gzip
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path
import os
import argparse
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_map_sample(file_path: str) -> Dict[str, Any]:
    """Load the map_sample.json.gz file and return the data."""
    try:
        with gzip.open(file_path, 'rt') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded map_sample.json.gz from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading map_sample.json.gz: {e}")
        raise


def get_task_ids(map_data: Dict[str, Any], key: str) -> List[str]:
    """Extract task IDs from the map data for the specified key."""
    if key not in map_data:
        available_keys = list(map_data.keys())
        raise KeyError(f"'{key}' key not found in map_sample.json.gz. Available keys: {available_keys}")
    
    task_ids = map_data[key]
    logger.info(f"Found {len(task_ids)} task IDs for key '{key}'")
    return task_ids


def download_from_s3(task_ids: List[str], bucket_name: str, s3_prefix: str, 
                    local_dir: str = "./downloaded_chgcars") -> None:
    """
    Download files from S3 for the given task IDs.
    
    Args:
        task_ids: List of task IDs to download
        bucket_name: S3 bucket name
        s3_prefix: S3 prefix (folder path)
        local_dir: Local directory to save files
    """
    # Create local directory if it doesn't exist
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize S3 client with no-sign-request for public bucket
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    downloaded_count = 0
    failed_count = 0
    
    for task_id in task_ids:
        s3_key = f"{s3_prefix}/{task_id}.json.gz"
        local_file_path = Path(local_dir) / f"{task_id}.json.gz"
        
        try:
            logger.info(f"Downloading {s3_key} to {local_file_path}")
            s3_client.download_file(bucket_name, s3_key, str(local_file_path))
            downloaded_count += 1
            logger.info(f"Successfully downloaded {task_id}.json.gz")
            
        except Exception as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            failed_count += 1
    
    logger.info(f"Download complete: {downloaded_count} successful, {failed_count} failed")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Materials Project task files from S3 based on map_sample.json.gz"
    )
    parser.add_argument(
        "--key", 
        default="GGA",
        help="Key to extract from map_sample.json.gz (default: GGA)"
    )
    parser.add_argument(
        "--map-file",
        default="../map/map_sample.json.gz",
        help="Path to map_sample.json.gz file (default: ../map/map_sample.json.gz)"
    )
    parser.add_argument(
        "--output-dir",
        default="./downloaded_chgcars",
        help="Local directory to save downloaded files (default: ./downloaded_chgcars)"
    )
    parser.add_argument(
        "--bucket",
        default="materialsproject-parsed",
        help="S3 bucket name (default: materialsproject-parsed)"
    )
    parser.add_argument(
        "--prefix",
        default="chgcars",
        help="S3 prefix/folder path (default: chgcars)"
    )
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="List available keys in map_sample.json.gz and exit"
    )
    
    return parser.parse_args()


def list_available_keys(map_data: Dict[str, Any]) -> None:
    """List all available keys in the map data."""
    print("Available keys in map_sample.json.gz:")
    for key, task_ids in map_data.items():
        print(f"  {key}: {len(task_ids)} task IDs")
        if len(task_ids) <= 10:
            print(f"    Task IDs: {task_ids}")
        else:
            print(f"    Task IDs: {task_ids[:10]}... (and {len(task_ids) - 10} more)")


def main():
    """Main function to orchestrate the download process."""
    args = parse_arguments()
    
    try:
        # Load the map sample data
        logger.info(f"Loading {args.map_file}...")
        map_data = load_map_sample(args.map_file)
        
        # List available keys if requested
        if args.list_keys:
            list_available_keys(map_data)
            return
        
        # Extract task IDs for the specified key
        logger.info(f"Extracting task IDs for key '{args.key}'...")
        task_ids = get_task_ids(map_data, args.key)
        
        # Print the task IDs for verification
        logger.info(f"Task IDs for '{args.key}': {task_ids}")
        
        # Download files from S3
        logger.info(f"Starting download from s3://{args.bucket}/{args.prefix}/...")
        download_from_s3(task_ids, args.bucket, args.prefix, args.output_dir)
        
        logger.info("Script completed successfully!")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise


if __name__ == "__main__":
    main()
