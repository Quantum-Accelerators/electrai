#!/usr/bin/env python3
"""
Script to read GGA items from map_sample.json.gz and fetch corresponding files from S3.

This script:
1. Reads the map_sample.json.gz file to get the list of GGA task IDs
2. Downloads the corresponding .json.gz files from s3://materialsproject-parsed/chgcars/
3. Saves them to a local directory

Note: No AWS credentials required - the Materials Project S3 bucket is public.
"""

import json
import gzip
import boto3
from pathlib import Path
import os
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


def get_gga_task_ids(map_data: Dict[str, Any]) -> List[str]:
    """Extract GGA task IDs from the map data."""
    if 'GGA' not in map_data:
        raise KeyError("'GGA' key not found in map_sample.json.gz")
    
    gga_ids = map_data['GGA']
    logger.info(f"Found {len(gga_ids)} GGA task IDs")
    return gga_ids


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
    s3_client = boto3.client('s3', config=boto3.session.Config(signature_version=boto3.UNSIGNED))
    
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


def main():
    """Main function to orchestrate the download process."""
    # Configuration
    map_sample_path = "map_sample.json.gz"
    bucket_name = "materialsproject-parsed"
    s3_prefix = "chgcars"
    local_output_dir = "./downloaded_chgcars"
    
    try:
        # Load the map sample data
        logger.info("Loading map_sample.json.gz...")
        map_data = load_map_sample(map_sample_path)
        
        # Extract GGA task IDs
        logger.info("Extracting GGA task IDs...")
        gga_task_ids = get_gga_task_ids(map_data)
        
        # Print the task IDs for verification
        logger.info(f"GGA task IDs: {gga_task_ids}")
        
        # Download files from S3
        logger.info(f"Starting download from s3://{bucket_name}/{s3_prefix}/...")
        download_from_s3(gga_task_ids, bucket_name, s3_prefix, local_output_dir)
        
        logger.info("Script completed successfully!")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise


if __name__ == "__main__":
    main()
