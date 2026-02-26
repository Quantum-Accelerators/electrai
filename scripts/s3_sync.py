#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3", "click"]
# ///
"""Sync training samples from S3 for benchmarking and local development.

Downloads CHGCAR input/label pairs from s3://openathena/electrai/,
filters by file size, and creates a map file for training.
Mirrors S3 structure under data/s3/ by default.

Examples:
    # Download 10 samples (≤25MB each)
    scripts/s3_sync.py

    # Download 50 samples with no size limit
    scripts/s3_sync.py -n 50 -M 0

    # Download to custom directory
    scripts/s3_sync.py -o /tmp/electrai
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import boto3
import click


def err(*a, **kw):
    return print(*a, file=sys.stderr, **kw)  # noqa: T201


# fmt: off
@click.command()
@click.option("-b", "--bucket", default="openathena", help="S3 bucket name")
@click.option("-M", "--max-file-size", default=25.0, type=float, help="Skip files larger than N MB (0=no limit)")
@click.option("-n", "--num-samples", default=10, type=int, help="Max samples to download")
@click.option("-o", "--output", default="data/s3", help="Output directory")
@click.option("-p", "--prefix", default="electrai", help="S3 key prefix")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def main(
    bucket: str,
    max_file_size: float,
    num_samples: int,
    output: str,
    prefix: str,
    verbose: bool,
):
    """Sync training samples from S3.

    Mirrors s3://{bucket}/{prefix}/ under the output directory.
    Writes DATASET_HASH to $GITHUB_OUTPUT when running in CI.
    """
    out = Path(output)
    input_dir = out / "input"
    label_dir = out / "label"
    input_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    max_bytes = int(max_file_size * 1024 * 1024) if max_file_size > 0 else None

    # List input files with sizes
    err(f"Listing s3://{bucket}/{prefix}/input/ ...")
    paginator = s3.get_paginator("list_objects_v2")
    candidates = []
    skipped = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/input/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            size = obj["Size"]
            if not key.endswith(".CHGCAR"):
                continue
            if max_bytes and size > max_bytes:
                name = key.rsplit("/", 1)[-1]
                size_mb = size / 1048576
                if verbose:
                    err(f"  skip {name} ({size_mb:.1f}MB > {max_file_size}MB)")
                skipped += 1
                continue
            candidates.append((key, size))

    size_note = f" ≤{max_file_size}MB" if max_bytes else ""
    err(f"Found {len(candidates)} eligible samples{size_note} ({skipped} skipped)")

    # ListObjectsV2 returns keys in UTF-8 lexicographic order:
    # https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html
    selected = candidates[:num_samples]
    if len(candidates) > num_samples:
        err(f"Using first {num_samples} of {len(candidates)}")

    # Download input + label pairs
    downloaded = 0
    for key, size in selected:
        name = key.rsplit("/", 1)[-1]
        size_mb = size / 1048576
        label_key = f"{prefix}/label/{name}"

        err(f"  {name} ({size_mb:.1f}MB)")
        s3.download_file(bucket, key, str(input_dir / name))
        try:
            s3.download_file(bucket, label_key, str(label_dir / name))
        except Exception as exc:
            err(f"    warning: no label for {name}: {exc}")
            (input_dir / name).unlink(missing_ok=True)
            continue
        downloaded += 1

    err(f"Downloaded {downloaded} samples")

    # Create map file from whatever is in input_dir (supports incremental syncs)
    sample_ids = sorted(p.stem for p in input_dir.glob("*.CHGCAR"))
    map_path = out / "map_subset.json.gz"
    with gzip.open(map_path, "wt") as f:
        json.dump({"GGA": sample_ids}, f)
    err(f"Map: {map_path} ({len(sample_ids)} samples)")

    # Dataset hash: short MD5 of sorted sample IDs
    ds_hash = hashlib.md5("|".join(sample_ids).encode()).hexdigest()[:8]
    err(f"Dataset hash: {ds_hash}")

    # Write to $GITHUB_OUTPUT if in CI
    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with Path(gh_output).open("a") as f:
            f.write(f"DATASET_HASH={ds_hash}\n")

    print(ds_hash)  # noqa: T201


if __name__ == "__main__":
    main()
