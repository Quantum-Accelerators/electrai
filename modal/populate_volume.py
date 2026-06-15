"""Populate Modal Volume with training data from S3."""

from __future__ import annotations

import modal

app = modal.App("electrai-populate")
volume = modal.Volume.from_name("electrai-data", create_if_missing=True)


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install("boto3"),
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("oa-electrai-read")],
    timeout=86400,  # 24h: full MP zarr set is ~1.25 TB / ~680K objects
    retries=0,
)
def sync_s3(
    bucket: str = "oa-electrai",
    prefix: str = "mp/chg_datasets",
    dest: str = "/data/mp/chg_datasets",
):
    """Sync dataset from S3 to Modal Volume."""
    import logging
    from pathlib import Path

    import boto3

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    # Count and list all objects
    objects = [
        obj
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
    ]

    total_bytes = sum(o["Size"] for o in objects)
    log.info("Found %d objects, %.1f GiB total", len(objects), total_bytes / (1024**3))

    downloaded = 0
    skipped = 0
    for i, obj in enumerate(objects):
        key = obj["Key"]
        rel = key[len(prefix) :].lstrip("/")
        local_path = Path(dest) / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already exists with same size
        if local_path.exists() and local_path.stat().st_size == obj["Size"]:
            skipped += 1
            continue

        s3.download_file(bucket, key, str(local_path))
        downloaded += 1

        if (i + 1) % 100 == 0:
            log.info(
                "Progress: %d/%d (downloaded %d, skipped %d)",
                i + 1,
                len(objects),
                downloaded,
                skipped,
            )

    log.info("Done: %d downloaded, %d skipped (already existed)", downloaded, skipped)
    volume.commit()


@app.local_entrypoint()
def main(
    bucket: str = "oa-electrai",
    prefix: str = "mp/chg_datasets",
    dest: str = "/data/mp/chg_datasets",
):
    import logging

    logging.basicConfig(level=logging.INFO)
    sync_s3.remote(bucket=bucket, prefix=prefix, dest=dest)
    logging.getLogger(__name__).info("Volume populated.")
