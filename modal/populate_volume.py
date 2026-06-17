"""Populate Modal Volume with training data from S3."""

from __future__ import annotations

import modal

app = modal.App("electrai-populate")
volume = modal.Volume.from_name("electrai-data", create_if_missing=True)


WORKERS = 32
BATCH = 5000  # commit cadence: every BATCH completions


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install("boto3"),
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("oa-electrai-read")],
    timeout=86400,  # 24h: full MP zarr set is ~1.25 TB / ~680K objects
    retries=0,
    cpu=4.0,
    memory=4096,
)
def sync_s3(
    bucket: str = "oa-electrai",
    prefix: str = "mp/chg_datasets",
    dest: str = "/data/mp/chg_datasets",
):
    """Sync dataset from S3 to the electrai-data Volume, parallelized.

    Lists once, then downloads files concurrently through a ThreadPoolExecutor
    (see WORKERS). Commits the Volume every BATCH completions so a preemption
    only loses at most one batch of in-flight work; the rest resumes via the
    size-equal skip in download_one. Per-file errors are logged and counted,
    not fatal.
    """
    import logging
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import boto3
    from botocore.config import Config

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    s3 = boto3.client(
        "s3",
        config=Config(max_pool_connections=WORKERS + 8, retries={"max_attempts": 5}),
    )
    paginator = s3.get_paginator("list_objects_v2")

    objects = [
        obj
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
    ]
    total = len(objects)
    total_bytes = sum(o["Size"] for o in objects)
    log.info("Found %d objects, %.1f GiB total", total, total_bytes / (1024**3))

    def download_one(obj):
        key = obj["Key"]
        rel = key[len(prefix) :].lstrip("/")
        local_path = Path(dest) / rel
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if local_path.exists() and local_path.stat().st_size == obj["Size"]:
                return ("skip", None)
            s3.download_file(bucket, key, str(local_path))
            return ("dl", None)
        except Exception as e:
            return ("err", f"{key}: {e!r}")

    downloaded = skipped = errors = 0
    seen = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        # Process in batches so memory stays bounded and commits are regular.
        for start in range(0, total, BATCH):
            batch = objects[start : start + BATCH]
            futures = [pool.submit(download_one, obj) for obj in batch]
            for future in as_completed(futures):
                kind, msg = future.result()
                seen += 1
                if kind == "dl":
                    downloaded += 1
                elif kind == "skip":
                    skipped += 1
                else:
                    errors += 1
                    if errors <= 20:
                        log.warning("download error: %s", msg)
            volume.commit()
            log.info(
                "Progress: %d/%d (downloaded %d, skipped %d, errors %d) — committed",
                seen,
                total,
                downloaded,
                skipped,
                errors,
            )

    log.info("Done: %d downloaded, %d skipped, %d errors", downloaded, skipped, errors)
    volume.commit()


@app.local_entrypoint()
def main(
    bucket: str = "oa-electrai",
    prefix: str = "mp/chg_datasets",
    dest: str = "/data/mp/chg_datasets",
):
    """Fire-and-forget: spawn sync_s3 and return so the remote run survives any
    local CLI disconnect. Pair with `modal run --detach`; monitor via the Modal
    web UI or `modal app logs <app-id>`.
    """
    import logging

    logging.basicConfig(level=logging.INFO)
    fc = sync_s3.spawn(bucket=bucket, prefix=prefix, dest=dest)
    log = logging.getLogger(__name__)
    log.info("Spawned sync_s3 FunctionCall id=%s", fc.object_id)
    log.info("Monitor at https://modal.com/apps (look for electrai-populate)")
