"""Populate Modal Volume with training data from S3, packing zarr stores.

Modal Volumes are capped at ~500K inodes per volume. Our zarr v3 layout has
~8 inodes per `<id>.zarr/` (3 files + 5 directory entries) times ~226K stores
is ~1.8M inodes — well over the cap.

To fit, each S3-side `<prefix>/.../<id>.zarr/{zarr.json,
charge_density_total/zarr.json, charge_density_total/c/0/0/0}` is packed into
a single `<prefix>/.../<id>.zarr.zip` on the Volume (zarr's `ZipStore` reads
it as a normal store). Non-zarr S3 keys (filelists, split files) pass through
unchanged. Net: ~226K + a few standalone files ≈ ~230K inodes — well under
the cap.

The matching loader change (auto-detect `.zarr.zip` via ZipStore) lives in
`src/electrai/dataloader/utils.py:load_zarr`.
"""

from __future__ import annotations

import modal

app = modal.App("electrai-populate")
volume = modal.Volume.from_name("electrai-data", create_if_missing=True)


WORKERS = 32
BATCH = 2000  # commit every BATCH item completions (stores or standalones)


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install("boto3"),
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("oa-electrai-read")],
    timeout=86400,  # 24h
    retries=0,
    cpu=4.0,
    memory=4096,
)
def sync_s3(
    bucket: str = "oa-electrai",
    prefix: str = "mp/chg_datasets",
    dest: str = "/data/mp/chg_datasets",
    wipe_first: bool = False,
):
    """Sync S3 -> Volume, packing each `<id>.zarr/` into `<id>.zarr.zip`.

    If `wipe_first=True`, recursively removes `dest` before populating to free
    inodes — required when re-populating after a previous unpacked attempt.
    """
    import logging
    import shutil
    import zipfile
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

    if wipe_first:
        target = Path(dest)
        if target.exists():
            log.info("Wiping %s (this can take a few minutes for many inodes)…", target)
            shutil.rmtree(target)
            volume.commit()
            log.info("Wipe complete; committed.")

    # ---- list S3 ---------------------------------------------------------
    paginator = s3.get_paginator("list_objects_v2")
    objects = [
        obj
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
    ]
    total_bytes = sum(o["Size"] for o in objects)
    log.info("Listed %d S3 objects, %.1f GiB", len(objects), total_bytes / (1024**3))

    # ---- group by .zarr/ store ------------------------------------------
    # Each store-key is the S3 prefix up to and including `.zarr`. Inner name
    # within the zip is the remainder after `.zarr/`.
    stores: dict[str, list[tuple[dict, str]]] = {}
    standalones: list[dict] = []
    for obj in objects:
        key = obj["Key"]
        if ".zarr/" in key:
            store_key, inner = key.split(".zarr/", 1)
            store_key = store_key + ".zarr"
            stores.setdefault(store_key, []).append((obj, inner))
        else:
            standalones.append(obj)
    log.info(
        "Grouping: %d zarr stores to pack, %d standalone files",
        len(stores),
        len(standalones),
    )

    # ---- error log on volume --------------------------------------------
    err_log_path = Path(dest).parent / "_populate_errors.log"
    err_log_path.parent.mkdir(parents=True, exist_ok=True)
    err_log = err_log_path.open("a")
    err_log.write(f"\n=== run start {bucket}/{prefix} (packed) ===\n")
    err_log.flush()

    # ---- workers --------------------------------------------------------
    def pack_store(store_key: str, parts: list[tuple[dict, str]]):
        rel = store_key[len(prefix) :].lstrip("/")
        zip_path = Path(dest) / (rel + ".zip")
        if zip_path.exists() and zip_path.stat().st_size > 0:
            return ("skip", None)
        try:
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = zip_path.with_name(zip_path.name + ".tmp")
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
                for obj, inner in parts:
                    body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                    z.writestr(inner, body)
            tmp.rename(zip_path)
            return ("dl", None)
        except Exception as e:
            return ("err", f"{store_key}: {e!r}")

    def copy_standalone(obj: dict):
        key = obj["Key"]
        rel = key[len(prefix) :].lstrip("/")
        local_path = Path(dest) / rel
        if local_path.exists() and local_path.stat().st_size == obj["Size"]:
            return ("skip", None)
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = local_path.with_name(local_path.name + ".tmp")
            with tmp.open("wb") as f:
                s3.download_fileobj(bucket, key, f)
            tmp.rename(local_path)
            return ("dl", None)
        except Exception as e:
            return ("err", f"{key}: {e!r}")

    # ---- process: stores first (the bulk), then standalones --------------
    work: list[tuple[str, object]] = [("store", item) for item in stores.items()] + [
        ("standalone", obj) for obj in standalones
    ]
    total = len(work)
    log.info("Total work items: %d", total)

    packed = skipped = errors = seen = 0
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for start in range(0, total, BATCH):
                batch = work[start : start + BATCH]
                futures = []
                for kind, payload in batch:
                    if kind == "store":
                        store_key, parts = payload
                        futures.append(pool.submit(pack_store, store_key, parts))
                    else:
                        futures.append(pool.submit(copy_standalone, payload))
                for future in as_completed(futures):
                    res, msg = future.result()
                    seen += 1
                    if res == "dl":
                        # pack_store returns "dl" too; we count both as "packed"
                        # for stores and "copied" for standalones; cheaper to
                        # just bucket together.
                        packed += 1
                    elif res == "skip":
                        skipped += 1
                    else:
                        errors += 1
                        err_log.write(msg + "\n")
                        if errors <= 50:
                            log.warning("error: %s", msg)
                err_log.flush()
                volume.commit()
                log.info(
                    "Progress: %d/%d (done %d, skipped %d, errors %d) — committed",
                    seen,
                    total,
                    packed,
                    skipped,
                    errors,
                )
    finally:
        err_log.close()

    log.info("Done: %d packed/copied, %d skipped, %d errors", packed, skipped, errors)
    volume.commit()


@app.local_entrypoint()
def main(
    bucket: str = "oa-electrai",
    prefix: str = "mp/chg_datasets",
    dest: str = "/data/mp/chg_datasets",
    wipe_first: bool = False,
):
    """Fire-and-forget: spawn sync_s3 and return so the remote run survives any
    local CLI disconnect. Pair with `modal run --detach`; monitor via the Modal
    web UI or `modal app logs <app-id>`.

    Pass `--wipe-first` on the first packed run to clear any unpacked partial
    state from previous attempts (frees inodes).
    """
    import logging

    logging.basicConfig(level=logging.INFO)
    fc = sync_s3.spawn(bucket=bucket, prefix=prefix, dest=dest, wipe_first=wipe_first)
    log = logging.getLogger(__name__)
    log.info(
        "Spawned sync_s3 FunctionCall id=%s (wipe_first=%s)", fc.object_id, wipe_first
    )
    log.info("Monitor at https://modal.com/apps (look for electrai-populate)")
