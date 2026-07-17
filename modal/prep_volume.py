"""Prepare the electrai-data Volume after a Globus transfer.

The Globus transfer lands the real data under .../rho_gga and .../rho_gga+u plus
the small per-functional metadata (mp_filelist.txt, split.json). This helper:

  1. (Re)creates the data/label symlinks the loader expects under
     functionals/{gga,gga+u}/ pointing at the matching rho_* dirs.
  2. Builds subset smoke filelists (mp_filelist_smoke.txt) used by
     config_gga_gga+u_f32_smoke.yaml.
  3. Sanity-checks that the first id of each filelist resolves to a real .zarr.
  4. Commits the Volume explicitly (no reliance on shell auto-commit).

Idempotent: if Globus already landed real data/label dirs (i.e. it followed the
symlinks), those are left untouched.

Usage:
    modal run modal/prep_volume.py            # 200-sample smoke filelists
    modal run modal/prep_volume.py --smoke-n 50
"""

from __future__ import annotations

import modal

app = modal.App("electrai-prep")

data_volume = modal.Volume.from_name("electrai-data", create_if_missing=True)

VOLUME_ROOT = "/data"
# functional dir name -> real data dir name (both under mp/chg_datasets/)
FUNCTIONALS = {"gga": "rho_gga", "gga+u": "rho_gga+u"}

image = modal.Image.debian_slim()


@app.function(image=image, volumes={VOLUME_ROOT: data_volume}, timeout=900)
def prep(smoke_n: int = 200):
    import logging
    from pathlib import Path

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    base = Path(VOLUME_ROOT) / "mp" / "chg_datasets"

    for func, rho in FUNCTIONALS.items():
        fdir = base / "functionals" / func
        filelist = fdir / "mp_filelist.txt"
        if not filelist.exists():
            raise FileNotFoundError(
                f"{filelist} missing — transfer functionals/{func}/mp_filelist.txt "
                "(and split.json) before running prep."
            )

        # 1. data/label symlinks -> ../../rho_*/{data,label}
        for sub in ("data", "label"):
            link = fdir / sub
            target = Path("../..") / rho / sub  # relative to fdir
            real = base / rho / sub
            if not real.exists():
                raise FileNotFoundError(
                    f"{real} missing — transfer the {rho} data dir before running prep."
                )
            if link.is_symlink():
                link.unlink()
                link.symlink_to(target)
                log.info("relinked %s -> %s", link, target)
            elif link.is_dir():
                log.info(
                    "%s is a real dir (Globus followed symlinks); leaving as-is", link
                )
            else:
                link.symlink_to(target)
                log.info("linked %s -> %s", link, target)

        # 2. smoke filelist
        ids = filelist.read_text().splitlines()
        smoke = fdir / "mp_filelist_smoke.txt"
        smoke.write_text("\n".join(ids[:smoke_n]) + "\n")
        log.info("wrote %s (%d ids)", smoke, min(smoke_n, len(ids)))

        # 3. sanity check: first id resolves to a `.zarr.zip` (packed) or
        # `.zarr/` (unpacked) store under data/.
        first = ids[0]
        store_dir = fdir / "data" / f"{first}.zarr"
        store_zip = fdir / "data" / f"{first}.zarr.zip"
        if not (store_zip.exists() or store_dir.exists()):
            raise FileNotFoundError(
                f"Neither {store_zip} nor {store_dir} found — "
                "data/ symlink or transfer is incomplete."
            )
        log.info(
            "OK: %s resolves (%d total ids)",
            store_zip if store_zip.exists() else store_dir,
            len(ids),
        )

    # 4. persist
    data_volume.commit()
    log.info("Volume committed.")


@app.local_entrypoint()
def main(smoke_n: int = 200):
    prep.remote(smoke_n=smoke_n)
