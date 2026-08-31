"""Load della data onto the electrai-data Modal Volume via Globus.

Runs Globus Connect Personal (GCP) inside a CPU-only Modal container with the
`electrai-data` Volume mounted at /data, registered as a personal endpoint using
a setup key generated in the Globus web UI. GCP makes only outbound connections,
so it works behind Modal's NAT. You then initiate the della -> this endpoint
transfer from the Globus file manager, writing into /data; the Volume is
committed periodically and on exit.

Steps:
    1. In the Globus web UI (https://app.globus.org), create a Globus Connect
       Personal collection and copy its setup key.
    2. Start the endpoint host (runs up to ~24h):
           modal run modal/globus_load.py --setup-key "<KEY>"
    3. In the Globus file manager (https://app.globus.org/file-manager) transfer:
           source: ROSENGROUP share, .../mp/chg_datasets/functionals/{gga,gga+u}
           dest:   this endpoint, path /data/mp/chg_datasets/functionals/
       The data/label entries are symlinks into rho_gga{,+u}; confirm Globus
       follows them (lands real .zarr stores). If it copies symlinks instead,
       transfer rho_gga and rho_gga+u and adjust the config roots.
    4. When the Globus task shows complete, stop this run (ctrl-C / `modal app
       stop electrai-globus-load`).
"""

from __future__ import annotations

import modal

app = modal.App("electrai-globus-load")

data_volume = modal.Volume.from_name("electrai-data", create_if_missing=True)

VOLUME_ROOT = "/data"
GCP_URL = (
    "https://downloads.globus.org/globus-connect-personal/linux/stable/"
    "globusconnectpersonal-latest.tgz"
)

image = (
    modal.Image.debian_slim()
    .apt_install("wget", "ca-certificates", "tar")
    .run_commands(
        "cd /opt && wget -q "
        f"{GCP_URL}"
        " -O gcp.tgz && tar xzf gcp.tgz && rm gcp.tgz "
        "&& mv globusconnectpersonal-* globusconnectpersonal"
    )
)


@app.function(
    image=image,
    volumes={VOLUME_ROOT: data_volume},
    timeout=86400,  # 24h max: one transfer window
    cpu=4.0,
)
def host_endpoint(setup_key: str, run_hours: float = 23.5, commit_every_s: int = 300):
    import logging
    import subprocess
    import time
    from pathlib import Path

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    gcp = "/opt/globusconnectpersonal/globusconnectpersonal"

    # Non-interactive endpoint registration using the web-UI setup key.
    log.info("Registering Globus Connect Personal endpoint...")
    subprocess.run([gcp, "-setup", "--setup-key", setup_key], check=True)

    # Grant read/write to the mounted Volume (path,writable,shareable).
    cfg_dir = Path.home() / ".globusonline" / "lta"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config-paths").write_text(f"{VOLUME_ROOT},1,0\n")
    log.info("config-paths -> %s,1,0", VOLUME_ROOT)

    # Start GCP in the background (outbound-only).
    log.info("Starting Globus Connect Personal...")
    proc = subprocess.Popen([gcp, "-start"])
    log.info(
        "Endpoint up. Initiate the della -> %s transfer in the Globus web UI now.",
        VOLUME_ROOT,
    )

    deadline = time.time() + run_hours * 3600
    try:
        while time.time() < deadline:
            time.sleep(commit_every_s)
            data_volume.commit()  # persist whatever has landed so far
            status = subprocess.run(
                [gcp, "-status"], capture_output=True, text=True, check=False
            )
            log.info("globus status: %s", (status.stdout or status.stderr).strip())
    finally:
        proc.terminate()
        data_volume.commit()
        log.info("Stopped GCP and committed Volume.")


@app.local_entrypoint()
def main(setup_key: str, run_hours: float = 23.5):
    host_endpoint.remote(setup_key=setup_key, run_hours=run_hours)
