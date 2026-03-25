"""Modal GPU CI for electrai e2e training test."""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

# Dependencies read from pyproject.toml (shared with train.py, populate_volume.py)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_pyproject(
        str(ROOT / "pyproject.toml"), optional_dependencies=["dev"]
    )
    .add_local_dir(str(ROOT / "src"), remote_path="/root/electrai/src", copy=True)
    .add_local_dir(
        str(ROOT / "scripts"), remote_path="/root/electrai/scripts", copy=True
    )
    .add_local_dir(str(ROOT / "tests"), remote_path="/root/electrai/tests", copy=True)
    .add_local_dir(str(ROOT / "data"), remote_path="/root/electrai/data", copy=True)
    .add_local_file(
        str(ROOT / "pyproject.toml"),
        remote_path="/root/electrai/pyproject.toml",
        copy=True,
    )
    .run_commands("cd /root/electrai && pip install --no-deps -e .")
)

app = modal.App("electrai-ci", image=image)


@app.function(gpu="L4", timeout=600, retries=0)
def run_e2e_test(epochs: int = 5, check: bool = True):
    """Run e2e training test on GPU."""
    import json
    import logging
    import sys

    log = logging.getLogger(__name__)

    sys.path.insert(0, "/root/electrai/scripts")
    from e2e_train import run_training

    results = run_training(epochs=epochs, gpu=True, verbose=True)
    log.info("Platform: %s", results["platform"])
    log.info("Final val_loss: %.6f", results["final_val_loss"])
    if results["final_train_loss"] is not None:
        log.info("Final train_loss: %.6f", results["final_train_loss"])
    log.info("Wallclock: %.1fs", results["wallclock_s"])

    if check:
        expected_file = Path("/root/electrai/tests/expected_values.json")
        expected_values = json.loads(expected_file.read_text())
        platform = results["platform"]
        if platform not in expected_values:
            raise ValueError(
                f"No expected values for platform {platform!r}, "
                f"available: {list(expected_values.keys())}"
            )
        expected = expected_values[platform]
        if expected.get("final_val_loss") is None:
            raise ValueError(f"Expected values for {platform!r} are null")
        expected_val_loss = expected["final_val_loss"]
        diff = abs(results["final_val_loss"] - expected_val_loss)
        tolerance = 0.001
        if diff > tolerance:
            raise AssertionError(
                f"val_loss {results['final_val_loss']:.6f} differs from expected "
                f"{expected_val_loss:.6f} by {diff:.6f} (tolerance: {tolerance})"
            )
        log.info(
            "PASS: val_loss matches expected within tolerance (%.6f <= %f)",
            diff,
            tolerance,
        )

    return results


@app.local_entrypoint()
def main(epochs: int = 5, check: bool = True):
    import logging

    logging.basicConfig(level=logging.INFO)
    results = run_e2e_test.remote(epochs=epochs, check=check)
    logging.getLogger(__name__).info(
        "Results: val_loss=%.6f in %.1fs",
        results["final_val_loss"],
        results["wallclock_s"],
    )
