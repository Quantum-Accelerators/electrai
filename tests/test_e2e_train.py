"""Deterministic e2e training test.

Imports `run_training` from `scripts/e2e_train.py` and asserts that the
final validation loss matches platform-specific expected values.
"""

from __future__ import annotations

import json
from pathlib import Path

EXPECTED_VALUES_FILE = Path(__file__).parent / "expected_values.json"
TOLERANCE = 0.001


def test_e2e_training():
    """Run deterministic e2e training and verify loss matches expected."""
    # Import here to avoid slow torch import at collection time
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from e2e_train import run_training

    results = run_training(epochs=5)
    platform = results["platform"]

    expected_values = json.loads(EXPECTED_VALUES_FILE.read_text())
    assert platform in expected_values, (
        f"No expected values for platform {platform!r}, "
        f"available: {list(expected_values.keys())}"
    )

    expected = expected_values[platform]
    assert expected.get("final_val_loss") is not None, (
        f"Expected values for {platform!r} are null (not yet generated)"
    )

    expected_val_loss = expected["final_val_loss"]
    actual_val_loss = results["final_val_loss"]
    diff = abs(actual_val_loss - expected_val_loss)
    assert diff <= TOLERANCE, (
        f"val_loss {actual_val_loss:.6f} differs from expected "
        f"{expected_val_loss:.6f} by {diff:.6f} (tolerance: {TOLERANCE})"
    )
