"""Pytest wrapper for e2e training test."""
import subprocess
import sys


def test_e2e_training():
    """Run deterministic e2e training and verify loss matches expected."""
    result = subprocess.run(
        [sys.executable, "tests/e2e_train.py", "--epochs", "3"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, f"e2e test failed: {result.stderr}"
    assert "PASS" in result.stdout, f"Expected PASS in output: {result.stdout}"
