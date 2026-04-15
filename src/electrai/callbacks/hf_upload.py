from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from lightning.pytorch.callbacks import Callback

if TYPE_CHECKING:
    from types import SimpleNamespace

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "hf_upload_manifest.json"


class HuggingFaceCallback(Callback):
    """Tracks saved checkpoints for deferred upload to HuggingFace Hub.

    On clusters without internet (e.g. Princeton Della), checkpoints are
    queued in a JSON manifest and uploaded later via ``electrai hf-push``.
    When ``hf.upload_immediate`` is True, uploads are attempted inline
    (failures are logged but never crash training).
    """

    def __init__(self, cfg: SimpleNamespace) -> None:
        super().__init__()
        hf = cfg.hf
        self.repo_id: str = hf["repo_id"]
        self.every_n_epochs: int = hf.get("upload_every_n_epochs", 5)
        self.upload_immediate: bool = hf.get("upload_immediate", False)
        self.ckpt_path = Path(getattr(cfg, "ckpt_path", "./checkpoints"))
        self.manifest_path = self.ckpt_path / MANIFEST_FILENAME
        self._manifest: list[dict] = []
        self._load_existing_manifest()

    def _load_existing_manifest(self) -> None:
        if self.manifest_path.exists():
            with self.manifest_path.open(encoding="utf-8") as f:
                self._manifest = json.load(f)

    def _save_manifest(self) -> None:
        self.ckpt_path.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2)

    def _queue_checkpoint(
        self, ckpt_file: Path, epoch: int | None, *, path_in_repo: str | None = None
    ) -> int:
        """Add or update a checkpoint entry in the manifest.

        De-duplicates by checkpoint path/path_in_repo so that resuming runs or
        re-running hf-push does not grow the manifest with duplicate entries.

        Returns the index of the (new or updated) manifest entry.
        """
        path_str = str(ckpt_file)
        resolved_path_in_repo = path_in_repo or ckpt_file.name

        # Try to find an existing entry for this checkpoint.
        existing_idx: int | None = None
        for i, item in enumerate(self._manifest):
            if (
                item.get("path") == path_str
                or item.get("path_in_repo") == resolved_path_in_repo
            ):
                existing_idx = i
                break

        if existing_idx is not None:
            entry = self._manifest[existing_idx]
            entry["epoch"] = epoch
            entry["repo_id"] = self.repo_id
            # Reset upload status when (re-)queuing.
            entry["uploaded"] = False
            entry["path"] = path_str
            entry["path_in_repo"] = resolved_path_in_repo
            idx = existing_idx
        else:
            entry = {
                "path": path_str,
                "path_in_repo": resolved_path_in_repo,
                "epoch": epoch,
                "repo_id": self.repo_id,
                "uploaded": False,
            }
            self._manifest.append(entry)
            idx = len(self._manifest) - 1

        self._save_manifest()
        logger.info("Queued checkpoint for HF upload: %s", ckpt_file.name)
        return idx

    def on_validation_end(self, trainer, pl_module) -> None:  # noqa: ARG002
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        if trainer.global_rank != 0:
            return

        # Save the current state to a stable epoch-specific file. This is
        # independent of ModelCheckpoint's last.ckpt (which Lightning reorders
        # to run after us in this hook, so last.ckpt would still be stale).
        stable_name = f"last_epoch{epoch + 1:03d}.ckpt"
        stable_path = self.ckpt_path / stable_name
        trainer.save_checkpoint(stable_path)

        idx = self._queue_checkpoint(stable_path, epoch, path_in_repo=stable_name)

        if self.upload_immediate:
            entry = self._manifest[idx]
            _upload_single(entry)
            if entry["uploaded"]:
                stable_path.unlink(missing_ok=True)
            self._save_manifest()

    def on_train_end(self, trainer, pl_module) -> None:  # noqa: ARG002
        if trainer.global_rank != 0:
            return
        # Queue best checkpoints that haven't been queued yet
        queued_paths = {e["path"] for e in self._manifest}
        had_immediate = False
        for ckpt_file in self.ckpt_path.glob("ckpt_*.ckpt"):
            if str(ckpt_file) not in queued_paths:
                self._queue_checkpoint(ckpt_file, epoch=None)
                if self.upload_immediate:
                    _upload_single(self._manifest[-1])
                    had_immediate = True
        if had_immediate:
            self._save_manifest()

        pending = sum(1 for e in self._manifest if not e["uploaded"])
        if pending:
            logger.info(
                "%d checkpoint(s) pending upload. "
                "Run 'electrai hf-push --ckpt-path %s' from a node with "
                "internet access.",
                pending,
                self.ckpt_path,
            )


def _upload_single(entry: dict) -> None:
    """Attempt to upload a single checkpoint. Logs errors, never raises."""
    path = Path(entry["path"])
    try:
        from huggingface_hub import upload_file
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        logger.warning(
            "huggingface-hub is not installed. "
            "Run 'uv sync --extra hf' to enable uploads."
        )
        return
    try:
        if not path.exists():
            logger.warning("Checkpoint file not found, skipping: %s", path)
            return
        path_in_repo = entry.get("path_in_repo", path.name)
        upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            repo_id=entry["repo_id"],
        )
        entry["uploaded"] = True
        logger.info("Uploaded %s to %s", path.name, entry["repo_id"])
    except HfHubHTTPError:
        logger.warning(
            "HF upload failed for %s (check repo_id, auth token, and "
            "network access). Will retry with hf-push.",
            path.name,
            exc_info=True,
        )
    except Exception:
        logger.warning(
            "HF upload failed for %s (will retry with hf-push)",
            path.name,
            exc_info=True,
        )


def hf_push(ckpt_path: str, *, clean: bool = False) -> None:
    """Upload pending checkpoints from a manifest file.

    Run this from a login node or machine with internet access.
    """
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "huggingface-hub is not installed. "
            "Run 'uv sync --extra hf' to enable uploads."
        ) from e

    ckpt_dir = Path(ckpt_path)
    manifest_path = ckpt_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise SystemExit(f"No manifest found at {manifest_path}")

    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    pending = [e for e in manifest if not e["uploaded"]]
    if not pending:
        logger.info("All checkpoints already uploaded.")
        return

    logger.info("Uploading %d pending checkpoint(s)...", len(pending))
    for entry in pending:
        _upload_single(entry)
        if clean and entry["uploaded"]:
            Path(entry["path"]).unlink(missing_ok=True)

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    still_pending = sum(1 for e in manifest if not e["uploaded"])
    if still_pending:
        logger.warning("%d checkpoint(s) still failed to upload.", still_pending)
    else:
        logger.info("All checkpoints uploaded successfully.")
