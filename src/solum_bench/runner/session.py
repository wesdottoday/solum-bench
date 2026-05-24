"""Run session management — metadata, checkpointing, resume."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


class BenchmarkSession:
    """Manages a benchmark run session with metadata and checkpointing."""

    def __init__(self, output_dir: Path, config_path: Path | None = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = output_dir / "run_metadata.json"
        self.checkpoint_path = output_dir / "checkpoint.json"
        self.metadata: dict = {}
        self._completed_ids: set[str] = set()

    def initialize(
        self,
        models: list[str],
        question_bank_dir: Path,
        total_questions: int,
        seed: int,
        judge_model: str | None = None,
    ):
        """Initialize a new session."""
        bank_hash = _hash_directory(question_bank_dir)

        self.metadata = {
            "run_id": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "models": models,
            "question_bank_version": "v1",
            "question_bank_sha256": bank_hash,
            "total_questions": total_questions,
            "seed": seed,
            "judge_model": judge_model,
        }
        self._save_metadata()

    def mark_completed(self, model_name: str, question_id: str):
        """Record that a question has been completed for a model."""
        key = f"{model_name}::{question_id}"
        self._completed_ids.add(key)
        self._save_checkpoint()

    def is_completed(self, model_name: str, question_id: str) -> bool:
        """Check if a question has already been completed for a model."""
        return f"{model_name}::{question_id}" in self._completed_ids

    def finalize(self):
        """Mark the session as complete."""
        self.metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._save_metadata()

    def load_checkpoint(self) -> bool:
        """Load checkpoint from a previous run. Returns True if checkpoint found."""
        if self.checkpoint_path.exists():
            data = json.loads(self.checkpoint_path.read_text())
            self._completed_ids = set(data.get("completed", []))
            return True
        if self.meta_path.exists():
            self.metadata = json.loads(self.meta_path.read_text())
        return False

    def _save_metadata(self):
        self.meta_path.write_text(json.dumps(self.metadata, indent=2))

    def _save_checkpoint(self):
        data = {"completed": sorted(self._completed_ids)}
        self.checkpoint_path.write_text(json.dumps(data))


def _hash_directory(dir_path: Path) -> str:
    """Hash all YAML files in a directory for reproducibility tracking."""
    hasher = hashlib.sha256()
    for f in sorted(dir_path.glob("*.yaml")):
        hasher.update(f.read_bytes())
    return hasher.hexdigest()
