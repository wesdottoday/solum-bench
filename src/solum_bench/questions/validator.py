"""Validate question bank integrity."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from pydantic import ValidationError

from solum_bench.questions.loader import load_questions_from_file


def validate_bank(bank_dir: Path) -> list[str]:
    """Validate an entire question bank. Returns a list of error messages (empty = valid)."""
    errors: list[str] = []

    # Check manifest exists
    manifest_path = bank_dir / "manifest.yaml"
    if not manifest_path.exists():
        errors.append(f"Missing manifest.yaml in {bank_dir}")
        return errors

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    if not manifest:
        errors.append("manifest.yaml is empty")
        return errors

    # Validate each file listed in manifest
    seen_ids: set[str] = set()
    total_count = 0

    for file_entry in manifest.get("files", []):
        file_path = bank_dir / file_entry["path"]
        expected_count = file_entry.get("count")

        if not file_path.exists():
            errors.append(f"File listed in manifest does not exist: {file_entry['path']}")
            continue

        # Validate checksum if present
        if "sha256" in file_entry and file_entry["sha256"]:
            actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if actual_hash != file_entry["sha256"]:
                errors.append(
                    f"Checksum mismatch for {file_entry['path']}: "
                    f"expected {file_entry['sha256'][:16]}..., got {actual_hash[:16]}..."
                )

        # Load and validate questions
        try:
            questions = load_questions_from_file(file_path)
        except (ValidationError, Exception) as e:
            errors.append(f"Failed to load {file_entry['path']}: {e}")
            continue

        # Check count
        if expected_count is not None and len(questions) != expected_count:
            errors.append(
                f"Count mismatch in {file_entry['path']}: manifest says {expected_count}, found {len(questions)}"
            )

        # Check for duplicate IDs
        for q in questions:
            if q.id in seen_ids:
                errors.append(f"Duplicate question ID: {q.id}")
            seen_ids.add(q.id)

        total_count += len(questions)

    # Check total count
    expected_total = manifest.get("total_questions")
    if expected_total is not None and total_count != expected_total:
        errors.append(f"Total count mismatch: manifest says {expected_total}, found {total_count}")

    # Check for YAML files not in manifest
    manifest_files = {f["path"] for f in manifest.get("files", [])}
    for yaml_file in bank_dir.glob("*.yaml"):
        if yaml_file.name == "manifest.yaml":
            continue
        if yaml_file.name not in manifest_files:
            errors.append(f"File {yaml_file.name} exists but is not listed in manifest")

    return errors
