"""Small local JSON experiment tracker used by training scripts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("experiments/experiments.json")


def record_experiment(record: dict[str, Any], path: Path = DEFAULT_PATH) -> dict[str, Any]:
    """Append one JSON record without requiring an external tracking service."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except json.JSONDecodeError:
        existing = []
    if isinstance(existing, dict):
        existing = existing.get("experiments", [])
    if not isinstance(existing, list):
        existing = []
    record = {
        "experiment_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "date": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    existing.append(record)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return record
