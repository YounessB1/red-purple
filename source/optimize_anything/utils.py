"""Shared utilities for the optimize_anything pipeline."""

import hashlib
import json


def candidate_hash(candidate: dict) -> str:
    return hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()
