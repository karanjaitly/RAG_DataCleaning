from __future__ import annotations

import math
from typing import Any


def to_json_ready(value: Any) -> Any:
    """Recursively convert pipeline outputs into strict JSON-safe values."""
    if isinstance(value, dict):
        return {key: to_json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_ready(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return to_json_ready(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
