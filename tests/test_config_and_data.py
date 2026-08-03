# tests/test_config_and_data.py
"""Tests for standalone config import and the synthetic ticket generator.

Run with pytest:   pytest tests/test_config_and_data.py -v
Or stdlib runner:  python tests/test_config_and_data.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

from scripts.generate_tickets import TEMPLATES, _fill_templates, generate
from src import config


def test_fill_is_consistent_across_fields():
    rng = random.Random(1)
    title, desc, res = _fill_templates(
        ("issue at {site} on {os}", "user at {site} runs {os}", "check {site}"),
        rng,
    )
    site = title.split(" at ")[1].split(" on ")[0]
    assert f"at {site} " in desc and f"check {site}" == res
    # same {os} in both fields
    os_val = title.split(" on ")[1]
    assert desc.endswith(os_val)


def test_config_imports_without_dotenv_or_env_file():
    # Import already happened at module load; verify key defaults exist.
    assert config.PRODUCT_NAME  # non-empty
    assert config.ASR_MODEL == os.getenv("ASR_MODEL", "openai/whisper-tiny")
    assert isinstance(config.TOP_K, int) and config.TOP_K > 0
    assert 0.0 <= config.SIMILARITY_THRESHOLD <= 1.0
    assert "llama-3.1-8b-instant" in config.PRICE_PER_MTOK


def test_generator_is_deterministic():
    a = generate(rows=30, seed=42)
    b = generate(rows=30, seed=42)
    assert a == b
    c = generate(rows=30, seed=7)
    assert a != c


def test_generator_schema_and_balance():
    rows = generate(rows=60, seed=42)
    assert len(rows) == 60
    expected_cols = {
        "ticket_id", "title", "description", "category",
        "priority", "resolution_steps", "resolved_date",
    }
    assert set(rows[0].keys()) == expected_cols
    # No unfilled {placeholders} anywhere
    for r in rows:
        joined = " ".join(r.values())
        assert "{" not in joined and "}" not in joined
    # Balanced categories: 60 rows / 6 categories = 10 each
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    assert set(counts) == set(TEMPLATES.keys())
    assert all(v == 10 for v in counts.values())
    # Unique ticket ids
    assert len({r["ticket_id"] for r in rows}) == 60


# ------------------------------------------------------- stdlib test runner
def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
