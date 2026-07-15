"""Focused stdlib tests for tools/analyze_ttft_repeatability.py."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from tools.analyze_ttft_repeatability import (
    EvidenceError,
    analyze,
    main,
    render_json,
    render_markdown,
)


SELECTORS = {
    "A": "cost_cachedisp_old",
    "B": "newest",
    "C": "cost_disp_current",
    "R": "waiting_random",
}


def summary(selector: str, routed: int, cost: float, *, seed: int = 0,
            local_violations: int = 0) -> dict:
    n = 100
    local_n = n - routed
    pessimistic_n = routed + local_violations
    return {
        "policy": "nimbus",
        "overall": {"n": n, "success": n, "cost_usd": cost},
        "local": {
            "n": local_n,
            "success": local_n,
            "slo_violations": local_violations,
            "ttft_p50_ms": 1000.0 + routed,
            "ttft_p95_ms": 2000.0 + routed,
            "ttft_p99_ms": 3000.0 + routed,
        },
        "cloud": {"n": routed, "success": routed},
        "pessimistic_combined": {
            "slo_violations": pessimistic_n,
            "slo_n": n,
            "slo_violation_pct": float(pessimistic_n),
            "cost_usd": cost,
        },
        "token_alignment": {
            "local_success_n": local_n,
            "measured_n": local_n,
            "missing_prompt_usage_n": 0,
            "prompt_exact_n": local_n,
            "decode_measured_n": local_n,
            "missing_completion_usage_n": 0,
            "decode_cap_hit_n": local_n,
        },
        "config": {
            "nimbus_trigger": "ttft_pred",
            "nimbus_selector": selector,
            "seed": seed,
        },
        "queue": {
            "nimbus_trigger": "ttft_pred",
            "nimbus_selector": selector,
        },
    }


def write_block(root: Path, name: str, rows: dict[str, tuple[int, float]]) -> Path:
    block = root / name
    block.mkdir()
    for arm, (routed, cost) in rows.items():
        payload = summary(SELECTORS[arm], routed, cost, seed=7 if arm == "R" else 0)
        (block / f"run_{arm}.summary.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return block


class TestRepeatabilityAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "campaign"
        self.root.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_campaign_root_pairs_and_aggregates_whole_blocks(self) -> None:
        write_block(self.root, "block01", {
            "A": (20, 0.10), "B": (30, 0.13), "C": (25, 0.12), "R": (35, 0.14),
        })
        write_block(self.root, "block02", {
            "A": (24, 0.11), "B": (31, 0.14), "C": (23, 0.13), "R": (36, 0.15),
        })

        result = analyze([self.root])

        self.assertEqual(result["statistical_unit"], "block")
        self.assertFalse(result["per_request_resampling"])
        self.assertEqual(result["validation"]["block_count"], 2)
        self.assertTrue(result["validation"]["evidence_valid"])
        ab = [
            row for row in result["paired_deltas"]
            if row["block"] == "block01" and row["comparison"] == "A-B"
        ][0]
        self.assertEqual(ab["delta_a_minus_other"]["routed_n"], -10.0)
        self.assertEqual(ab["winner"]["routed_n"], "A")
        ac_wins = result["paired_aggregate"]["A-C"]["routed_n"]["wins"]
        self.assertEqual(ac_wins, {"A": 1, "tie": 0, "C": 1})
        a_routed = result["aggregate_by_arm"]["A"]["routed_n"]
        self.assertEqual(a_routed["n_blocks"], 2)
        self.assertEqual(a_routed["mean"], 22.0)
        self.assertEqual(a_routed["median"], 22.0)
        self.assertEqual((a_routed["min"], a_routed["max"]), (20.0, 24.0))

        markdown = render_markdown(result)
        self.assertIn("one complete block/run", markdown)
        self.assertIn("-10 (A)", markdown)
        self.assertNotIn(self.temp.name, markdown)
        encoded = render_json(result)
        self.assertNotIn(self.temp.name, encoded)
        self.assertEqual(json.loads(encoded)["schema_version"], 1)

    def test_failed_request_is_rejected(self) -> None:
        block = write_block(self.root, "block01", {
            "A": (20, 0.10), "B": (30, 0.13), "C": (25, 0.12),
        })
        path = block / "run_A.summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["overall"]["success"] = 99
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "overall success 99/100"):
            analyze([block])

    def test_inexact_tokens_are_rejected(self) -> None:
        block = write_block(self.root, "block01", {
            "A": (20, 0.10), "B": (30, 0.13), "C": (25, 0.12),
        })
        path = block / "run_C.summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["token_alignment"]["prompt_exact_n"] -= 1
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "token alignment is not exact"):
            analyze([block])

    def test_local_violation_fails_by_default_but_can_be_diagnosed(self) -> None:
        block = write_block(self.root, "block01", {
            "A": (20, 0.10), "B": (30, 0.13), "C": (25, 0.12),
        })
        path = block / "run_B.summary.json"
        payload = summary(
            SELECTORS["B"], 30, 0.13, local_violations=1
        )
        path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(EvidenceError, "local SLO violations"):
            analyze([block])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_error:
            main([str(block)])
        self.assertEqual(exit_error.exception.code, 2)
        diagnostic = analyze([block], require_zero_local_violations=False)
        self.assertFalse(diagnostic["validation"]["zero_local_violations"])
        self.assertFalse(diagnostic["validation"]["evidence_valid"])
        self.assertIn("INVALID (diagnostic only)", render_markdown(diagnostic))

    def test_missing_required_arm_and_duplicate_random_are_rejected(self) -> None:
        block = write_block(self.root, "block01", {
            "A": (20, 0.10), "B": (30, 0.13),
        })
        with self.assertRaisesRegex(EvidenceError, "missing required arms C"):
            analyze([block])

        payload = summary(SELECTORS["C"], 25, 0.12)
        (block / "run_C.summary.json").write_text(json.dumps(payload), encoding="utf-8")
        random = summary(SELECTORS["R"], 35, 0.14, seed=1)
        (block / "run_R1.summary.json").write_text(json.dumps(random), encoding="utf-8")
        random["config"]["seed"] = 2
        (block / "run_R2.summary.json").write_text(json.dumps(random), encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "duplicate arm R"):
            analyze([block])


if __name__ == "__main__":
    unittest.main()
