"""Focused stdlib tests for tools/analyze_ttft_full_cell.py."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.analyze_ttft_full_cell import (
    DEFAULT_CONFIG_CONTRACT,
    EvidenceError,
    analyze,
    render_json,
    render_markdown,
)


IDENTITIES = {
    "A": ("nimbus", "ttft_pred", "cost_cachedisp_old"),
    "B": ("nimbus", "ttft_pred", "newest"),
    "C": ("nimbus", "ttft_pred", "cost_disp_current"),
    "K": ("nimbus", "kv_gap", "cost_disp_current"),
    "L": ("all_local", None, None),
}
ARM_TEXT = {
    "A": "ttft_pred:cost_cachedisp_old:0",
    "B": "ttft_pred:newest:0",
    "C": "ttft_pred:cost_disp_current:0",
    "K": "kv_gap:cost_disp_current:0",
    "L": "anchor:all_local:0",
}
FINGERPRINT = "f" * 64
TEST_MANIFEST_CONTRACT = {
    "commit": "test-commit",
    "trace_sha256": "1" * 64,
    "trace_manifest_sha256": "2" * 64,
    "profile_sha256": "3" * 64,
}
FROZEN_ARMS_LINE = " ".join(
    ARM_TEXT[arm] for arm in ("B", "K", "C", "A", "L")
)


def artifact(path: Path) -> dict:
    content = path.read_bytes()
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "nonempty_line_n": sum(bool(line.strip()) for line in content.splitlines()),
    }


def synthetic_cloud_cost(routed: int) -> float:
    return sum(
        ((100 + request_id) * float(DEFAULT_CONFIG_CONTRACT["in_price"])
         + (10 + request_id % 7) * float(DEFAULT_CONFIG_CONTRACT["out_price"]))
        / 1e6
        for request_id in range(routed)
    )


def make_summary(
    arm: str,
    *,
    n: int = 100,
    routed: int,
    local_violations: int = 0,
    cost: float | None = None,
    local_p50_ms: float = 2000.0,
    max_post_kick: float = 3.0,
) -> dict:
    policy, trigger, selector = IDENTITIES[arm]
    if cost is None:
        cost = synthetic_cloud_cost(routed)
    local_n = n - routed
    combined_n = routed + local_violations
    queue = {}
    if policy == "nimbus":
        queue.update({
            "nimbus_trigger": trigger,
            "nimbus_selector": selector,
            "nimbus_ticks": 1,
            "nimbus_kick_rounds": 1 if routed else 0,
            "nimbus_kicked": routed,
            "nimbus_applied_kick_rounds": 1 if routed else 0,
            "nimbus_stale_decisions": 0,
            "nimbus_decision_calls": 1,
            "nimbus_decision_mean_ms": 0.25,
            "nimbus_decision_max_ms": 0.25,
        })
    if arm in ("A", "B", "C"):
        limit = (
            float(DEFAULT_CONFIG_CONTRACT["slo_s"])
            - float(DEFAULT_CONFIG_CONTRACT["ttft_guard_ms"]) / 1000.0
        )
        predicted = max(limit + 1.0, max_post_kick + 1.0)
        queue.update({
            "nimbus_prediction_scope": "waiting_only",
            "nimbus_prediction_model": "seq_slots_shared_prefill_lane_v1",
            "nimbus_max_waiting_predicted_ttft_s": predicted,
            "nimbus_max_waiting_post_kick_ttft_s": max_post_kick,
        })
    if arm == "K":
        queue["kv_read_failures"] = 0
    return {
        "policy": policy,
        "slo_s": 5.0,
        "overall": {"n": n, "success": n, "cost_usd": cost},
        "local": {
            "n": local_n,
            "success": local_n,
            "slo_violations": local_violations,
            "slo_measured_n": local_n,
            "slo_violation_pct": 100.0 * local_violations / max(local_n, 1),
            "ttft_p50_ms": local_p50_ms,
            "ttft_p95_ms": local_p50_ms + 500.0,
            "ttft_p99_ms": local_p50_ms + 900.0,
        },
        "cloud": {"n": routed, "success": routed},
        "pessimistic_combined": {
            "slo_violations": combined_n,
            "slo_n": n,
            "slo_violation_pct": 100.0 * combined_n / n,
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
            **DEFAULT_CONFIG_CONTRACT,
            "nimbus_trigger": trigger or "kv_gap",
            "nimbus_selector": selector or "cost_disp_current",
        },
        "queue": queue,
    }


def make_raw(n: int, routed: int, local_violations: int) -> str:
    rows = []
    for request_id in range(n):
        cloud = request_id < routed
        prompt = 100 + request_id
        decode = 10 + (request_id % 7)
        cost = (
            prompt * float(DEFAULT_CONFIG_CONTRACT["in_price"])
            + decode * float(DEFAULT_CONFIG_CONTRACT["out_price"])
        ) / 1e6 if cloud else 0.0
        is_local_violation = (
            not cloud and routed <= request_id < routed + local_violations
        )
        rows.append(json.dumps({
            "request_id": request_id,
            "success": True,
            "error": None,
            "error_type": None,
            "endpoint": "cloud" if cloud else "local",
            "routed_only": cloud,
            "model": DEFAULT_CONFIG_CONTRACT["local_model"],
            "cache_mode": "none",
            "prompt_tokens": prompt,
            "completion_tokens": decode,
            "scheduler_prompt_tokens": prompt,
            "scheduler_uncached_prompt_tokens": prompt,
            "scheduler_decode_tokens": decode,
            "ttft_ms": None if cloud else (6000.0 if is_local_violation else 2000.0),
            "cost_usd": cost,
        }, sort_keys=True))
    return "".join(row + "\n" for row in rows)


def make_decisions(arm: str, summary: dict) -> str:
    if arm == "L":
        return ""
    trigger = summary["config"]["nimbus_trigger"]
    selector = summary["config"]["nimbus_selector"]
    routed = summary["cloud"]["n"]
    victims = list(range(routed))
    row = {
        "decision_id": 1,
        "at_s": 1.0,
        "status": "applied" if victims else "no_op",
        "trigger": trigger,
        "selector": selector,
        "proposed_victim_ids": victims,
        "applied_victim_ids": victims,
        "decision_ms": 0.25,
    }
    if arm in ("A", "B", "C"):
        row.update({
            "waiting_predicted_max_ttft_s": summary["queue"][
                "nimbus_max_waiting_predicted_ttft_s"
            ],
            "waiting_post_kick_max_ttft_s": summary["queue"][
                "nimbus_max_waiting_post_kick_ttft_s"
            ],
        })
    return json.dumps(row, sort_keys=True) + "\n"


class TestFullCellAnalyzer(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_matrix(
        self, overrides: dict[str, dict] | None = None, *, n: int = 100
    ) -> Path:
        values = {
            "A": dict(routed=20),
            "B": dict(routed=30),
            "C": dict(routed=22),
            "K": dict(routed=30, local_violations=5),
            "L": dict(routed=0, local_violations=95, local_p50_ms=6000.0),
        }
        for arm, update in (overrides or {}).items():
            values[arm].update(update)
        matrix = self.root / f"matrix_{len(list(self.root.iterdir()))}"
        matrix.mkdir()
        (matrix / "matrix_manifest.txt").write_text(
            f"run_fingerprint={FINGERPRINT}\n"
            f"commit={TEST_MANIFEST_CONTRACT['commit']}\n"
            f"trace_sha256={TEST_MANIFEST_CONTRACT['trace_sha256']} "
            f"trace_manifest_sha256={TEST_MANIFEST_CONTRACT['trace_manifest_sha256']} "
            f"trace_n={n}\n"
            f"profile_sha256={TEST_MANIFEST_CONTRACT['profile_sha256']}\n"
            f"arms={FROZEN_ARMS_LINE} \n",
            encoding="utf-8",
        )
        for arm, kwargs in values.items():
            stem = f"run_{arm}"
            raw_path = matrix / f"{stem}.jsonl"
            summary_path = matrix / f"{stem}.summary.json"
            decisions_path = matrix / f"{stem}.decisions.jsonl"
            summary = make_summary(arm, n=n, **kwargs)
            raw_path.write_text(
                make_raw(
                    n,
                    kwargs["routed"],
                    kwargs.get("local_violations", 0),
                ),
                encoding="utf-8",
            )
            summary_path.write_text(
                json.dumps(summary), encoding="utf-8"
            )
            decisions_path.write_text(
                make_decisions(arm, summary), encoding="utf-8"
            )
            self.write_marker(matrix, arm)
        return matrix

    def write_marker(
        self, matrix: Path, arm: str, *, fingerprint: str = FINGERPRINT,
        arm_text: str | None = None,
    ) -> None:
        stem = f"run_{arm}"
        marker = {
            "schema_version": 1,
            "run_fingerprint": fingerprint,
            "arm": arm_text or ARM_TEXT[arm],
            "artifacts": {
                "raw": artifact(matrix / f"{stem}.jsonl"),
                "summary": artifact(matrix / f"{stem}.summary.json"),
                "decisions": artifact(matrix / f"{stem}.decisions.jsonl"),
            },
        }
        (matrix / f"{stem}.complete.json").write_text(
            json.dumps(marker), encoding="utf-8"
        )

    def mutate(self, matrix: Path, arm: str, callback) -> None:
        path = matrix / f"run_{arm}.summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        callback(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def analyze_small(self, matrix: Path) -> dict:
        return analyze(
            matrix,
            expected_n=100,
            equivalence_band_n=3,
            expected_fingerprint=FINGERPRINT,
            expected_manifest=TEST_MANIFEST_CONTRACT,
        )

    def test_passing_gate_reports_all_metrics_and_freezes_default(self) -> None:
        result = self.analyze_small(self.write_matrix())
        self.assertTrue(result["heldout_gate_pass"])
        self.assertFalse(result["default_flip_authorized"])
        self.assertEqual(result["statistical_unit"], "one complete arm/run")
        self.assertFalse(result["per_request_bootstrap"])
        self.assertTrue(result["completion_markers_valid"])
        self.assertEqual(result["run_fingerprint"], FINGERPRINT)
        self.assertEqual(result["arms"]["A"]["routed_n"], 20)
        self.assertEqual(result["arms"]["A"]["local_n"], 80)
        self.assertEqual(result["arms"]["A"]["local_ttft_p50_s"], 2.0)
        self.assertEqual(
            result["gates"]["a_vs_c_equivalence"]["classification"],
            "route_equivalent_within_band",
        )
        markdown = render_markdown(result)
        self.assertIn("No request-level bootstrap", markdown)
        self.assertIn("heldout_gate_pass = true", markdown)
        self.assertIn("default_flip_authorized = false", markdown)
        self.assertTrue(json.loads(render_json(result))["heldout_gate_pass"])

    def test_c_beating_a_beyond_band_rejects(self) -> None:
        result = self.analyze_small(self.write_matrix({"C": {"routed": 10}}))
        gate = result["gates"]["a_vs_c_equivalence"]
        self.assertFalse(gate["pass"])
        self.assertEqual(gate["classification"], "C_beats_A_beyond_band_reject")
        self.assertFalse(result["heldout_gate_pass"])

    def test_a_must_beat_b_by_five_points(self) -> None:
        result = self.analyze_small(self.write_matrix({"B": {"routed": 23}}))
        self.assertFalse(result["gates"]["a_vs_b_signal"]["pass"])
        self.assertEqual(result["gates"]["a_vs_b_signal"]["b_minus_a_pp"], 3.0)

    def test_a_must_pareto_improve_k(self) -> None:
        result = self.analyze_small(self.write_matrix({
            "K": {"routed": 15, "local_violations": 0}
        }))
        gate = result["gates"]["a_pareto_vs_k"]
        self.assertFalse(gate["pass"])
        self.assertFalse(gate["no_worse_combined"])

    def test_all_local_must_demonstrate_pressure(self) -> None:
        result = self.analyze_small(self.write_matrix({
            "L": {"local_violations": 80, "local_p50_ms": 4500.0}
        }))
        self.assertFalse(result["gates"]["pressure_anchor"]["pass"])
        self.assertFalse(result["heldout_gate_pass"])

    def test_token_integrity_error_is_not_an_outcome_failure(self) -> None:
        matrix = self.write_matrix()
        self.mutate(
            matrix, "A",
            lambda payload: payload["token_alignment"].update(prompt_exact_n=79),
        )
        self.write_marker(matrix, "A")
        with self.assertRaisesRegex(EvidenceError, "token alignment is not exact"):
            self.analyze_small(matrix)

    def test_raw_jsonl_must_parse_and_have_unique_exact_request_ids(self) -> None:
        matrix = self.write_matrix()
        raw_path = matrix / "run_A.jsonl"
        lines = raw_path.read_text(encoding="utf-8").splitlines()
        lines[1] = "not-json"
        raw_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.write_marker(matrix, "A")
        with self.assertRaisesRegex(EvidenceError, "line 2 is not valid JSON"):
            self.analyze_small(matrix)

        matrix = self.write_matrix()
        raw_path = matrix / "run_A.jsonl"
        rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
        rows[1]["request_id"] = rows[0]["request_id"]
        raw_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        self.write_marker(matrix, "A")
        with self.assertRaisesRegex(EvidenceError, "raw request IDs are not unique"):
            self.analyze_small(matrix)

    def test_raw_recomputes_no_cache_tokens_violations_and_cloud_cost(self) -> None:
        matrix = self.write_matrix()
        raw_path = matrix / "run_A.jsonl"
        rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
        local = next(row for row in rows if row["endpoint"] == "local")
        local["scheduler_uncached_prompt_tokens"] += 1
        raw_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        self.write_marker(matrix, "A")
        with self.assertRaisesRegex(EvidenceError, "no-cache prompt/scheduler"):
            self.analyze_small(matrix)

        matrix = self.write_matrix()
        raw_path = matrix / "run_A.jsonl"
        rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
        local = next(row for row in rows if row["endpoint"] == "local")
        local["ttft_ms"] = 5000.0001
        raw_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        self.write_marker(matrix, "A")
        with self.assertRaisesRegex(EvidenceError, "raw local 5s violations=1"):
            self.analyze_small(matrix)

        matrix = self.write_matrix()
        raw_path = matrix / "run_A.jsonl"
        rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
        cloud = next(row for row in rows if row["endpoint"] == "cloud")
        cloud["cost_usd"] += 0.001
        raw_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        self.write_marker(matrix, "A")
        with self.assertRaisesRegex(EvidenceError, "does not match token-price cost"):
            self.analyze_small(matrix)

    def test_decision_application_sets_and_boundary_are_audited(self) -> None:
        matrix = self.write_matrix()
        path = matrix / "run_A.decisions.jsonl"
        decision = json.loads(path.read_text())
        decision["proposed_victim_ids"][-1] = 20
        decision["applied_victim_ids"][-1] = 20
        path.write_text(json.dumps(decision) + "\n", encoding="utf-8")
        self.write_marker(matrix, "A")
        with self.assertRaisesRegex(EvidenceError, r"union\(applied victims\)"):
            self.analyze_small(matrix)

        matrix = self.write_matrix()
        path = matrix / "run_A.decisions.jsonl"
        first = json.loads(path.read_text())
        second = {
            **first,
            "decision_id": 2,
            "at_s": 2.0,
            "proposed_victim_ids": [0],
            "applied_victim_ids": [0],
        }
        path.write_text(
            json.dumps(first) + "\n" + json.dumps(second) + "\n",
            encoding="utf-8",
        )
        self.write_marker(matrix, "A")
        with self.assertRaisesRegex(EvidenceError, "was applied more than once"):
            self.analyze_small(matrix)

        matrix = self.write_matrix()
        path = matrix / "run_A.decisions.jsonl"
        decision = json.loads(path.read_text())
        limit = 5.0 - 1.712
        decision["waiting_predicted_max_ttft_s"] = limit
        path.write_text(json.dumps(decision) + "\n", encoding="utf-8")
        self.mutate(
            matrix,
            "A",
            lambda payload: payload["queue"].update(
                nimbus_max_waiting_predicted_ttft_s=limit
            ),
        )
        self.write_marker(matrix, "A")
        with self.assertRaisesRegex(EvidenceError, "requires pre-kick TTFT >"):
            self.analyze_small(matrix)

    def test_empty_applied_stale_bounded_record_is_valid(self) -> None:
        matrix = self.write_matrix()
        path = matrix / "run_A.decisions.jsonl"
        first = json.loads(path.read_text())
        bounded = {
            **first,
            "decision_id": 2,
            "at_s": 2.0,
            "status": "applied_stale_bounded",
            "proposed_victim_ids": [],
            "applied_victim_ids": [],
            "waiting_predicted_max_ttft_s": 3.0,
            "waiting_post_kick_max_ttft_s": 3.0,
        }
        path.write_text(
            json.dumps(first) + "\n" + json.dumps(bounded) + "\n",
            encoding="utf-8",
        )
        self.mutate(
            matrix,
            "A",
            lambda payload: payload["queue"].update(
                nimbus_ticks=2,
                nimbus_decision_calls=2,
            ),
        )
        self.write_marker(matrix, "A")
        result = self.analyze_small(matrix)
        self.assertTrue(result["integrity_valid"])

    def test_frozen_config_and_manifest_contract_are_evidence_requirements(self) -> None:
        matrix = self.write_matrix()
        self.mutate(
            matrix, "A", lambda payload: payload["config"].update(slo_s=4.9)
        )
        self.write_marker(matrix, "A")
        with self.assertRaisesRegex(EvidenceError, "config.slo_s=.*frozen"):
            self.analyze_small(matrix)

        matrix = self.write_matrix()
        manifest = matrix / "matrix_manifest.txt"
        text = manifest.read_text(encoding="utf-8").replace(
            f"commit={TEST_MANIFEST_CONTRACT['commit']}", "commit=wrong"
        )
        manifest.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "contract mismatch for commit"):
            self.analyze_small(matrix)

    def test_prediction_identity_is_integrity_but_post_kick_breach_is_outcome(self) -> None:
        matrix = self.write_matrix()
        self.mutate(
            matrix, "B",
            lambda payload: payload["queue"].update(nimbus_prediction_scope="all_rows"),
        )
        self.write_marker(matrix, "B")
        with self.assertRaisesRegex(EvidenceError, "waiting-only/shared-prefill"):
            self.analyze_small(matrix)

        result = self.analyze_small(
            self.write_matrix({"C": {"max_post_kick": 4.01}})
        )
        self.assertFalse(result["gates"]["abc_post_kick_bound"]["pass"])
        self.assertFalse(result["heldout_gate_pass"])
        self.assertTrue(result["integrity_valid"])

    def test_k_kv_read_failure_is_integrity_error(self) -> None:
        matrix = self.write_matrix()
        self.mutate(matrix, "K", lambda payload: payload["queue"].update(kv_read_failures=1))
        self.write_marker(matrix, "K")
        with self.assertRaisesRegex(EvidenceError, "kv_read_failures=1"):
            self.analyze_small(matrix)

    def test_requires_exactly_five_completion_markers(self) -> None:
        matrix = self.write_matrix()
        (matrix / "run_L.complete.json").unlink()
        with self.assertRaisesRegex(EvidenceError, "exactly 5 completion markers"):
            self.analyze_small(matrix)

    def test_artifact_tamper_breaks_exact_completion_marker(self) -> None:
        for suffix in (".jsonl", ".summary.json", ".decisions.jsonl"):
            with self.subTest(artifact=suffix):
                matrix = self.write_matrix()
                # Whitespace preserves the nonempty-line count (and JSON
                # validity for the summary) but changes the SHA-256.
                with (matrix / f"run_A{suffix}").open("a", encoding="utf-8") as output:
                    output.write("\n")
                with self.assertRaisesRegex(
                    EvidenceError, "does not exactly match current artifacts"
                ):
                    self.analyze_small(matrix)

    def test_fingerprint_must_be_shared_and_exactly_bound_in_manifest(self) -> None:
        matrix = self.write_matrix()
        self.write_marker(matrix, "A", fingerprint="a" * 64)
        with self.assertRaisesRegex(EvidenceError, "do not share one run_fingerprint"):
            self.analyze_small(matrix)

        matrix = self.write_matrix()
        (matrix / "matrix_manifest.txt").write_text(
            f"run_fingerprint={'0' * 64}\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(EvidenceError, "exact shared run_fingerprint"):
            self.analyze_small(matrix)

    def test_marker_arm_is_bound_to_summary_identity_and_seed(self) -> None:
        matrix = self.write_matrix()
        self.write_marker(matrix, "A", arm_text=ARM_TEXT["B"])
        with self.assertRaisesRegex(EvidenceError, "does not match summary identity/seed"):
            self.analyze_small(matrix)

        matrix = self.write_matrix()
        self.write_marker(
            matrix, "A", arm_text="ttft_pred:cost_cachedisp_old:9"
        )
        with self.assertRaisesRegex(EvidenceError, "require seed==0"):
            self.analyze_small(matrix)

    def test_manifest_arm_order_and_optional_expected_fingerprint_are_bound(self) -> None:
        matrix = self.write_matrix()
        result = analyze(
            matrix,
            expected_n=100,
            equivalence_band_n=3,
            expected_fingerprint=FINGERPRINT,
            expected_manifest=TEST_MANIFEST_CONTRACT,
        )
        self.assertEqual(result["run_fingerprint"], FINGERPRINT)
        with self.assertRaisesRegex(EvidenceError, "!= expected"):
            analyze(
                matrix,
                expected_n=100,
                equivalence_band_n=3,
                expected_fingerprint="e" * 64,
                expected_manifest=TEST_MANIFEST_CONTRACT,
            )

        (matrix / "matrix_manifest.txt").write_text(
            f"run_fingerprint={FINGERPRINT}\narms={ARM_TEXT['A']} {ARM_TEXT['B']} "
            f"{ARM_TEXT['C']} {ARM_TEXT['K']} {ARM_TEXT['L']}\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(EvidenceError, "frozen arm order"):
            self.analyze_small(matrix)

    def test_whole_row_equivalence_boundary_is_inclusive_at_363(self) -> None:
        common = {
            "A": {"routed": 5000},
            "B": {"routed": 6000},
            "K": {"routed": 5500, "local_violations": 200},
            "L": {"local_violations": 10445},
        }
        inclusive = dict(common)
        inclusive["C"] = {"routed": 4637}
        result = analyze(
            self.write_matrix(inclusive, n=11605),
            expected_fingerprint=FINGERPRINT,
            expected_manifest=TEST_MANIFEST_CONTRACT,
        )
        gate = result["gates"]["a_vs_c_equivalence"]
        self.assertTrue(gate["pass"])
        self.assertEqual(gate["a_minus_c_n"], 363)
        self.assertAlmostEqual(gate["a_minus_c_pp"], 100.0 * 363 / 11605)
        self.assertEqual(gate["band_n"], 363)

        outside = dict(common)
        outside["C"] = {"routed": 4636}
        result = analyze(
            self.write_matrix(outside, n=11605),
            expected_fingerprint=FINGERPRINT,
            expected_manifest=TEST_MANIFEST_CONTRACT,
        )
        gate = result["gates"]["a_vs_c_equivalence"]
        self.assertFalse(gate["pass"])
        self.assertEqual(gate["a_minus_c_n"], 364)
        self.assertEqual(gate["classification"], "C_beats_A_beyond_band_reject")


if __name__ == "__main__":
    unittest.main()
