"""Focused stdlib tests for the staged E12 current-turn gate analyzer."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.analyze_ttft_current_turn_gate import (
    EvidenceError,
    _parse_manifest,
    analyze,
    recompute_run_fingerprint,
    render_json,
    render_markdown,
)


N = 6
TRACE_SHA = "1" * 64
TRACE_MANIFEST_SHA = "2" * 64
PROFILE_SHA = "3" * 64
ARMS = {
    "L": "anchor:all_local:0",
    "A": "ttft_pred:cost_cachedisp_old:0",
    "C": "ttft_pred:cost_disp_current:0",
}
IDENTITIES = {
    "L": ("all_local", "kv_gap", "cost_disp_current"),
    "A": ("nimbus", "ttft_pred", "cost_cachedisp_old"),
    "C": ("nimbus", "ttft_pred", "cost_disp_current"),
}
ROUTES = {"L": set(), "A": {0, 1}, "C": {1, 2, 3}}
EVENT_TIMES = {
    "L": ("2026-07-17T00:00:00Z", "2026-07-17T00:00:10Z", "2026-07-17T00:00:11Z"),
    "A": ("2026-07-17T00:00:31Z", "2026-07-17T00:00:40Z", "2026-07-17T00:00:41Z"),
    "C": ("2026-07-17T00:01:01Z", "2026-07-17T00:01:10Z", "2026-07-17T00:01:11Z"),
}
IN_PRICE = 0.08
OUT_PRICE = 0.28


def artifact(path: Path) -> dict:
    content = path.read_bytes()
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "nonempty_line_n": sum(bool(line.strip()) for line in content.splitlines()),
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(q / 100 * (len(ordered) - 1)))))]


class GateFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_stage(
        self, label: str, *, routes: set[int] | None = None,
        local_violations: set[int] | None = None,
    ) -> Path:
        routes = set(ROUTES[label] if routes is None else routes)
        if local_violations is None:
            local_violations = {4, 5} if label == "L" else set()
        directory = self.root / label
        directory.mkdir()
        fingerprint_placeholder = "FINGERPRINT_PENDING"
        stem = f"run_{label}"
        raw_path = directory / f"{stem}.jsonl"
        summary_path = directory / f"{stem}.summary.json"
        decisions_path = directory / f"{stem}.decisions.jsonl"
        marker_path = directory / f"{stem}.complete.json"

        rows = []
        local_ttfts = []
        total_cost = 0.0
        for request_id in range(N):
            prompt = 100 + request_id
            decode = 10 + request_id
            cloud = request_id in routes
            ttft = None if cloud else (6000.0 if request_id in local_violations else 1000.0 + request_id)
            cost = (prompt * IN_PRICE + decode * OUT_PRICE) / 1e6 if cloud else 0.0
            total_cost += cost
            if not cloud:
                local_ttfts.append(ttft)
            rows.append({
                "request_id": request_id,
                "endpoint": "cloud" if cloud else "local",
                "model": "qwen3-32b",
                "routed_only": cloud,
                "success": True,
                "error": None,
                "error_type": None,
                "ttft_ms": ttft,
                "prompt_tokens": prompt,
                "completion_tokens": decode,
                "scheduler_prompt_tokens": prompt,
                "scheduler_uncached_prompt_tokens": prompt,
                "scheduler_decode_tokens": decode,
                "payload_mode": "sharegpt_current_turn_retokenized",
                "cache_mode": "none",
                "cost_usd": cost,
            })
        raw_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

        policy, trigger, selector = IDENTITIES[label]
        decisions = []
        if label != "L":
            decisions.append({
                "decision_id": 1,
                "at_s": 1.0,
                "status": "applied",
                "trigger": trigger,
                "selector": selector,
                "proposed_victim_ids": sorted(routes),
                "applied_victim_ids": sorted(routes),
                "decision_ms": 0.2,
                "prediction_scope": "waiting_only",
                "prediction_model": "seq_slots_shared_prefill_lane_v1",
                "waiting_predicted_max_ttft_s": 7.0,
                "waiting_post_kick_max_ttft_s": 3.0,
            })
        decisions_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions),
            encoding="utf-8",
        )

        local_n = N - len(routes)
        violation_n = len(local_violations)
        local = {
            "n": local_n,
            "success": local_n,
            "routed_only": 0,
            "slo_violations": violation_n,
            "slo_measured_n": local_n,
            "slo_violation_pct": 100.0 * violation_n / local_n,
            "ttft_p50_ms": percentile(local_ttfts, 50),
            "ttft_p95_ms": percentile(local_ttfts, 95),
            "ttft_p99_ms": percentile(local_ttfts, 99),
        }
        config = {
            "scenario": "extreme_burst_1200",
            "seed": 0,
            "time_scale": 1.0,
            "max_inflight": 128,
            "max_tokens_override": None,
            "temperature": 0.0,
            "ignore_eos": True,
            "local_ignore_eos": True,
            "cloud_ignore_eos": True,
            "nimbus_trigger": trigger,
            "nimbus_selector": selector,
            "prefill_tput": 3000.0,
            "tpot_ms": 150.0,
            "first_token_overhead_ms": 400.0,
            "slo_s": 5.0,
            "timeout_s": 7200.0,
            "ttft_guard_ms": 1700.0,
            "nimbus_tick_ms": 250.0,
            "kv_capacity_tokens": 112000.0,
            "kv_hysteresis_fraction": 0.05,
            "cloud": "null",
            "cloud_url": None,
            "cloud_model": "qwen3-32b",
            "cloud_api_key_env": None,
            "cloud_max_concurrency": 32,
            "cloud_provider_order": None,
            "cloud_no_fallbacks": False,
            "cloud_stop_after_first_token": False,
            "local_url": "http://127.0.0.1:8010/v1/chat/completions",
            "local_model": "qwen3-32b",
            "in_price": IN_PRICE,
            "out_price": OUT_PRICE,
        }
        summary = {
            "policy": policy,
            "slo_s": 5.0,
            "overall": {
                "n": N,
                "success": N,
                "slo_measured_n": local_n,
                "cost_usd": total_cost,
            },
            "local": local,
            "cloud": {"n": len(routes), "success": len(routes), "routed_only": len(routes)},
            "pessimistic_combined": {
                "slo_violations": violation_n + len(routes),
                "slo_n": N,
                "slo_violation_pct": 100.0 * (violation_n + len(routes)) / N,
                "cost_usd": total_cost,
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
            "config": config,
            "queue": ({
                "nimbus_trigger": trigger,
                "nimbus_selector": selector,
                "nimbus_ticks": len(decisions),
                "nimbus_kicked": len(routes),
            } if label != "L" else {}),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        manifest_lines = [
            f"run_fingerprint={fingerprint_placeholder}",
            f"started_at=2026-07-17T00:00:0{len(list(self.root.iterdir()))}Z",
            "commit=test-commit",
            f"trace_sha256={TRACE_SHA} trace_manifest_sha256={TRACE_MANIFEST_SHA} trace_n={N}",
            f"profile_sha256={PROFILE_SHA}",
            "cache_mode=none server_pid=123 server_log=/scratch/server.log server_log_prefix_sha256=" + "4" * 64,
            "server_log_sha256_at_manifest=" + label.lower() * 64,
            "endpoint_version_sha256=" + "5" * 64 + " endpoint_models_identity_sha256=" + "6" * 64,
            "base_url=http://127.0.0.1:8010 chat_url=http://127.0.0.1:8010/v1/chat/completions model=qwen3-32b",
            "python=Python 3.12.0",
            "scenario=extreme_burst_1200 max_inflight=128 kv_cap=112000",
            "prefill_tput=3000 tpot_ms=150 first_token_overhead_ms=400 slo_s=5 timeout_s=7200 guard_ms=1700 tick_ms=250",
            f"temperature=0 ignore_eos=1 in_price={IN_PRICE} out_price={OUT_PRICE}",
            "local_ignore_eos=1 cloud_ignore_eos=1",
            "cloud=null cloud_url= cloud_model= cloud_api_key_env=",
            "cloud_max_concurrency=32 cloud_provider= cloud_no_fallbacks=0 cloud_stop_after_first_token=0",
            "real_cloud_expected_trace_n=11605 secret_value_recorded=false",
            "arm_order_mode=explicit arm_order_seed=0",
            f"arms={ARMS[label]} ",
            "evidence_scope=single-pass exploratory; repeat full matrices before paper claims",
        ]
        manifest_path = directory / "matrix_manifest.txt"
        manifest_path.write_text(
            "\n".join(manifest_lines) + "\n", encoding="utf-8"
        )
        parsed_manifest = _parse_manifest(manifest_path)
        fingerprint = recompute_run_fingerprint(parsed_manifest)
        manifest_path.write_text(
            manifest_path.read_text(encoding="utf-8").replace(
                fingerprint_placeholder, fingerprint
            ),
            encoding="utf-8",
        )
        start, finish, matrix_finish = EVENT_TIMES[label]
        (directory / "matrix_events.log").write_text(
            f"arm_started_at={start} arm={ARMS[label]} command=python -m router.run\n"
            f"arm_finished_at={finish} arm={ARMS[label]}\n"
            f"matrix_finished_at={matrix_finish} run_fingerprint={fingerprint}\n",
            encoding="utf-8",
        )
        self.rewrite_marker(directory, label)
        return directory

    def rewrite_marker(self, directory: Path, label: str) -> None:
        stem = f"run_{label}"
        manifest = _parse_manifest(directory / "matrix_manifest.txt")
        marker = {
            "schema_version": 2,
            "run_fingerprint": manifest["run_fingerprint"],
            "arm": ARMS[label],
            "artifacts": {
                "raw": artifact(directory / f"{stem}.jsonl"),
                "summary": artifact(directory / f"{stem}.summary.json"),
                "decisions": artifact(directory / f"{stem}.decisions.jsonl"),
            },
        }
        (directory / f"{stem}.complete.json").write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8"
        )

    def make_gate(self, **overrides) -> tuple[Path, Path, Path]:
        return tuple(
            self.make_stage(label, **overrides.get(label, {}))
            for label in ("L", "A", "C")
        )  # type: ignore[return-value]

    def analyze_gate(
        self, dirs: tuple[Path, Path, Path], *, min_cooldown_s: float = 20.0,
    ) -> dict:
        return analyze(
            *dirs,
            expected_n=N,
            expected_trace_sha256=TRACE_SHA,
            expected_trace_manifest_sha256=TRACE_MANIFEST_SHA,
            expected_profile_sha256=PROFILE_SHA,
            min_cooldown_s=min_cooldown_s,
        )

    def test_passing_gate_reports_text_free_aggregates(self) -> None:
        result = self.analyze_gate(self.make_gate())
        self.assertTrue(result["integrity_valid"])
        self.assertTrue(result["gate_pass"])
        self.assertEqual(result["arms"]["L"]["local_violation_n"], 2)
        self.assertEqual(result["arms"]["A"]["routed_n"], 2)
        self.assertEqual(result["arms"]["C"]["routed_n"], 3)
        self.assertEqual(result["victim_overlap"]["intersection_n"], 1)
        self.assertEqual(result["victim_overlap"]["union_n"], 4)
        self.assertEqual(result["victim_overlap"]["jaccard"], 0.25)
        self.assertEqual(result["deltas_c_minus_a"]["routed_n"], 1)
        timing = result["evidence"]["stage_timing"]
        self.assertEqual(timing["L"]["arm_started_at"], EVENT_TIMES["L"][0])
        self.assertEqual(timing["C"]["matrix_finished_at"], EVENT_TIMES["C"][2])
        self.assertEqual(
            timing["A"]["matrix_events_sha256"],
            hashlib.sha256((self.root / "A" / "matrix_events.log").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            result["evidence"]["cooldowns_s"],
            {
                "l_matrix_finish_to_a_start": 20.0,
                "a_matrix_finish_to_c_start": 20.0,
            },
        )
        self.assertFalse(result["text_payload_in_output"])
        self.assertNotIn("prompt_text", render_json(result))
        self.assertIn("Victim intersection/union", render_markdown(result))

    def test_zero_pressure_or_retained_violation_is_gate_failure_not_bad_evidence(self) -> None:
        result = self.analyze_gate(self.make_gate(L={"local_violations": set()}))
        self.assertFalse(result["gate_pass"])
        self.assertFalse(result["gates"]["l_pressure_observed"]["pass"])

        self.root = self.root / "second"
        self.root.mkdir()
        result = self.analyze_gate(
            self.make_gate(A={"local_violations": {2}})
        )
        self.assertFalse(result["gate_pass"])
        self.assertFalse(result["gates"]["a_retained_local_safety"]["pass"])

    def test_marker_tamper_is_rejected(self) -> None:
        dirs = self.make_gate()
        with (dirs[1] / "run_A.jsonl").open("a", encoding="utf-8") as output:
            output.write("\n")
        with self.assertRaisesRegex(EvidenceError, "does not exactly bind"):
            self.analyze_gate(dirs)

    def test_manifest_mismatch_and_wrong_arm_are_rejected(self) -> None:
        dirs = self.make_gate()
        manifest = dirs[2] / "matrix_manifest.txt"
        manifest.write_text(
            manifest.read_text().replace(f"profile_sha256={PROFILE_SHA}", "profile_sha256=" + "9" * 64),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(EvidenceError, "run_fingerprint|profile_sha256"):
            self.analyze_gate(dirs)

        self.root = self.root / "second"
        self.root.mkdir()
        dirs = self.make_gate()
        manifest = dirs[1] / "matrix_manifest.txt"
        manifest.write_text(
            manifest.read_text().replace(ARMS["A"], ARMS["C"]), encoding="utf-8"
        )
        with self.assertRaisesRegex(EvidenceError, "run_fingerprint|expected"):
            self.analyze_gate(dirs)

    def test_duplicate_ids_token_mismatch_and_victim_mismatch_are_rejected(self) -> None:
        dirs = self.make_gate()
        raw = dirs[1] / "run_A.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        rows[1]["request_id"] = rows[0]["request_id"]
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.rewrite_marker(dirs[1], "A")
        with self.assertRaisesRegex(EvidenceError, "not unique"):
            self.analyze_gate(dirs)

        self.root = self.root / "second"
        self.root.mkdir()
        dirs = self.make_gate()
        raw = dirs[2] / "run_C.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        local = next(row for row in rows if row["endpoint"] == "local")
        local["scheduler_prompt_tokens"] += 1
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.rewrite_marker(dirs[2], "C")
        with self.assertRaisesRegex(EvidenceError, "prompt tokens are not exact"):
            self.analyze_gate(dirs)

        self.root = self.root / "third"
        self.root.mkdir()
        dirs = self.make_gate()
        decisions = dirs[1] / "run_A.decisions.jsonl"
        row = json.loads(decisions.read_text())
        row["applied_victim_ids"] = [0]
        decisions.write_text(json.dumps(row) + "\n", encoding="utf-8")
        self.rewrite_marker(dirs[1], "A")
        with self.assertRaisesRegex(EvidenceError, "exact proposed|do not equal NullCloud"):
            self.analyze_gate(dirs)

    def test_fingerprint_binds_all_runner_inputs_and_requires_them(self) -> None:
        dirs = self.make_gate()
        manifest = dirs[1] / "matrix_manifest.txt"
        manifest.write_text(
            manifest.read_text().replace("arm_order_seed=0", "arm_order_seed=9"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(EvidenceError, "run_fingerprint"):
            self.analyze_gate(dirs)

        self.root = self.root / "missing"
        self.root.mkdir()
        dirs = self.make_gate()
        manifest = dirs[2] / "matrix_manifest.txt"
        manifest.write_text(
            manifest.read_text().replace(" cloud_provider=", " omitted_provider="),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(EvidenceError, "cloud_provider"):
            self.analyze_gate(dirs)

    def test_decision_status_timing_and_prediction_contracts(self) -> None:
        dirs = self.make_gate()
        decisions = dirs[1] / "run_A.decisions.jsonl"
        row = json.loads(decisions.read_text())
        row["status"] = "stale_retry"
        decisions.write_text(json.dumps(row) + "\n", encoding="utf-8")
        self.rewrite_marker(dirs[1], "A")
        with self.assertRaisesRegex(EvidenceError, "only applied status"):
            self.analyze_gate(dirs)

        self.root = self.root / "timing"
        self.root.mkdir()
        dirs = self.make_gate()
        decisions = dirs[2] / "run_C.decisions.jsonl"
        row = json.loads(decisions.read_text())
        row["decision_ms"] = -0.1
        decisions.write_text(json.dumps(row) + "\n", encoding="utf-8")
        self.rewrite_marker(dirs[2], "C")
        with self.assertRaisesRegex(EvidenceError, "decision_ms must be nonnegative"):
            self.analyze_gate(dirs)

        self.root = self.root / "prediction"
        self.root.mkdir()
        dirs = self.make_gate()
        decisions = dirs[1] / "run_A.decisions.jsonl"
        row = json.loads(decisions.read_text())
        row["prediction_scope"] = "all_rows"
        decisions.write_text(json.dumps(row) + "\n", encoding="utf-8")
        self.rewrite_marker(dirs[1], "A")
        with self.assertRaisesRegex(EvidenceError, "prediction_scope"):
            self.analyze_gate(dirs)

    def test_nonempty_applied_stale_bounded_is_valid(self) -> None:
        dirs = self.make_gate()
        decisions = dirs[1] / "run_A.decisions.jsonl"
        row = json.loads(decisions.read_text())
        row["status"] = "applied_stale_bounded"
        decisions.write_text(json.dumps(row) + "\n", encoding="utf-8")
        self.rewrite_marker(dirs[1], "A")

        result = self.analyze_gate(dirs)
        self.assertTrue(result["gate_pass"])
        self.assertEqual(result["arms"]["A"]["applied_victim_n"], 2)

    def test_matrix_events_tamper_is_rejected(self) -> None:
        dirs = self.make_gate()
        events = dirs[1] / "matrix_events.log"
        events.write_text(
            events.read_text().replace(
                f"arm={ARMS['A']}", f"arm={ARMS['C']}", 1
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(EvidenceError, "arm does not match"):
            self.analyze_gate(dirs)

        self.root = self.root / "extra_line"
        self.root.mkdir()
        dirs = self.make_gate()
        events = dirs[2] / "matrix_events.log"
        events.write_text(events.read_text() + "unexpected=extra\n", encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "exactly three"):
            self.analyze_gate(dirs)

    def test_stage_cooldown_is_enforced_and_parameterized(self) -> None:
        dirs = self.make_gate()
        events = dirs[1] / "matrix_events.log"
        events.write_text(
            events.read_text().replace(
                "arm_started_at=2026-07-17T00:00:31Z",
                "arm_started_at=2026-07-17T00:00:30Z",
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(EvidenceError, "require >= 20.0s"):
            self.analyze_gate(dirs)
        self.assertTrue(self.analyze_gate(dirs, min_cooldown_s=19)["gate_pass"])
        with self.assertRaisesRegex(EvidenceError, "finite and nonnegative"):
            self.analyze_gate(dirs, min_cooldown_s=-1)

    def test_raw_model_payload_mode_and_nonnegative_tokens_are_bound(self) -> None:
        mutations = (
            ("model", "wrong-model", "model does not match"),
            ("payload_mode", "synthetic", "payload_mode must be"),
            ("scheduler_prompt_tokens", -1, "must be nonnegative"),
        )
        for index, (field, value, message) in enumerate(mutations):
            with self.subTest(field=field):
                if index:
                    self.root = self.root / f"case_{index}"
                    self.root.mkdir()
                dirs = self.make_gate()
                raw = dirs[2] / "run_C.jsonl"
                rows = [json.loads(line) for line in raw.read_text().splitlines()]
                rows[-1][field] = value
                raw.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                self.rewrite_marker(dirs[2], "C")
                with self.assertRaisesRegex(EvidenceError, message):
                    self.analyze_gate(dirs)


if __name__ == "__main__":
    unittest.main()
