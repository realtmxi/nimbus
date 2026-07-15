"""Focused stdlib tests for analyze_ttft_violation_context.py."""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from tools.analyze_ttft_violation_context import (
    EvidenceError,
    analyze,
    main,
    render_json,
    render_markdown,
)


MODEL = "seq_slots_shared_prefill_lane_v1"
DEPLOYMENT_MODEL = "synthetic-model"
FINGERPRINT = "synthetic-run-fingerprint"
ARM = "ttft_pred:newest:0"
SERVER_PREFIX = b"synthetic server boot\n"
SERVER_PREFIX_SHA = hashlib.sha256(SERVER_PREFIX).hexdigest()


def profile() -> dict:
    return {
        "valid": True,
        "model": DEPLOYMENT_MODEL,
        "predictor_model": MODEL,
        "kv_capacity_tokens": 100,
        "cache_mode_required": "none",
        "server_pid": 123,
        "server_log": "synthetic-server.log",
        "server_log_bytes_at_start": len(SERVER_PREFIX),
        "server_log_sha256_at_start": SERVER_PREFIX_SHA,
        "profile_config": {"decode_tokens": 20},
        "predictor_calibration": {
            "target_slo_s": 5.0,
            "recommended_tpot_ms": 100.0,
            "recommended_first_token_overhead_ms": 50.0,
            "recommended_ttft_guard_ms": 1000.0,
            "nimbus_tick_guard_ms": 250.0,
            "recommended_prefill_tput_tokens_per_s": 1000.0,
            "scope": "synthetic no-queue profile",
        },
        "cells": [
            {
                "offered_concurrency": 1,
                "prompt_tokens": 10,
                "actual_estimated_peak_tokens_max": 30,
            },
            {
                "offered_concurrency": 2,
                "prompt_tokens": 20,
                "actual_estimated_peak_tokens_max": 60,
            },
        ],
    }


def local_row(
    request_id: int,
    arrival_s: float,
    queue_ms: float,
    service_ms: float,
    e2e_ms: float,
    prompt: int,
    decode: int,
    tpot_ms: float | None,
) -> dict:
    return {
        "request_id": request_id,
        "relative_arrival_s": arrival_s,
        "endpoint": "local",
        "success": True,
        "cache_mode": "none",
        "queue_delay_ms": queue_ms,
        "service_ttft_ms": service_ms,
        "ttft_ms": queue_ms + service_ms,
        "e2e_ms": e2e_ms,
        "tpot_ms": tpot_ms,
        "prompt_tokens": prompt,
        "completion_tokens": decode,
        "scheduler_prompt_tokens": prompt,
        "scheduler_uncached_prompt_tokens": prompt,
        "scheduler_decode_tokens": decode,
    }


def cloud_row(request_id: int, arrival_s: float, prompt: int, decode: int) -> dict:
    return {
        "request_id": request_id,
        "relative_arrival_s": arrival_s,
        "endpoint": "cloud",
        "success": True,
        "cache_mode": "none",
        "queue_delay_ms": 500.0,
        "service_ttft_ms": None,
        "ttft_ms": None,
        "e2e_ms": None,
        "tpot_ms": None,
        "prompt_tokens": prompt,
        "completion_tokens": decode,
        "scheduler_prompt_tokens": prompt,
        "scheduler_uncached_prompt_tokens": prompt,
        "scheduler_decode_tokens": decode,
    }


def decision(
    decision_id: int,
    at_s: float,
    *,
    status: str = "no_op",
    predicted: float = 3.0,
    post: float = 3.0,
    applied: list[int] | None = None,
) -> dict:
    applied = [] if applied is None else applied
    return {
        "decision_id": decision_id,
        "at_s": at_s,
        "status": status,
        "trigger": "ttft_pred",
        "selector": "newest",
        "prediction_model": MODEL,
        "prediction_scope": "waiting_only",
        "snapshot_n": 2,
        "snapshot_hash": f"hash-{decision_id}",
        "inflight_n": 2,
        "inflight_decode_n": 1,
        "inflight_prefill_n": 1,
        "inflight_prefill_tokens": 20,
        "waiting_predicted_max_ttft_s": predicted,
        "waiting_post_kick_max_ttft_s": post,
        "proposed_victim_ids": applied,
        "applied_victim_ids": applied,
        "decision_ms": 0.2,
    }


def summary() -> dict:
    return {
        "policy": "nimbus",
        "slo_s": 5.0,
        "config": {
            "nimbus_trigger": "ttft_pred",
            "nimbus_selector": "newest",
            "seed": 0,
            "slo_s": 5.0,
            "ttft_guard_ms": 1000.0,
            "prefill_tput": 1000.0,
            "tpot_ms": 100.0,
            "first_token_overhead_ms": 50.0,
            "kv_capacity_tokens": 100.0,
            "local_model": DEPLOYMENT_MODEL,
        },
        "queue": {
            "nimbus_trigger": "ttft_pred",
            "nimbus_selector": "newest",
            "nimbus_prediction_model": MODEL,
            "nimbus_prediction_scope": "waiting_only",
        },
        "overall": {"n": 5, "success": 5},
        "local": {
            "n": 4,
            "success": 4,
            "slo_measured_n": 4,
            "slo_violations": 2,
        },
        "cloud": {"n": 1, "success": 1},
    }


def artifact(path: Path) -> dict:
    content = path.read_bytes()
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "nonempty_line_n": sum(bool(line.strip()) for line in content.splitlines()),
    }


class TestViolationContext(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "arm.jsonl"
        self.decisions = self.root / "arm.decisions.jsonl"
        self.summary = self.root / "arm.summary.json"
        self.marker = self.root / "arm.complete.json"
        self.manifest = self.root / "matrix_manifest.txt"
        self.profile = self.root / "profile.json"
        self.log = self.root / "server.log"

        self.raw_rows = [
            local_row(1, 0.0, 0.0, 1000.0, 10000.0, 20, 20, 120.0),
            local_row(2, 1.0, 100.0, 4500.0, 9000.0, 20, 20, 130.0),
            local_row(3, 2.0, 100.0, 5100.0, 8000.0, 30, 30, 150.0),
            local_row(4, 3.0, 1000.0, 4500.0, 7000.0, 25, 25, 200.0),
            cloud_row(5, 3.0, 40, 40),
        ]
        self.decision_rows = [
            decision(1, 0.0, predicted=1.0, post=1.0),
            decision(2, 1.09, predicted=2.0, post=2.0),
            decision(3, 2.09, predicted=3.1, post=3.1),
            decision(
                4, 3.5, status="applied", predicted=6.0, post=3.0, applied=[5]
            ),
        ]
        self.summary_payload = summary()
        self._write_log()
        self._write_inputs()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_log(self, prefix: bytes = SERVER_PREFIX) -> None:
        metrics = "\n".join([
            "INFO 01-01 00:00:00 Engine 000: Avg prompt throughput: 1000.0 tokens/s, Avg generation throughput: 10.0 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 10.0%, Prefix cache hit rate: 0.0%",
            "INFO 01-01 00:00:02 Engine 000: Avg prompt throughput: 800.0 tokens/s, Avg generation throughput: 20.0 tokens/s, Running: 3 reqs, Waiting: 2 reqs, GPU KV cache usage: 98.0%, Prefix cache hit rate: 0.0%",
            "INFO 01-01 00:00:05 Engine 000: Avg prompt throughput: 700.0 tokens/s, Avg generation throughput: 30.0 tokens/s, Running: 4 reqs, Waiting: 3 reqs, GPU KV cache usage: 99.0%, Prefix cache hit rate: 0.0%",
            "INFO 01-01 00:00:10 Engine 000: Avg prompt throughput: 600.0 tokens/s, Avg generation throughput: 40.0 tokens/s, Running: 4 reqs, Waiting: 4 reqs, GPU KV cache usage: 100.0%, Prefix cache hit rate: 0.0%",
        ]) + "\n"
        self.log.write_bytes(prefix + metrics.encode())

    def _write_marker(self) -> None:
        payload = {
            "schema_version": 1,
            "run_fingerprint": FINGERPRINT,
            "arm": ARM,
            "artifacts": {
                "raw": artifact(self.raw),
                "summary": artifact(self.summary),
                "decisions": artifact(self.decisions),
            },
        }
        self.marker.write_text(json.dumps(payload), encoding="utf-8")

    def _write_inputs(self) -> None:
        self.raw.write_text(
            "\n".join(json.dumps(row) for row in self.raw_rows) + "\n",
            encoding="utf-8",
        )
        self.decisions.write_text(
            "\n".join(json.dumps(row) for row in self.decision_rows) + "\n",
            encoding="utf-8",
        )
        self.summary.write_text(json.dumps(self.summary_payload), encoding="utf-8")
        self.profile.write_text(json.dumps(profile()), encoding="utf-8")
        self._write_marker()
        profile_sha = hashlib.sha256(self.profile.read_bytes()).hexdigest()
        self.manifest.write_text(
            "\n".join([
                f"run_fingerprint={FINGERPRINT}",
                "trace_sha256=" + "1" * 64 + " trace_manifest_sha256="
                + "2" * 64 + " trace_n=5",
                f"profile_sha256={profile_sha}",
                "cache_mode=none server_pid=123 server_log=synthetic-server.log "
                f"server_log_prefix_sha256={SERVER_PREFIX_SHA}",
                f"model={DEPLOYMENT_MODEL}",
                "prefill_tput=1000.0 tpot_ms=100.0 first_token_overhead_ms=50.0 "
                "slo_s=5.0 guard_ms=1000.0 tick_ms=250.0",
                "scenario=synthetic max_inflight=4 kv_cap=100.0",
                f"arms={ARM}",
            ]) + "\n",
            encoding="utf-8",
        )

    def _analyze(self, **kwargs):
        return analyze(
            self.raw,
            self.decisions,
            self.profile,
            summary_path=self.summary,
            marker_path=self.marker,
            manifest_path=self.manifest,
            **kwargs,
        )

    def test_reconstructs_violation_mechanism_and_bound_support_domain(self) -> None:
        result = self._analyze()
        self.assertEqual(result["counts"]["local_violation_n"], 2)
        queue = result["queue_vs_service"]
        self.assertEqual(queue["component_exceeds_slo"]["queue_delay_n"], 0)
        self.assertEqual(queue["component_exceeds_slo"]["service_ttft_n"], 1)
        classes = queue["mutually_exclusive_classification"]
        self.assertEqual(classes["service_component_only_n"], 1)
        self.assertEqual(classes["neither_component_but_sum_n"], 1)

        context = result["dispatch_load_context"]
        self.assertEqual(context["violations"]["active_n"]["min"], 3.0)
        self.assertEqual(
            context["violations"]["planned_peak_commitment_tokens"]["min"], 140.0
        )
        self.assertIn("pre_first_token_n", context["violations"])
        self.assertIn("post_first_token_n", context["violations"])
        self.assertNotIn("active_prefill_n", context["violations"])
        self.assertIn("not engine scheduler phases", context["reconstruction_scope"])
        support = context["profile_support"]
        self.assertEqual(support["violation_commitment_above_profile_max_n"], 2)
        self.assertEqual(
            support["violation_commitment_above_declared_kv_capacity_n"], 2
        )
        self.assertIn("profile_prompt_cell_target_points", support)

        decision_context = result["decisions"]["violation_closest_pre_dispatch_context"]
        self.assertEqual(decision_context["mapped_violation_n"], 2)
        self.assertEqual(decision_context["status_counts"], {"applied": 1, "no_op": 1})
        self.assertFalse(result["selector_replay"]["available"])
        temporal = result["temporal_and_cohort_concentration"]
        self.assertEqual(temporal["clustering_method"], "deterministic_single_link_chaining_by_arrival_gap")
        self.assertIn("do not", temporal["interpretation_caveat"])
        self.assertAlmostEqual(
            result["tpot_comparison"]["violation_p50_over_calibrated_ratio"], 1.75
        )

        validation = result["validation"]
        self.assertTrue(validation["completion_marker_exactly_matches_raw_summary_decisions"])
        self.assertTrue(validation["supplied_profile_sha256_matches_matrix_manifest"])
        encoded = render_json(result)
        self.assertNotIn(self.temp.name, encoded)
        markdown = render_markdown(result)
        self.assertIn("Selector replay", markdown)

    def test_wrong_profile_and_tampered_marker_are_rejected(self) -> None:
        wrong = profile()
        wrong["kv_capacity_tokens"] = 101
        self.profile.write_text(json.dumps(wrong), encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "manifest profile_sha256"):
            self._analyze()

        self._write_inputs()
        marker = json.loads(self.marker.read_text(encoding="utf-8"))
        marker["artifacts"]["raw"]["sha256"] = "0" * 64
        self.marker.write_text(json.dumps(marker), encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "completion marker does not exactly match"):
            self._analyze()

    def test_summary_runtime_config_is_cross_checked_after_valid_hash_binding(self) -> None:
        self.summary_payload["config"]["tpot_ms"] = 99.0
        self.summary.write_text(json.dumps(self.summary_payload), encoding="utf-8")
        self._write_marker()
        with self.assertRaisesRegex(EvidenceError, "config.tpot_ms"):
            self._analyze()

    def test_server_alignment_is_prefix_bound_and_labels_gauge_conservatively(self) -> None:
        unavailable = self._analyze(server_log=self.log)["server_alignment"]
        self.assertFalse(unavailable["available"])
        self.assertIn("server log clock", unavailable["reason"])

        aligned = self._analyze(
            server_log=self.log,
            server_log_arm_start="01-01 00:00:00",
            server_log_max_gap_s=6.0,
        )["server_alignment"]
        self.assertTrue(aligned["available"])
        self.assertTrue(aligned["gauge_semantics_unverified"])
        self.assertTrue(aligned["gauge_semantics_architecture_dependent"])
        self.assertIn("deployment-specific gauge probe", aligned["gauge_interpretation_caveat"])
        first = aligned["first_token"]
        gauge = first["engine_reported_gpu_cache_usage_gauge_pct"]
        self.assertEqual(first["aligned_violation_n"], 2)
        self.assertGreaterEqual(gauge["min"], 98.0)
        self.assertNotIn("gpu_kv_usage_pct", first)

        wrong_log = self.root / "wrong.log"
        wrong_log.write_bytes(b"unrelated server log\n" + self.log.read_bytes())
        with self.assertRaisesRegex(EvidenceError, "server-log prefix does not match"):
            self._analyze(
                server_log=wrong_log,
                server_log_arm_start="01-01 00:00:00",
            )

    def test_rejects_raw_and_decision_semantic_mismatch(self) -> None:
        self.raw_rows[2]["ttft_ms"] += 1.0
        self._write_inputs()
        with self.assertRaisesRegex(EvidenceError, "ttft_ms !="):
            self._analyze()

        self.raw_rows[2]["ttft_ms"] -= 1.0
        self.decision_rows[-1]["applied_victim_ids"] = []
        self._write_inputs()
        with self.assertRaisesRegex(EvidenceError, "applied/cloud request-id mismatch"):
            self._analyze()

    def test_cli_requires_and_uses_all_binding_artifacts(self) -> None:
        argv = [
            "--raw", str(self.raw),
            "--decisions", str(self.decisions),
            "--summary", str(self.summary),
            "--marker", str(self.marker),
            "--manifest", str(self.manifest),
            "--profile", str(self.profile),
            "--json",
        ]
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = main(argv)
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(stdout.getvalue())["counts"]["local_violation_n"], 2)

        self.decision_rows[0]["trigger"] = "kv_gap"
        self._write_inputs()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            rc = main(argv)
        self.assertEqual(rc, 2)
        self.assertIn("expected trigger 'ttft_pred'", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
