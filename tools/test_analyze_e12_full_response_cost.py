"""Tests for the text-free E12 selected-cohort full-cap cost analyzer."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import analyze_e12_full_response_cost as analyzer


SECRET_PROMPT = "private prompt that must never enter analyzer output"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class Fixture:
    n = 4

    def __init__(self, root: Path) -> None:
        self.root = root
        self.trace = root / "trace.jsonl"
        self.trace_manifest = root / "trace.manifest.json"
        self.price = root / "price.json"
        self.a_raw = root / "a.jsonl"
        self.c_raw = root / "c.jsonl"
        self.final_audit = root / "audit.json"

        # Deliberately not arrival-ordered, exercising load_trace's stable sort.
        self.source_rows = [
            self.trace_row(analyzer.SCENARIO_START + 10, 10, 2, 100),
            self.trace_row(analyzer.SCENARIO_START, 20, 3, 101),
            self.trace_row(analyzer.SCENARIO_START + 5, 30, 4, 102),
            self.trace_row(analyzer.SCENARIO_START + 5, 40, 5, 103),
        ]
        dump_jsonl(self.trace, self.source_rows)
        ordered = sorted(enumerate(self.source_rows), key=lambda item: (item[1]["arrived_at"], item[0]))
        self.ordered = [row for _, row in ordered]
        dump_json(
            self.trace_manifest,
            {
                "schema_version": 1,
                "output_sha256": sha256(self.trace),
                "n": self.n,
                "scenario": analyzer.SCENARIO,
                "payload_mode": analyzer.PAYLOAD_MODE,
                "cache_mode": analyzer.CACHE_MODE,
                "max_decode_tokens": analyzer.MAX_DECODE_TOKENS,
                "max_context_tokens": analyzer.MAX_CONTEXT_TOKENS,
                "overflow_policy": "drop",
                "actual_prompt_tokens": {
                    "sum": sum(int(row["num_prefill_tokens"]) for row in self.source_rows)
                },
                "output_decode_tokens": {
                    "sum": sum(int(row["num_decode_tokens"]) for row in self.source_rows)
                },
            },
        )
        dump_json(
            self.price,
            {
                "schema_version": 1,
                "status": "pass",
                "captured_at_utc": "2026-07-20T00:00:00Z",
                "http_status": 200,
                "model": analyzer.MODEL,
                "provider": analyzer.PROVIDER,
                "matching_endpoint_count": 1,
                "context_length": analyzer.MAX_CONTEXT_TOKENS,
                "prompt_price_per_token_usd": "0.00000008",
                "completion_price_per_token_usd": "0.00000028",
                "prompt_price_per_million_usd": "0.08",
                "completion_price_per_million_usd": "0.28",
                "request_price_per_request_usd": "0",
                "request_price_source": "absent_not_advertised",
            },
        )
        # A routes IDs 0 and 2; ID 2 is a failed cloud row. C routes ID 1.
        self.a_rows = self.raw_rows({0, 2}, failed={2})
        self.c_rows = self.raw_rows({1})
        dump_jsonl(self.a_raw, self.a_rows)
        dump_jsonl(self.c_raw, self.c_rows)
        self.refresh_audit()

    @staticmethod
    def trace_row(arrived: int, prompt: int, decode: int, source: int) -> dict[str, object]:
        return {
            "arrived_at": arrived,
            "num_prefill_tokens": prompt,
            "uncached_prompt_tokens": prompt,
            "num_cached_tokens": 0,
            "num_decode_tokens": decode,
            "source_request_index": source,
            "prompt_text": f"{SECRET_PROMPT} {source}",
            "payload_mode": analyzer.PAYLOAD_MODE,
            "cache_mode": analyzer.CACHE_MODE,
        }

    def raw_rows(self, cloud: set[int], failed: set[int] | None = None) -> list[dict[str, object]]:
        failed = failed or set()
        rows = []
        for request_id, trace_row in enumerate(self.ordered):
            rows.append(
                {
                    "request_id": request_id,
                    "endpoint": "cloud" if request_id in cloud else "local",
                    "success": request_id not in failed,
                    "payload_mode": analyzer.PAYLOAD_MODE,
                    "cache_mode": analyzer.CACHE_MODE,
                    "scheduler_prompt_tokens": trace_row["num_prefill_tokens"],
                    "scheduler_uncached_prompt_tokens": trace_row[
                        "uncached_prompt_tokens"
                    ],
                    "scheduler_decode_tokens": trace_row["num_decode_tokens"],
                    "source_request_index": trace_row["source_request_index"],
                }
            )
        return rows

    def refresh_audit(self) -> str:
        def stage(label: str, rows: list[dict[str, object]], raw: Path) -> dict[str, object]:
            cloud = [row for row in rows if row["endpoint"] == "cloud"]
            successes = sum(row["success"] is True for row in cloud)
            return {
                "arm": analyzer.ARM_IDENTITIES[label],
                "n": self.n,
                "cloud_n": len(cloud),
                "cloud_success_n": successes,
                "cloud_failure_n": len(cloud) - successes,
                "applied_victim_n": len(cloud),
                "applied_victims_equal_cloud_ids": True,
                "artifacts": {"raw_sha256": sha256(raw)},
            }

        audit = {
            "schema_version": 1,
            "analysis": "E12 live ShareGPT current-turn A/C audit",
            "verdict": "pass",
            "integrity_valid": True,
            "text_payload_in_output": False,
            "evidence": {
                "trace_n": self.n,
                "trace_sha256": sha256(self.trace),
                "trace_manifest_sha256": sha256(self.trace_manifest),
                "price_snapshot_sha256": sha256(self.price),
            },
            "stages": {
                "A": stage("A", self.a_rows, self.a_raw),
                "C": stage("C", self.c_rows, self.c_raw),
            },
        }
        dump_json(self.final_audit, audit)
        return sha256(self.final_audit)

    def analyze(self) -> dict[str, object]:
        with mock.patch.object(analyzer, "TRACE_N", self.n):
            return analyzer.analyze(
                trace_path=self.trace,
                trace_manifest_path=self.trace_manifest,
                price_snapshot_path=self.price,
                final_audit_path=self.final_audit,
                expected_final_audit_sha256=sha256(self.final_audit),
                a_raw_path=self.a_raw,
                c_raw_path=self.c_raw,
            )


class FullResponseCostTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = Fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_happy_path_is_exact_and_text_free(self) -> None:
        result = self.fixture.analyze()
        a = result["arms"]["A"]
        c = result["arms"]["C"]
        # Arrival-sort IDs make A select P/D=(20/3)+(40/5), C=(30/4).
        self.assertEqual(a["route_n"], 2)
        self.assertEqual(a["prompt_tokens"], 60)
        self.assertEqual(a["capped_decode_tokens"], 8)
        self.assertEqual(a["modeled_full_cap_cost_usd"], "0.00000704")
        self.assertEqual(c["modeled_full_cap_cost_usd"], "0.00000352")
        self.assertEqual(result["comparison"]["cloud_id_intersection_n"], 0)
        encoded = json.dumps(result, sort_keys=True)
        self.assertNotIn(SECRET_PROMPT, encoded)
        self.assertNotIn(str(self.root), encoded)
        self.assertNotIn("prompt_text", encoded)
        self.assertNotIn("request_id", encoded)

    def test_failed_cloud_route_is_in_primary_and_excluded_from_sensitivity(self) -> None:
        result = self.fixture.analyze()
        arm = result["arms"]["A"]
        self.assertEqual(arm["route_n"], 2)
        self.assertEqual(arm["failure_n"], 1)
        self.assertEqual(arm["modeled_full_cap_cost_usd"], "0.00000704")
        sensitivity = arm["successful_cloud_rows_only_sensitivity"]
        self.assertEqual(sensitivity["route_n"], 1)
        self.assertEqual(sensitivity["prompt_tokens"], 20)
        self.assertEqual(sensitivity["capped_decode_tokens"], 3)
        self.assertEqual(sensitivity["modeled_full_cap_cost_usd"], "0.00000244")

    def test_expected_final_audit_sha_is_required(self) -> None:
        with mock.patch.object(analyzer, "TRACE_N", self.fixture.n):
            with self.assertRaisesRegex(analyzer.EvidenceError, "expected SHA256"):
                analyzer.analyze(
                    trace_path=self.fixture.trace,
                    trace_manifest_path=self.fixture.trace_manifest,
                    price_snapshot_path=self.fixture.price,
                    final_audit_path=self.fixture.final_audit,
                    expected_final_audit_sha256="0" * 64,
                    a_raw_path=self.fixture.a_raw,
                    c_raw_path=self.fixture.c_raw,
                )

    def test_raw_hash_mismatch_fails_closed(self) -> None:
        self.fixture.a_raw.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(analyzer.EvidenceError, "does not match"):
            self.fixture.analyze()

    def test_scheduler_token_drift_fails_even_when_audit_rebinds_raw(self) -> None:
        self.fixture.a_rows[0]["scheduler_decode_tokens"] = 999
        dump_jsonl(self.fixture.a_raw, self.fixture.a_rows)
        self.fixture.refresh_audit()
        with self.assertRaisesRegex(analyzer.EvidenceError, "tokens do not match"):
            self.fixture.analyze()

    def test_duplicate_request_id_fails_even_when_audit_rebinds_raw(self) -> None:
        self.fixture.a_rows[1]["request_id"] = self.fixture.a_rows[0]["request_id"]
        dump_jsonl(self.fixture.a_raw, self.fixture.a_rows)
        self.fixture.refresh_audit()
        with self.assertRaisesRegex(analyzer.EvidenceError, "request IDs are invalid"):
            self.fixture.analyze()

    def test_price_drift_fails_even_when_audit_rebinds_snapshot(self) -> None:
        price = json.loads(self.fixture.price.read_text(encoding="utf-8"))
        price["completion_price_per_million_usd"] = "0.29"
        dump_json(self.fixture.price, price)
        self.fixture.refresh_audit()
        with self.assertRaisesRegex(analyzer.EvidenceError, "changed from the frozen"):
            self.fixture.analyze()

    def test_scenario_provenance_drift_fails_even_when_audit_rebinds_manifest(self) -> None:
        manifest = json.loads(self.fixture.trace_manifest.read_text(encoding="utf-8"))
        manifest["scenario"] = "full"
        dump_json(self.fixture.trace_manifest, manifest)
        self.fixture.refresh_audit()
        with self.assertRaisesRegex(analyzer.EvidenceError, "scenario is inconsistent"):
            self.fixture.analyze()

    def test_output_is_new_mode_0600_and_refuses_overwrite(self) -> None:
        payload = b'{"safe":true}\n'
        output = self.root / "derived.json"
        analyzer._write_new(output, payload)
        self.assertEqual(output.read_bytes(), payload)
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(analyzer.EvidenceError, "already exists"):
            analyzer._write_new(output, payload)


if __name__ == "__main__":
    unittest.main()
