"""Tests for secret-free OpenRouter usage snapshots."""
from __future__ import annotations

import io
import hashlib
import json
import tempfile
import unittest
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from tools.openrouter_usage_snapshot import (
    SnapshotError,
    capture_usage_snapshot,
    snapshot_from_environment,
    write_json_atomic,
)
from tools.openrouter_key_contract import E12_MARKETPLACE_KEY_CONTRACT


SECRET = "TEST-OPENROUTER-KEY-DO-NOT-LEAK"
RAW_LABEL = "raw-provider-key-label-must-not-leak"
ENV_LABEL = "PRIVATE_OPENROUTER_KEY_LABEL_MUST_NOT_LEAK"


class FakeResponse:
    def __init__(self, payload: dict, *, url: str, status: int = 200):
        self.status = status
        self._url = url
        self._raw = json.dumps(payload).encode("utf-8")

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self._url

    def read(self, size: int = -1) -> bytes:
        return self._raw[:size]

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class TrackingBody(io.BytesIO):
    def __init__(self, value: bytes):
        super().__init__(value)
        self.read_called = False
        self.close_called = False

    def read(self, *args, **kwargs):
        self.read_called = True
        return super().read(*args, **kwargs)

    def close(self):
        self.close_called = True
        super().close()


class RecordingOpener:
    def __init__(self):
        self.requests = []

    def __call__(self, request, *, timeout):
        self.requests.append((request, timeout))
        if request.full_url.endswith("/api/v1/key"):
            return FakeResponse({
                "data": {
                    "label": RAW_LABEL,
                    "usage": 1.25,
                    "limit": 3.0,
                    "limit_remaining": 1.75,
                    "is_free_tier": False,
                    "is_management_key": False,
                    "is_provisioning_key": False,
                    "limit_reset": None,
                    "include_byok_in_limit": True,
                    "expires_at": None,
                }
            }, url=request.full_url)
        if request.full_url.endswith("/api/v1/credits"):
            return FakeResponse({
                "data": {
                    "total_usage": 4.5,
                    "total_credits": 10.0,
                    "private_note": SECRET,
                }
            }, url=request.full_url)
        raise AssertionError(request.full_url)


class TestOpenRouterUsageSnapshot(unittest.TestCase):
    def test_explicit_e12_marketplace_mode_keeps_real_byok_metadata(self):
        class MarketplaceOpener(RecordingOpener):
            def __call__(self, request, *, timeout):
                if request.full_url.endswith("/api/v1/key"):
                    return FakeResponse({"data": {
                        "usage": 1.9,
                        "limit": 5,
                        "limit_remaining": 3.1,
                        "limit_reset": None,
                        "include_byok_in_limit": False,
                        "byok_usage": 0.25,
                        "is_free_tier": False,
                        "is_management_key": False,
                        "is_provisioning_key": False,
                        "expires_at": None,
                    }}, url=request.full_url)
                return super().__call__(request, timeout=timeout)

        result = capture_usage_snapshot(
            api_key=SECRET,
            opener=MarketplaceOpener(),
            key_contract_mode=E12_MARKETPLACE_KEY_CONTRACT,
        )
        self.assertEqual(result["key"]["limit"], 5)
        self.assertIs(result["key"]["include_byok_in_limit"], False)
        self.assertEqual(result["key"]["byok_usage"], 0.25)

        broken = MarketplaceOpener()
        original = broken.__call__
        # Strict mode remains fail-closed for this otherwise accepted key.
        with self.assertRaises(SnapshotError):
            capture_usage_snapshot(api_key=SECRET, opener=original)

    def test_fetches_both_endpoints_but_emits_only_whitelisted_numbers(self):
        opener = RecordingOpener()
        snapshot = snapshot_from_environment(
            api_key_env=ENV_LABEL,
            environ={ENV_LABEL: SECRET},
            opener=opener,
            now=lambda: datetime(2026, 7, 18, 1, 2, 3, tzinfo=timezone.utc),
        )
        self.assertEqual(snapshot, {
            "schema_version": 1,
            "captured_at_utc": "2026-07-18T01:02:03Z",
            "key": {
                "http_status": 200,
                "usage": 1.25,
                "limit": 3.0,
                "limit_remaining": 1.75,
                "key_fingerprint_sha256": hashlib.sha256(
                    SECRET.encode()
                ).hexdigest(),
                "limit_reset": None,
                "include_byok_in_limit": True,
                "is_management_key": False,
                "is_provisioning_key": False,
                "is_free_tier": False,
                "expires_at_utc": None,
            },
            "account": {
                "http_status": 200,
                "total_usage": 4.5,
                "total_credits": 10.0,
                "remaining_credits": 5.5,
            },
        })
        serialized = json.dumps(snapshot)
        self.assertNotIn(SECRET, serialized)
        self.assertNotIn(RAW_LABEL, serialized)
        self.assertNotIn(ENV_LABEL, serialized)
        self.assertNotIn(str(len(SECRET)), serialized)
        self.assertEqual(
            snapshot["key"]["key_fingerprint_sha256"],
            hashlib.sha256(SECRET.encode()).hexdigest(),
        )
        for request, timeout in opener.requests:
            self.assertEqual(request.get_header("Authorization"), f"Bearer {SECRET}")
            self.assertEqual(timeout, 15.0)

    def test_atomic_file_is_also_secret_and_label_free(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = snapshot_from_environment(
                api_key_env=ENV_LABEL,
                environ={ENV_LABEL: SECRET},
                opener=RecordingOpener(),
            )
            output = root / "nested" / "snapshot.json"
            write_json_atomic(output, snapshot)
            serialized = output.read_text(encoding="utf-8")
            self.assertNotIn(SECRET, serialized)
            self.assertNotIn(RAW_LABEL, serialized)
            self.assertNotIn(ENV_LABEL, serialized)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            with self.assertRaisesRegex(SnapshotError, "already exists"):
                write_json_atomic(output, {"changed": True}, overwrite=False)
            self.assertEqual(output.read_text(encoding="utf-8"), serialized)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_missing_environment_key_error_omits_label(self):
        with self.assertRaises(SnapshotError) as raised:
            snapshot_from_environment(api_key_env=ENV_LABEL, environ={})
        self.assertNotIn(ENV_LABEL, str(raised.exception))
        self.assertNotIn(SECRET, str(raised.exception))

    def test_http_error_does_not_read_or_emit_response_body(self):
        secret_body = TrackingBody(
            f"provider says key={SECRET} label={RAW_LABEL}".encode()
        )

        def failing_opener(request, *, timeout):
            raise urllib.error.HTTPError(
                request.full_url, 429, "secret-ish reason", {}, secret_body
            )

        with self.assertRaises(SnapshotError) as raised:
            capture_usage_snapshot(api_key=SECRET, opener=failing_opener)
        rendered = str(raised.exception)
        self.assertEqual(rendered, "OpenRouter metadata request returned HTTP 429")
        self.assertNotIn(SECRET, rendered)
        self.assertNotIn(RAW_LABEL, rendered)
        self.assertFalse(secret_body.read_called)
        self.assertTrue(secret_body.close_called)

    def test_normal_key_may_record_credits_forbidden_without_body(self):
        secret_body = TrackingBody(f"account detail={SECRET}".encode())

        class NormalKeyOpener(RecordingOpener):
            def __call__(self, request, *, timeout):
                if request.full_url.endswith("/api/v1/credits"):
                    raise urllib.error.HTTPError(
                        request.full_url, 403, "forbidden", {}, secret_body
                    )
                return super().__call__(request, timeout=timeout)

        snapshot = capture_usage_snapshot(
            api_key=SECRET,
            opener=NormalKeyOpener(),
        )
        self.assertEqual(snapshot["account"], {
            "http_status": 403,
            "total_usage": None,
            "total_credits": None,
            "remaining_credits": None,
        })
        self.assertEqual(snapshot["key"]["limit"], 3.0)
        self.assertFalse(secret_body.read_called)
        self.assertTrue(secret_body.close_called)

    def test_rejects_non_numeric_whitelisted_field_without_copying_payload(self):
        class BadOpener(RecordingOpener):
            def __call__(self, request, *, timeout):
                if request.full_url.endswith("/api/v1/key"):
                    return FakeResponse(
                        {"data": {"usage": SECRET}}, url=request.full_url
                    )
                return super().__call__(request, timeout=timeout)

        with self.assertRaises(SnapshotError) as raised:
            capture_usage_snapshot(api_key=SECRET, opener=BadOpener())
        self.assertNotIn(SECRET, str(raised.exception))

    def test_rejects_reset_management_provisioning_or_free_tier_keys(self):
        mutations = (
            {"limit_reset": "daily"},
            {"is_management_key": True},
            {"is_provisioning_key": True},
            {"is_free_tier": True},
            {"include_byok_in_limit": False},
            {"expires_at": "2000-01-01T00:00:00Z"},
        )

        for mutation in mutations:
            class BadKeyOpener(RecordingOpener):
                def __call__(self, request, *, timeout):
                    if request.full_url.endswith("/api/v1/key"):
                        data = {
                            "label": RAW_LABEL,
                            "usage": 0.0,
                            "limit": 3.0,
                            "limit_remaining": 3.0,
                            "limit_reset": None,
                            "is_management_key": False,
                            "is_provisioning_key": False,
                            "is_free_tier": False,
                            "include_byok_in_limit": True,
                            "expires_at": None,
                        }
                        data.update(mutation)
                        return FakeResponse({"data": data}, url=request.full_url)
                    return super().__call__(request, timeout=timeout)

            with self.subTest(mutation=mutation):
                with self.assertRaises(SnapshotError):
                    capture_usage_snapshot(
                        api_key=SECRET,
                        opener=BadKeyOpener(),
                    )

    def test_rejects_missing_expiration_field(self):
        class MissingExpiryOpener(RecordingOpener):
            def __call__(self, request, *, timeout):
                if request.full_url.endswith("/api/v1/key"):
                    data = {
                        "usage": 0.0,
                        "limit": 3.0,
                        "limit_remaining": 3.0,
                        "limit_reset": None,
                        "is_management_key": False,
                        "is_provisioning_key": False,
                        "is_free_tier": False,
                        "include_byok_in_limit": True,
                    }
                    return FakeResponse({"data": data}, url=request.full_url)
                return super().__call__(request, timeout=timeout)

        with self.assertRaisesRegex(SnapshotError, "expires_at is missing"):
            capture_usage_snapshot(api_key=SECRET, opener=MissingExpiryOpener())

    def test_rejects_redirected_response_before_reading_body(self):
        body = TrackingBody(f"redirected secret={SECRET}".encode())

        class RedirectedResponse:
            status = 200

            def getcode(self):
                return self.status

            def geturl(self):
                return "https://attacker.invalid/capture"

            def read(self, *args, **kwargs):
                return body.read(*args, **kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

        with self.assertRaisesRegex(SnapshotError, "redirected"):
            capture_usage_snapshot(
                api_key=SECRET,
                opener=lambda request, *, timeout: RedirectedResponse(),
            )
        self.assertFalse(body.read_called)


if __name__ == "__main__":
    unittest.main()
