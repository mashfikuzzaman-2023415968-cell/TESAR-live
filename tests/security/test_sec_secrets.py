"""Secret hygiene: redaction, log filtering, security audit log (spec S.10)."""

from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path

import pytest
import yaml

from mdc.common import credentials as C
from mdc.common import redact as R
from mdc.common.config import dump_yaml, load_config
from mdc.common.logging_setup import setup_logging, teardown_logging
from mdc.common.redact import MASK, RedactingFilter, is_secret_key, redact, redact_text
from mdc.common.safe_names import sanitize_for_log, validate_id, validate_name
from mdc.coordinator.security_log import SECURITY_EVENTS, SecurityLog, read_security_log

pytestmark = pytest.mark.security

CODE = C.encode_join_code(
    C.JoinCode("10.0.0.1", 50010, "w_abc", "phone-a", bytes(range(32)), "sha256:" + "a" * 64)
)
PEM = "-----BEGIN " + "EC PRIVATE KEY-----\nMHcCAQEEIBx\n-----END EC PRIVATE KEY-----"


# ------------------------------------------------------------------ redact


@pytest.mark.parametrize(
    ("key", "secret"),
    [
        ("secret", True),
        ("device_secret", True),
        ("secret_b64", True),
        ("admin_token", True),
        ("token", True),
        ("auth", True),
        ("proof", True),
        ("password", True),
        ("private_key", True),
        ("api-key", True),
        ("Session.Token", True),
        ("auth_fail_limit", False),
        ("key_path", False),
        ("keyboard", False),
        ("author", False),
        ("name", False),
        ("tokens_left", False),
        (5, False),
    ],
)
def test_is_secret_key(key: object, secret: bool) -> None:
    assert is_secret_key(key) is secret


def test_redact_nested_structures() -> None:
    data = {
        "name": "phone-a",
        "secret_b64": "c2VjcmV0",
        "auth": "ab" * 32,
        "nested": [{"admin_token": "t0k"}, ("x", {"proof": "p"})],
        "note": f"code {CODE} inline",
        "raw": b"\x00\x01",
        "empty_secret": None,
        "count": 3,
    }
    out = redact(data)
    text = json.dumps(out, default=str)
    assert "c2VjcmV0" not in text and "t0k" not in text and CODE not in text
    assert out["name"] == "phone-a"
    assert out["secret_b64"] == MASK
    assert out["nested"][0]["admin_token"] == MASK
    assert out["nested"][1][1]["proof"] == MASK
    assert isinstance(out["nested"][1], tuple)
    assert out["note"] == "code MDC1.<redacted> inline"
    assert out["raw"] == MASK
    assert out["empty_secret"] is None
    assert out["count"] == 3
    assert data["secret_b64"] == "c2VjcmV0"  # noqa: S105 - input untouched


def test_redact_depth_limit() -> None:
    deep: dict = {}
    node = deep
    for _ in range(40):
        node["x"] = {}
        node = node["x"]
    assert MASK in json.dumps(redact(deep))


def test_redact_text_join_codes_and_keys() -> None:
    assert CODE not in redact_text(f"start {CODE}")
    assert "MHcCAQEE" not in redact_text(f"key:\n{PEM}\ntrailer")
    assert "trailer" in redact_text(f"key:\n{PEM}\ntrailer")
    truncated = "-----BEGIN " + "PRIVATE KEY-----\nAAAA"
    assert "AAAA" not in redact_text(truncated)
    assert redact_text("frames start with MDC1 magic") == "frames start with MDC1 magic"


def test_logging_never_writes_secrets(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    logfile = tmp_path / "log.txt"
    caplog.set_level(logging.DEBUG)
    try:
        setup_logging("DEBUG", logfile)
        log = logging.getLogger("mdc.test.secrets")
        log.info("worker joined with %s", CODE)
        log.warning("loaded %s", PEM)
        try:
            raise ValueError(f"bad code {CODE}")
        except ValueError:
            log.exception("failure")
        for handler in logging.getLogger().handlers:
            handler.flush()
    finally:
        teardown_logging()
    text = logfile.read_text()
    assert CODE not in text
    assert "MHcCAQEE" not in text
    assert text.count("MDC1.<redacted>") == 2  # message + exception line
    assert "Traceback" in text
    # Handlers we did not install (here: pytest's own capture) see redacted records too.
    assert CODE not in caplog.text
    assert "MHcCAQEE" not in caplog.text


def test_record_redaction_install_is_idempotent_and_reversible() -> None:
    before = logging.getLogRecordFactory()
    R.install_record_redaction()
    R.install_record_redaction()
    record = logging.getLogger("x").makeRecord("x", logging.INFO, "f", 1, "%s", (CODE,), None)
    assert CODE not in record.getMessage()
    R.uninstall_record_redaction()
    R.uninstall_record_redaction()
    assert logging.getLogRecordFactory() is before


def test_stack_info_redacted() -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "m", None, None, sinfo=CODE)
    RedactingFilter().filter(record)
    assert record.stack_info == "MDC1.<redacted>"


def test_filter_survives_broken_format_string() -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "%d %d", (CODE,), None)
    assert RedactingFilter().filter(record)
    assert CODE not in record.getMessage()


def test_config_dump_is_redacted() -> None:
    dumped = yaml.safe_load(dump_yaml(load_config()))
    assert dumped["security"]["auth_fail_limit"] == 5  # useful values remain readable
    assert dumped["security"]["state_dir"] == "~/.mdc"


# -------------------------------------------------------------- safe names


@pytest.mark.parametrize("name", ["phone-a", "Laptop_B.2", "x" * 32])
def test_valid_names(name: str) -> None:
    assert validate_name(name) == name


@pytest.mark.parametrize("name", ["", "x" * 33, "a b", "a/b", "<b>", "..", ".", "é", None, 3])
def test_invalid_names(name: object) -> None:
    with pytest.raises(ValueError):
        validate_name(name)


def test_validate_id() -> None:
    assert validate_id("att_1") == "att_1"
    with pytest.raises(ValueError, match="attempt_id"):
        validate_id("a b", "attempt_id")


def test_sanitize_for_log() -> None:
    hostile = "ok\nFAKE LOG LINE\r\x1b[31mred\u202eevil\x00"
    clean = sanitize_for_log(hostile)
    for bad in ("\n", "\r", "\x1b", "\u202e", "\x00"):
        assert bad not in clean
    assert clean.startswith("okFAKE LOG LINE")
    assert sanitize_for_log("x" * 300).endswith("...[truncated]")
    assert len(sanitize_for_log("x" * 300)) == 256 + len("...[truncated]")
    assert sanitize_for_log(b"\n") == "b'\\n'"


# ----------------------------------------------------------- security log


def test_security_log_writes_jsonl(tmp_state_dir: Path) -> None:
    sec_log = SecurityLog.for_state_dir(tmp_state_dir)
    sec_log.emit("auth_failed", peer="192.168.0.9", reason="bad_mac")
    sec_log.close()
    sec_log.close()  # idempotent
    path = tmp_state_dir / "logs" / "security.jsonl"
    (record,) = read_security_log(path)
    assert record["type"] == "auth_failed"
    assert record["peer"] == "192.168.0.9"
    assert "wall" in record


def test_security_log_redacts_and_sanitizes(tmp_path: Path) -> None:
    sec_log = SecurityLog(tmp_path / "sec.jsonl")
    sec_log.emit(
        "validation_failed",
        peer="1.2.3.4",
        detail=f"name\nINJECTED {CODE}",
        auth="ab" * 32,
        fields={"secret": "s", "bad\nkey": "v"},
        many=list(range(200)),
    )
    sec_log.close()
    raw = (tmp_path / "sec.jsonl").read_text()
    assert raw.count("\n") == 1  # one record, one line
    assert CODE not in raw
    assert "ab" * 32 not in raw
    (record,) = read_security_log(tmp_path / "sec.jsonl")
    assert "\n" not in record["detail"]
    assert record["auth"] == MASK
    assert record["fields"]["secret"] == MASK
    assert "badkey" in record["fields"]
    assert len(record["many"]) == 50


def test_security_log_rejects_unknown_types() -> None:
    sec_log = SecurityLog(None)
    with pytest.raises(ValueError):
        sec_log.emit("auth_faild")
    for event in SECURITY_EVENTS:
        sec_log.emit(event)
    assert len(sec_log.records) == len(SECURITY_EVENTS)


def test_security_log_rate_limits_floods() -> None:
    now = [0.0]
    sec_log = SecurityLog(None, max_per_second=5, clock=lambda: now[0])
    written = [sec_log.emit("connection_limit", peer="x") for _ in range(1000)]
    assert sum(r is not None for r in written) == 5
    now[0] = 1.5  # next window: the first record reports how many were dropped
    record = sec_log.emit("connection_limit", peer="x")
    assert record is not None
    assert record["suppressed_before"] == 995
    # Other event types have their own budget.
    assert sec_log.emit("auth_failed") is not None


def test_security_log_rotates(tmp_path: Path) -> None:
    sec_log = SecurityLog(tmp_path / "s.jsonl", max_per_second=10**6, max_bytes=2000, backups=2)
    for i in range(200):
        sec_log.emit("frame_rejected", peer=f"10.0.0.{i % 250}", reason="bad magic")
    sec_log.close()
    files = sorted(tmp_path.iterdir())
    assert [p.name for p in files] == ["s.jsonl", "s.jsonl.1", "s.jsonl.2"]
    # Every file, including those created by rotation, is owner-only.
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in files)


def test_log_files_are_private_even_with_permissive_umask(tmp_path: Path) -> None:
    old = os.umask(0o002)
    try:
        sec_log = SecurityLog(tmp_path / "sec.jsonl")
        sec_log.emit("auth_failed")
        sec_log.close()
        loose = tmp_path / "app.log"
        loose.write_text("")
        os.chmod(loose, 0o644)  # pre-existing loose file gets tightened
        setup_logging("INFO", loose)
        logging.getLogger("mdc.t").info("x")
    finally:
        teardown_logging()
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "sec.jsonl").stat().st_mode) == 0o600
    assert stat.S_IMODE(loose.stat().st_mode) == 0o600


def test_security_log_memory_bounded() -> None:
    sec_log = SecurityLog(None, max_per_second=10**6)
    for _ in range(1500):
        sec_log.emit("frame_rejected")
    assert len(sec_log.records) == 1000


@pytest.mark.parametrize("field", ["type", "wall", "suppressed_before"])
def test_security_log_fields_cannot_overwrite_the_event_type(field: str) -> None:
    """Regression: a peer-supplied value in a field named "type" overwrote the event type
    (audit-log forgery). Reserved names are now refused."""
    sec_log = SecurityLog(None)
    with pytest.raises(ValueError, match="may not be named"):
        sec_log.emit("validation_failed", **{field: "ip_banned"})
    assert sec_log.records == []
