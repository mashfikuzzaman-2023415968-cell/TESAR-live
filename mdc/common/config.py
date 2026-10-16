"""Configuration loading and validation (spec Section 5, security S.12 row 0).

The configuration is a tree of dataclasses that mirrors ``configs/default.yaml``
exactly. Loading works in three layers, later layers winning:

1. the dataclass defaults below (kept identical to ``configs/default.yaml`` — a
   unit test compares the two so they can never drift apart),
2. an optional user YAML file, deep-merged over the defaults,
3. command-line overrides of the form ``--set scheduler.alpha=0.7``.

Why so strict? A typo such as ``sheduler.alpha`` would otherwise be silently
ignored and an experiment would run with the wrong parameters. Likewise a value
of the wrong type (``tls: "no"``) must never be "interpreted". So:

* unknown keys are an error,
* wrong types are an error (``bool`` is never accepted where a number is
  expected, even though ``bool`` is a subclass of ``int`` in Python),
* out-of-range values are an error,
* insecure combinations are refused (fail closed, spec rule 0.1.12).

This module is used by the coordinator only. It imports PyYAML, which is not
installed on phones; the worker never imports this module (see DECISIONS.md and
``scripts/check_worker_isolation.py``).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mdc.common.redact import redact

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
POLICIES = frozenset({"local", "round_robin", "fastest", "most_reliable", "tesar"})
JOB_TYPES = frozenset({"image", "synthetic"})
COMPRESSIONS = frozenset({"zlib", "none"})
BEACON_MODES = frozenset({"broadcast", "multicast"})
LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

# Range limits used by the semantic validation below.
MAX_PORT = 65535
MIN_HEADER_LIMIT = 1024  # a header limit below this could not carry a HELLO
MAX_HEADER_LIMIT = 1024 * 1024
MAX_ZLIB_LEVEL = 9
MAX_CERT_DAYS = 3650
MIN_BASE_TILE = 8
MAX_PIXEL_VALUE = 255
MAX_SLOTS = 8  # spec S.6.8: self-reported slots are clamped to <= 8
MAX_PREFETCH = 8
MAX_LOCAL_SLOTS = 64
MAX_REPLICAS = 8
PERCENT = 100.0


class ConfigError(ValueError):
    """Raised for any invalid configuration: unknown key, bad type, bad range,
    or an insecure combination of settings."""


# --------------------------------------------------------------------------
# Dataclasses (defaults identical to configs/default.yaml)
# --------------------------------------------------------------------------


@dataclass
class NetworkConfig:
    """Sockets, discovery and wire-level size limits."""

    # Binding all interfaces is by design: LAN workers must reach the coordinator
    # (DECISIONS.md, "bandit B104 suppression").
    coordinator_bind: str = "0.0.0.0"  # noqa: S104  # nosec B104
    tcp_port: int = 50010
    discovery_port: int = 50000
    multicast_group: str = "239.255.77.77"
    beacon_interval_s: float = 1.0
    beacon_modes: list[str] = field(default_factory=lambda: ["broadcast", "multicast"])
    advertise_host: str | None = None
    max_header_bytes: int = 16384
    max_payload_bytes: int = 33554432
    max_raw_array_bytes: int = 67108864
    max_connections: int = 64
    max_connections_per_ip: int = 4
    handshake_timeout_s: float = 5.0
    idle_read_timeout_s: float = 30.0
    compression: str = "zlib"
    zlib_level: int = 1


@dataclass
class MaxTaskParams:
    """Hard limits on task parameters, enforced on both coordinator and worker."""

    blur_radius: int = 5
    blur_iterations: int = 64
    chain_len: int = 5000000
    max_tile_pixels: int = 4194304


@dataclass
class SecurityConfig:
    """TLS, state directory, auth rate limiting and lab-mode switches."""

    tls: bool = True
    state_dir: str = "~/.mdc"
    cert_days: int = 825
    auth_fail_limit: int = 5
    auth_fail_window_s: float = 60.0
    auth_ban_s: float = 300.0
    lab_mode: bool = False
    max_task_params: MaxTaskParams = field(default_factory=MaxTaskParams)
    security_log: str = "security.jsonl"


@dataclass
class LivenessConfig:
    """Ping/pong failure detection and per-task timeouts."""

    ping_interval_s: float = 1.0
    suspect_after_s: float = 3.0
    dead_after_s: float = 6.0
    task_timeout_factor: float = 3.0
    task_timeout_slack_s: float = 2.0
    task_timeout_min_s: float = 3.0
    max_attempts_per_task: int = 6


@dataclass
class JobConfig:
    """What job to run and its compute-intensity knobs."""

    type: str = "image"
    image_path: str = "data/images/test_2048.png"
    base_tile: int = 128
    blur_radius: int = 2
    blur_iterations: int = 8
    edge_threshold: int = 64
    synthetic_tasks: int = 64
    synthetic_chain_len: int = 200000
    output_dir: str = "runs/latest"


@dataclass
class WorkersConfig:
    """Coordinator-side view of worker capacity and the trusted local worker."""

    default_slots: int = 1
    prefetch: int = 1
    min_battery_pct: float = 20.0
    local_worker: bool = True
    local_slots: int = 1


@dataclass
class NetestConfig:
    """RTT/bandwidth estimation parameters (Phase 6)."""

    rtt_ewma_alpha: float = 0.3
    bw_ewma_alpha: float = 0.3
    probe_bytes: int = 262144
    default_bw_bytes_per_s: float = 1000000.0
    min_bw_bytes_per_s: float = 10000.0


@dataclass
class ChunkingConfig:
    """Adaptive chunking parameters (Phase 6)."""

    adaptive: bool = True
    target_task_s: float = 2.0
    max_chunk_tiles: int = 8
    min_chunk_tiles: int = 1
    loss_tolerance: float = 0.1
    fixed_chunk_tiles: int = 1


@dataclass
class ReputationConfig:
    """Beta reputation and failure-rate parameters (Phase 7)."""

    prior_success: float = 1.0
    prior_failure: float = 1.0
    decay: float = 0.98
    weight_timeout: float = 1.0
    weight_disconnect: float = 1.0
    weight_error: float = 1.0
    weight_wrong_result: float = 5.0
    failure_rate_ewma_alpha: float = 0.2
    default_failure_rate_per_s: float = 0.001


@dataclass
class EnergyConfig:
    """Energy model constants (Phase 8). Rough placeholders until calibrated."""

    enabled: bool = True
    p_compute_w_phone: float = 2.0
    p_compute_w_laptop: float = 15.0
    p_tx_w_phone: float = 1.2
    p_tx_w_laptop: float = 2.0
    battery_capacity_wh_phone: float = 15.0
    battery_capacity_wh_laptop: float = 50.0
    max_job_energy_pct: float = 5.0
    nominal_voltage_v: float = 3.85
    sample_interval_s: float = 2.0


@dataclass
class SchedulerConfig:
    """Scheduling policy and the TESAR speed/reliability tradeoff knob."""

    policy: str = "tesar"
    alpha: float = 0.5
    kappa: float = 10.0
    replication: bool = True
    reliability_target: float = 0.90
    max_replicas: int = 2
    allow_local: bool = True


@dataclass
class VerificationConfig:
    """Spot-check verification parameters (Phase 10)."""

    enabled: bool = True
    base_prob: float = 0.05
    low_rep_boost: float = 0.5
    probation_tasks: int = 3
    strikes_to_quarantine: int = 2
    prefer_local_verifier: bool = True


@dataclass
class DashboardConfig:
    """Web dashboard settings (Phase 11)."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8080
    demo_controls: bool = False
    session_ttl_s: float = 28800.0
    login_rate_limit_per_min: int = 10


@dataclass
class LoggingConfig:
    """Console/file logging and the structured event log file name."""

    level: str = "INFO"
    events_file: str = "events.jsonl"


@dataclass
class ExperimentConfig:
    """Experiment-wide settings."""

    seed: int = 42


@dataclass
class Config:
    """Root configuration object. Field names match the YAML sections."""

    network: NetworkConfig = field(default_factory=NetworkConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    liveness: LivenessConfig = field(default_factory=LivenessConfig)
    job: JobConfig = field(default_factory=JobConfig)
    workers: WorkersConfig = field(default_factory=WorkersConfig)
    netest: NetestConfig = field(default_factory=NetestConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    reputation: ReputationConfig = field(default_factory=ReputationConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain nested dict (suitable for YAML/JSON dumping)."""
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        """Short, stable hash of the effective configuration.

        Logged in ``coordinator_started`` events so that every result can be
        traced back to the exact configuration that produced it.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Generic dict -> dataclass conversion with strict type checking
# --------------------------------------------------------------------------


def _type_error(path: str, expected: str, value: object) -> ConfigError:
    return ConfigError(f"{path}: expected {expected}, got {type(value).__name__} {value!r}")


def _convert_scalar(path: str, tp: Any, value: object) -> object:
    """Check (and, for int->float, widen) a single scalar against its type."""
    # bool must be tested first: isinstance(True, int) is True in Python.
    if tp is bool:
        if not isinstance(value, bool):
            raise _type_error(path, "bool", value)
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _type_error(path, "int", value)
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _type_error(path, "number", value)
        as_float = float(value)
        if not math.isfinite(as_float):
            raise ConfigError(f"{path}: must be a finite number, got {value!r}")
        return as_float
    if tp is str:
        if not isinstance(value, str):
            raise _type_error(path, "string", value)
        return value
    raise ConfigError(f"{path}: unsupported config type {tp!r}")  # pragma: no cover


def _convert_value(path: str, tp: Any, value: object) -> object:
    """Convert ``value`` to the annotated type ``tp`` or raise ConfigError."""
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        if not isinstance(value, dict):
            raise _type_error(path, "mapping", value)
        return _build_dataclass(tp, value, path)
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is list:
        if not isinstance(value, list):
            raise _type_error(path, "list", value)
        return [_convert_value(f"{path}[{i}]", args[0], item) for i, item in enumerate(value)]
    if args and type(None) in args:
        # Optional[X] written as ``X | None``.
        if value is None:
            return None
        (inner,) = [a for a in args if a is not type(None)]
        return _convert_value(path, inner, value)
    return _convert_scalar(path, tp, value)


def _build_dataclass(cls: type[Any], data: dict[str, Any], path: str) -> Any:
    """Build dataclass ``cls`` from ``data``, rejecting unknown keys."""
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        where = path or "<root>"
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(known)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        sub_path = f"{path}.{name}" if path else name
        kwargs[name] = _convert_value(sub_path, hints[name], value)
    return cls(**kwargs)


# --------------------------------------------------------------------------
# Merging and overrides
# --------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict: ``overlay`` merged recursively over ``base``.

    Nested mappings are merged key by key; any other value (including lists)
    in ``overlay`` replaces the base value entirely. Inputs are not modified.
    """
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_override(text: str) -> tuple[list[str], object]:
    """Parse one ``section.key=value`` override.

    The value is parsed with ``yaml.safe_load`` so ``0.7`` becomes a float,
    ``true`` a bool and ``[a, b]`` a list. ``safe_load`` (never ``load``) is
    used because override text comes from the command line.
    """
    if "=" not in text:
        raise ConfigError(f"override {text!r} must look like section.key=value")
    key, raw_value = text.split("=", 1)
    parts = key.strip().split(".")
    if any(not p for p in parts):
        raise ConfigError(f"override {text!r} has an empty key component")
    try:
        value = yaml.safe_load(raw_value) if raw_value.strip() else ""
    except yaml.YAMLError as exc:
        raise ConfigError(f"override {text!r}: value is not valid YAML: {exc}") from exc
    return parts, value


def apply_override(data: dict[str, Any], parts: list[str], value: object) -> None:
    """Set ``data[parts[0]][parts[1]]... = value`` in place.

    Intermediate sections must already exist (they always do, because
    overrides are applied on top of the full defaults); this is what turns a
    typo such as ``sheduler.alpha`` into an error instead of a silent no-op.
    """
    node: Any = data
    for i, part in enumerate(parts[:-1]):
        if not isinstance(node, dict) or part not in node or not isinstance(node[part], dict):
            prefix = ".".join(parts[: i + 1])
            raise ConfigError(f"override: unknown config section {prefix!r}")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise ConfigError(f"override: unknown config key {'.'.join(parts)!r}")
    node[parts[-1]] = value


# --------------------------------------------------------------------------
# Semantic validation (ranges and insecure combinations)
# --------------------------------------------------------------------------


def _check(cond: bool, message: str) -> None:
    """Raise ConfigError(message) unless ``cond``. (Never ``assert``: asserts
    are stripped under ``python -O``.)"""
    if not cond:
        raise ConfigError(message)


def _check_ranges(cfg: Config) -> None:  # noqa: PLR0915 - a flat list of checks reads best
    """Reject values outside their meaningful range."""
    n, s, lv = cfg.network, cfg.security, cfg.liveness
    _check(0 <= n.tcp_port <= MAX_PORT, "network.tcp_port must be in 0..65535 (0 = ephemeral)")
    _check(0 <= n.discovery_port <= MAX_PORT, "network.discovery_port must be in 0..65535")
    _check(n.beacon_interval_s > 0, "network.beacon_interval_s must be > 0")
    _check(
        set(n.beacon_modes) <= BEACON_MODES,
        f"network.beacon_modes must be a subset of {sorted(BEACON_MODES)}",
    )
    _check(
        MIN_HEADER_LIMIT <= n.max_header_bytes <= MAX_HEADER_LIMIT,
        "network.max_header_bytes out of range",
    )
    _check(n.max_payload_bytes >= 1, "network.max_payload_bytes must be >= 1")
    _check(n.max_raw_array_bytes >= 1, "network.max_raw_array_bytes must be >= 1")
    _check(n.max_connections >= 1, "network.max_connections must be >= 1")
    _check(n.max_connections_per_ip >= 1, "network.max_connections_per_ip must be >= 1")
    _check(n.handshake_timeout_s > 0, "network.handshake_timeout_s must be > 0")
    _check(n.idle_read_timeout_s > 0, "network.idle_read_timeout_s must be > 0")
    _check(n.compression in COMPRESSIONS, f"network.compression must be one of {COMPRESSIONS}")
    _check(0 <= n.zlib_level <= MAX_ZLIB_LEVEL, "network.zlib_level must be in 0..9")

    _check(1 <= s.cert_days <= MAX_CERT_DAYS, "security.cert_days must be in 1..3650")
    _check(s.auth_fail_limit >= 1, "security.auth_fail_limit must be >= 1")
    _check(s.auth_fail_window_s > 0, "security.auth_fail_window_s must be > 0")
    _check(s.auth_ban_s >= 0, "security.auth_ban_s must be >= 0")
    _check(bool(s.state_dir.strip()), "security.state_dir must not be empty")
    mtp = s.max_task_params
    _check(mtp.blur_radius >= 1, "security.max_task_params.blur_radius must be >= 1")
    _check(mtp.blur_iterations >= 1, "security.max_task_params.blur_iterations must be >= 1")
    _check(mtp.chain_len >= 1, "security.max_task_params.chain_len must be >= 1")
    _check(mtp.max_tile_pixels >= 1, "security.max_task_params.max_tile_pixels must be >= 1")

    _check(lv.ping_interval_s > 0, "liveness.ping_interval_s must be > 0")
    _check(
        0 < lv.suspect_after_s < lv.dead_after_s,
        "liveness: need 0 < suspect_after_s < dead_after_s",
    )
    _check(lv.task_timeout_factor >= 1, "liveness.task_timeout_factor must be >= 1")
    _check(lv.task_timeout_slack_s >= 0, "liveness.task_timeout_slack_s must be >= 0")
    _check(lv.task_timeout_min_s > 0, "liveness.task_timeout_min_s must be > 0")
    _check(lv.max_attempts_per_task >= 1, "liveness.max_attempts_per_task must be >= 1")

    j = cfg.job
    _check(j.type in JOB_TYPES, f"job.type must be one of {sorted(JOB_TYPES)}")
    _check(j.base_tile >= MIN_BASE_TILE, "job.base_tile must be >= 8")
    _check(
        1 <= j.blur_radius <= mtp.blur_radius,
        "job.blur_radius must be in 1..security.max_task_params.blur_radius",
    )
    _check(
        1 <= j.blur_iterations <= mtp.blur_iterations,
        "job.blur_iterations must be in 1..security.max_task_params.blur_iterations",
    )
    _check(0 <= j.edge_threshold <= MAX_PIXEL_VALUE, "job.edge_threshold must be in 0..255")
    _check(j.synthetic_tasks >= 1, "job.synthetic_tasks must be >= 1")
    _check(
        1 <= j.synthetic_chain_len <= mtp.chain_len,
        "job.synthetic_chain_len must be in 1..security.max_task_params.chain_len",
    )

    w = cfg.workers
    _check(1 <= w.default_slots <= MAX_SLOTS, "workers.default_slots must be in 1..8")
    _check(0 <= w.prefetch <= MAX_PREFETCH, "workers.prefetch must be in 0..8")
    _check(0 <= w.min_battery_pct <= PERCENT, "workers.min_battery_pct must be in 0..100")
    _check(1 <= w.local_slots <= MAX_LOCAL_SLOTS, "workers.local_slots must be in 1..64")

    ne = cfg.netest
    _check(0 < ne.rtt_ewma_alpha <= 1, "netest.rtt_ewma_alpha must be in (0, 1]")
    _check(0 < ne.bw_ewma_alpha <= 1, "netest.bw_ewma_alpha must be in (0, 1]")
    _check(ne.probe_bytes >= 1, "netest.probe_bytes must be >= 1")
    _check(ne.min_bw_bytes_per_s > 0, "netest.min_bw_bytes_per_s must be > 0")
    _check(
        ne.default_bw_bytes_per_s >= ne.min_bw_bytes_per_s,
        "netest.default_bw_bytes_per_s must be >= min_bw_bytes_per_s",
    )

    c = cfg.chunking
    _check(c.target_task_s > 0, "chunking.target_task_s must be > 0")
    _check(
        1 <= c.min_chunk_tiles <= c.max_chunk_tiles,
        "chunking: need 1 <= min_chunk_tiles <= max_chunk_tiles",
    )
    _check(0 < c.loss_tolerance < 1, "chunking.loss_tolerance must be in (0, 1)")
    _check(c.fixed_chunk_tiles >= 1, "chunking.fixed_chunk_tiles must be >= 1")

    r = cfg.reputation
    _check(r.prior_success > 0 and r.prior_failure > 0, "reputation priors must be > 0")
    _check(0 < r.decay <= 1, "reputation.decay must be in (0, 1]")
    for name in ("weight_timeout", "weight_disconnect", "weight_error", "weight_wrong_result"):
        _check(getattr(r, name) >= 0, f"reputation.{name} must be >= 0")
    _check(0 < r.failure_rate_ewma_alpha <= 1, "reputation.failure_rate_ewma_alpha in (0, 1]")
    _check(r.default_failure_rate_per_s >= 0, "reputation.default_failure_rate_per_s >= 0")

    e = cfg.energy
    for name in (
        "p_compute_w_phone",
        "p_compute_w_laptop",
        "p_tx_w_phone",
        "p_tx_w_laptop",
        "battery_capacity_wh_phone",
        "battery_capacity_wh_laptop",
        "nominal_voltage_v",
        "sample_interval_s",
    ):
        _check(getattr(e, name) > 0, f"energy.{name} must be > 0")
    _check(0 < e.max_job_energy_pct <= PERCENT, "energy.max_job_energy_pct must be in (0, 100]")

    sc = cfg.scheduler
    _check(sc.policy in POLICIES, f"scheduler.policy must be one of {sorted(POLICIES)}")
    _check(0 <= sc.alpha <= 1, "scheduler.alpha must be in [0, 1]")
    _check(sc.kappa >= 0, "scheduler.kappa must be >= 0")
    _check(0 < sc.reliability_target <= 1, "scheduler.reliability_target must be in (0, 1]")
    _check(1 <= sc.max_replicas <= MAX_REPLICAS, "scheduler.max_replicas must be in 1..8")

    v = cfg.verification
    _check(0 <= v.base_prob <= 1, "verification.base_prob must be in [0, 1]")
    _check(v.low_rep_boost >= 0, "verification.low_rep_boost must be >= 0")
    _check(v.probation_tasks >= 0, "verification.probation_tasks must be >= 0")
    _check(v.strikes_to_quarantine >= 1, "verification.strikes_to_quarantine must be >= 1")

    d = cfg.dashboard
    _check(0 <= d.port <= MAX_PORT, "dashboard.port must be in 0..65535")
    _check(d.session_ttl_s > 0, "dashboard.session_ttl_s must be > 0")
    _check(d.login_rate_limit_per_min >= 1, "dashboard.login_rate_limit_per_min must be >= 1")

    _check(cfg.logging.level in LOG_LEVELS, f"logging.level must be one of {sorted(LOG_LEVELS)}")
    _check(bool(cfg.logging.events_file.strip()), "logging.events_file must not be empty")


def check_security(cfg: Config, insecure_dev: bool = False) -> None:
    """Refuse insecure combinations of settings (spec S.3.1, S.9, S.12 row 0).

    * ``security.tls: false`` is only allowed with the explicit
      ``--insecure-dev`` flag, and even then only when the coordinator binds
      to loopback (so plaintext traffic never leaves the machine).
    * ``dashboard.demo_controls`` lets a browser crash or corrupt workers; it
      therefore additionally requires ``security.lab_mode``.
    """
    if not cfg.security.tls:
        if not insecure_dev:
            raise ConfigError(
                "security.tls is false but --insecure-dev was not given; refusing to start "
                "without TLS"
            )
        if cfg.network.coordinator_bind not in LOOPBACK_HOSTS:
            raise ConfigError(
                "--insecure-dev (no TLS) is only allowed when network.coordinator_bind is "
                f"127.0.0.1, not {cfg.network.coordinator_bind!r}"
            )
    if cfg.dashboard.demo_controls and not cfg.security.lab_mode:
        raise ConfigError("dashboard.demo_controls requires security.lab_mode: true")


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------


def default_dict() -> dict[str, Any]:
    """The built-in defaults as a plain dict."""
    return Config().to_dict()


def config_from_dict(data: dict[str, Any], insecure_dev: bool = False) -> Config:
    """Validate a full or partial config dict (merged over defaults)."""
    if not isinstance(data, dict):
        raise _type_error("<root>", "mapping", data)
    merged = deep_merge(default_dict(), data)
    cfg = _build_dataclass(Config, merged, "")
    if not isinstance(cfg, Config):  # pragma: no cover - _build_dataclass(Config) returns Config
        raise ConfigError("internal error building Config")
    _check_ranges(cfg)
    check_security(cfg, insecure_dev=insecure_dev)
    return cfg


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
    insecure_dev: bool = False,
) -> Config:
    """Load, merge, override and validate the configuration.

    Args:
        path: optional YAML file; its values are deep-merged over defaults.
        overrides: ``section.key=value`` strings, applied last.
        insecure_dev: the explicit ``--insecure-dev`` flag (allows tls=false
            on loopback only).

    Raises:
        ConfigError: for anything invalid. Never returns a partially valid config.
    """
    data = default_dict()
    if path is not None:
        file_path = Path(path)
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read config file {file_path}: {exc}") from exc
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"config file {file_path} is not valid YAML: {exc}") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise _type_error(str(file_path), "mapping at top level", loaded)
        # Validate the user file's keys before merging, so an unknown key in
        # the file is reported with the file's own structure.
        _build_dataclass(Config, deep_merge(default_dict(), loaded), "")
        data = deep_merge(data, loaded)
    for text_override in overrides or []:
        parts, value = parse_override(text_override)
        apply_override(data, parts, value)
    return config_from_dict(data, insecure_dev=insecure_dev)


def dump_yaml(cfg: Config) -> str:
    """Serialize a config to YAML text (used for ``config_effective.yaml``).

    Secrets are never stored in the config itself; as defense in depth the
    dict still passes through :func:`mdc.common.redact.redact`.
    """
    return yaml.safe_dump(redact(cfg.to_dict()), sort_keys=False)
