"""Tests for mdc.common.config: defaults, merging, overrides and strictness."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mdc.common.config import (
    Config,
    ConfigError,
    apply_override,
    config_from_dict,
    deep_merge,
    default_dict,
    dump_yaml,
    load_config,
    parse_override,
)


def test_defaults_match_default_yaml(repo_root: Path) -> None:
    """The dataclass defaults and configs/default.yaml must never drift apart."""
    from_file = load_config(repo_root / "configs" / "default.yaml")
    assert from_file == Config()
    raw = yaml.safe_load((repo_root / "configs" / "default.yaml").read_text())

    # Same sections and keys, recursively.
    def keys(d: dict, prefix: str = "") -> set[str]:
        out = set()
        for k, v in d.items():
            out.add(prefix + k)
            if isinstance(v, dict):
                out |= keys(v, prefix + k + ".")
        return out

    assert keys(raw) == keys(default_dict())


def test_load_without_file_gives_defaults() -> None:
    cfg = load_config()
    assert cfg.scheduler.alpha == 0.5
    assert cfg.network.tcp_port == 50010
    assert cfg.security.tls is True
    assert cfg.security.max_task_params.blur_radius == 5
    assert cfg.network.beacon_modes == ["broadcast", "multicast"]
    assert cfg.network.advertise_host is None


def test_user_yaml_is_deep_merged(tmp_path: Path) -> None:
    user = tmp_path / "user.yaml"
    user.write_text("scheduler:\n  alpha: 0.8\nsecurity:\n  max_task_params:\n    chain_len: 10\n")
    cfg = load_config(user, overrides=["job.synthetic_chain_len=5"])
    assert cfg.scheduler.alpha == 0.8
    assert cfg.scheduler.policy == "tesar"  # untouched sibling kept
    assert cfg.security.max_task_params.chain_len == 10
    assert cfg.security.max_task_params.blur_radius == 5  # untouched nested sibling


def test_overrides_parse_yaml_types() -> None:
    cfg = load_config(
        overrides=[
            "scheduler.alpha=0.7",
            "scheduler.replication=false",
            "scheduler.policy=fastest",
            "network.beacon_modes=[multicast]",
            "network.advertise_host=192.168.0.10",
            "workers.prefetch=0",
        ]
    )
    assert cfg.scheduler.alpha == 0.7
    assert cfg.scheduler.replication is False
    assert cfg.scheduler.policy == "fastest"
    assert cfg.network.beacon_modes == ["multicast"]
    assert cfg.network.advertise_host == "192.168.0.10"
    assert cfg.workers.prefetch == 0


def test_override_order_last_wins() -> None:
    cfg = load_config(overrides=["scheduler.alpha=0.1", "scheduler.alpha=0.9"])
    assert cfg.scheduler.alpha == 0.9


def test_int_accepted_for_float_and_widened() -> None:
    cfg = load_config(overrides=["scheduler.alpha=1"])
    assert cfg.scheduler.alpha == 1.0
    assert isinstance(cfg.scheduler.alpha, float)


def test_null_override_for_optional_field() -> None:
    cfg = load_config(overrides=["network.advertise_host=null"])
    assert cfg.network.advertise_host is None


@pytest.mark.parametrize(
    "override",
    [
        "sheduler.alpha=0.5",  # typo in section
        "scheduler.alpah=0.5",  # typo in key
        "scheduler.alpha.x=1",  # descending into a scalar
        "nosection=1",
    ],
)
def test_unknown_override_key_rejected(override: str) -> None:
    with pytest.raises(ConfigError, match="unknown"):
        load_config(overrides=[override])


@pytest.mark.parametrize("override", ["scheduler.alpha", "=1", "scheduler..alpha=1"])
def test_malformed_override_rejected(override: str) -> None:
    with pytest.raises(ConfigError):
        load_config(overrides=[override])


def test_override_with_invalid_yaml_value_rejected() -> None:
    with pytest.raises(ConfigError, match="not valid YAML"):
        parse_override("scheduler.policy=[unclosed")


def test_empty_override_value_is_empty_string() -> None:
    assert parse_override("logging.level=") == (["logging", "level"], "")
    with pytest.raises(ConfigError):  # empty string is not a valid level
        load_config(overrides=["logging.level="])


def test_unknown_key_in_file_rejected(tmp_path: Path) -> None:
    user = tmp_path / "user.yaml"
    user.write_text("scheduler:\n  alpah: 0.5\n")
    with pytest.raises(ConfigError, match=r"scheduler: unknown key\(s\) \['alpah'\]"):
        load_config(user)


def test_unknown_section_in_file_rejected(tmp_path: Path) -> None:
    user = tmp_path / "user.yaml"
    user.write_text("bogus:\n  x: 1\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(user)


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ("security.tls=no_thanks", "expected bool"),
        ("security.tls=1", "expected bool"),
        ("network.tcp_port=true", "expected int"),  # bool is not an int here
        ("network.tcp_port=1.5", "expected int"),
        ("network.tcp_port='50010'", "expected int"),
        ("scheduler.alpha=high", "expected number"),
        ("scheduler.alpha=false", "expected number"),
        ("scheduler.alpha=.nan", "finite"),
        ("scheduler.alpha=.inf", "finite"),
        ("scheduler.policy=3", "expected string"),
        ("network.beacon_modes=broadcast", "expected list"),
        ("network.beacon_modes=[1]", "expected string"),
        ("security.max_task_params=5", "expected mapping"),
        ("network.advertise_host=[a]", "expected string"),
    ],
)
def test_wrong_types_rejected(override: str, fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment):
        load_config(overrides=[override])


@pytest.mark.parametrize(
    "override",
    [
        "scheduler.alpha=1.01",
        "scheduler.alpha=-0.01",
        "scheduler.policy=greedy",
        "network.tcp_port=65536",
        "network.tcp_port=-1",
        "network.beacon_modes=[unicast]",
        "network.compression=gzip",
        "network.zlib_level=10",
        "liveness.suspect_after_s=6.0",  # must be < dead_after_s
        "job.type=video",
        "job.blur_radius=6",  # above security.max_task_params.blur_radius
        "job.blur_iterations=0",
        "job.edge_threshold=256",
        "workers.default_slots=9",
        "workers.min_battery_pct=101",
        "netest.rtt_ewma_alpha=0",
        "chunking.min_chunk_tiles=9",  # above max_chunk_tiles
        "chunking.loss_tolerance=1",
        "reputation.decay=0",
        "reputation.weight_wrong_result=-1",
        "energy.max_job_energy_pct=0",
        "energy.p_tx_w_phone=0",
        "verification.base_prob=1.5",
        "verification.strikes_to_quarantine=0",
        "dashboard.port=70000",
        "logging.level=VERBOSE",
        "security.cert_days=0",
        "security.state_dir=' '",
    ],
)
def test_out_of_range_rejected(override: str) -> None:
    with pytest.raises(ConfigError):
        load_config(overrides=[override])


@pytest.mark.parametrize(
    "override",
    ["scheduler.alpha=0", "scheduler.alpha=1", "network.tcp_port=0", "workers.prefetch=0"],
)
def test_boundary_values_accepted(override: str) -> None:
    load_config(overrides=[override])


def test_missing_file_is_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_file_is_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("scheduler: [unclosed\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(bad)


def test_non_mapping_file_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "list.yaml"
    bad.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(bad)


def test_empty_file_gives_defaults(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    assert load_config(empty) == Config()


def test_config_from_dict_rejects_non_mapping() -> None:
    with pytest.raises(ConfigError):
        config_from_dict([1, 2])  # type: ignore[arg-type]


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"b": 1, "c": [1]}, "x": 1}
    overlay = {"a": {"b": 2}, "y": {"z": 3}}
    merged = deep_merge(base, overlay)
    assert merged == {"a": {"b": 2, "c": [1]}, "x": 1, "y": {"z": 3}}
    assert base == {"a": {"b": 1, "c": [1]}, "x": 1}
    merged["a"]["c"].append(2)
    assert base["a"]["c"] == [1]


def test_deep_merge_replaces_lists_and_scalars_with_dicts() -> None:
    assert deep_merge({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}
    assert deep_merge({"a": 1}, {"a": {"b": 1}}) == {"a": {"b": 1}}


def test_apply_override_sets_value() -> None:
    data = default_dict()
    apply_override(data, ["scheduler", "kappa"], 3)
    assert data["scheduler"]["kappa"] == 3


def test_config_hash_stable_and_sensitive() -> None:
    a = load_config()
    b = load_config()
    c = load_config(overrides=["scheduler.alpha=0.6"])
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != c.config_hash()
    assert len(a.config_hash()) == 16


def test_dump_yaml_round_trips(tmp_path: Path) -> None:
    cfg = load_config(overrides=["scheduler.alpha=0.25"])
    path = tmp_path / "effective.yaml"
    path.write_text(dump_yaml(cfg))
    assert load_config(path) == cfg
