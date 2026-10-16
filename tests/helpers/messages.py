"""Valid sample messages of every type, shared by protocol/validation/fuzz tests."""

from __future__ import annotations

import numpy as np

from mdc.common import protocol as P

BATTERY = {
    "pct": 87.0,
    "charging": False,
    "current_ma": -412.0,
    "voltage_v": 3.85,
    "source": "mock",
    "ts_w": 1234.5,
}
HEX32 = "0123456789abcdef" * 2
HEX64 = HEX32 * 2
LIMITS = {"blur_radius": 5, "blur_iterations": 64, "chain_len": 5000000, "max_tile_pixels": 4194304}


def tile_array(seed: int = 1, shape: tuple[int, int] = (40, 50)) -> np.ndarray:
    """A deterministic uint8 tile."""
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


def sample_messages() -> dict[str, P.Message]:
    """One valid message per type (TASK_ASSIGN/RESULT with real array payloads)."""
    assign_fields, assign_payload = P.pack_array(tile_array(1))
    result_fields, result_payload = P.pack_array(tile_array(2, (30, 40)), "none")
    return {
        P.CHALLENGE: P.make_message(P.CHALLENGE, nonce=HEX32, server_version="0.1.0"),
        P.HELLO: P.make_message(
            P.HELLO,
            worker_id="w_0123456789abcdef",
            name="phone-a",
            platform="android",
            device_class="phone",
            cpu_count=8,
            slots=2,
            bench_ms_per_unit=12.5,
            battery=BATTERY,
            version="0.1.0",
            trusted_local=False,
            nonce_w=HEX32,
            auth=HEX64,
        ),
        P.WELCOME: P.make_message(
            P.WELCOME,
            session_id="s_1",
            ping_interval_s=1.0,
            accepted_slots=2,
            proof=HEX64,
            limits=LIMITS,
        ),
        P.REJECT: P.make_message(P.REJECT, reason="auth"),
        P.PING: P.make_message(P.PING, seq=7, t_c=100.25),
        P.PONG: P.make_message(
            P.PONG, seq=7, t_c=100.25, active_tasks=1, queued_tasks=0, battery=BATTERY
        ),
        P.PROBE: P.make_message(P.PROBE, payload=b"\x00" * 1000, probe_id="p_1", t_c=5.0),
        P.PROBE_ACK: P.make_message(P.PROBE_ACK, probe_id="p_1", t_c=5.0, received_bytes=1000),
        P.TASK_ASSIGN: P.make_message(
            P.TASK_ASSIGN,
            payload=assign_payload,
            task_id="s1_r00_c00",
            attempt_id="att_1",
            kind="img_tile",
            params={"r": 2, "iters": 3},
            **assign_fields,
        ),
        P.TASK_RESULT: P.make_message(
            P.TASK_RESULT,
            payload=result_payload,
            task_id="s1_r00_c00",
            attempt_id="att_1",
            status="ok",
            error=None,
            compute_ms=12.0,
            queue_ms=0.5,
            t_recv_w=10.0,
            t_send_w=10.02,
            **result_fields,
        ),
        P.TASK_CANCEL: P.make_message(P.TASK_CANCEL, attempt_id="att_1"),
        P.CONTROL: P.make_message(P.CONTROL, action="set_slowdown", value=3.0),
        P.BYE: P.make_message(P.BYE, reason="leaving"),
    }
