# Wire protocol (MDC1)

Implementation: [`mdc/common/protocol.py`](../mdc/common/protocol.py) (framing, arrays),
[`mdc/common/validation.py`](../mdc/common/validation.py) (schemas),
[`mdc/common/auth.py`](../mdc/common/auth.py) (handshake MACs),
[`mdc/common/tls.py`](../mdc/common/tls.py) (TLS + pinning).

All coordinator↔worker traffic is one long-lived **TLS** connection per worker, opened by the
worker (only the coordinator listens). Inside TLS, messages are exchanged as frames.

## 1. Frame format

```
+--------+----------------+-----------------+----------------------+-------------------+
| magic  | header_len     | payload_len     | header (UTF-8 JSON)  | payload (bytes)   |
| 4 bytes| uint32 big-end | uint32 big-end  | header_len bytes     | payload_len bytes |
| "MDC1" |                |                 |                      |                   |
+--------+----------------+-----------------+----------------------+-------------------+
          struct.pack("!4sII", b"MDC1", header_len, payload_len)
```

* TCP is a byte stream without message boundaries; the 12-byte prefix tells the reader exactly
  how many bytes belong to this message. `read_frame` uses `readexactly`, so a frame split over
  many TCP segments (even delivered byte by byte) is reassembled correctly.
* The prefix is checked **before** anything else is read: wrong magic,
  `header_len = 0`, `header_len > network.max_header_bytes` (16 KiB) or
  `payload_len > network.max_payload_bytes` (32 MiB) → `ProtocolError`, connection closed.
  A peer that claims a 4 GB payload costs the receiver 12 bytes, not 4 GB.
* Header: a compact, ASCII JSON **object** with at least `"type"` (`^[A-Z][A-Z0-9_]{0,31}$`) and
  `"v": 1` (exactly the integer 1, not `true` or `1.0`). Parsing is strict: invalid UTF-8,
  `NaN`/`Infinity`, numbers overflowing to infinity (`1e999`), integers longer than 20 digits,
  duplicate keys and nesting deeper than 8 are all rejected.
* Writes on one connection are serialized with an `asyncio.Lock` (`write_frame`), so two
  coroutines can never interleave their bytes.

### Arrays in payloads

Image tiles travel as raw bytes in the payload, described by these header fields:

| field | meaning |
|---|---|
| `shape` | list of 1–3 ints, each 1..16384 |
| `dtype` | `"uint8"`, `"int32"` or `"int64"` (always little-endian on the wire) |
| `comp` | `"zlib"` or `"none"` |
| `raw_len` | uncompressed size = `prod(shape) * itemsize`, ≤ `network.max_raw_array_bytes` (64 MiB) |
| `sha256` | hex SHA-256 of the **uncompressed** bytes |

Receiving order: validate fields → bounded decompression
(`decompressobj().decompress(data, raw_len + 1)`; reject if the output length ≠ `raw_len`,
if more output is pending, or if the stream is truncated or followed by extra data) →
SHA-256 check. A 64 KB "zip bomb" that would inflate to 64 MB is rejected after producing at
most `raw_len + 1` bytes.

## 2. Errors

| Exception | Meaning | Reaction |
|---|---|---|
| `ProtocolError` | malformed frame/header | close the connection, security-log `frame_rejected` |
| `ValidationError` (a `ProtocolError`) | well-formed but breaks a schema/limit | reject the message; close during the handshake; security-log `validation_failed` |
| `UnknownMessageType` (a `ValidationError`) | valid syntax, unknown type | after auth: log and ignore; more than 10 → disconnect |
| `IntegrityError` | payload hash mismatch (corruption in transit) | the attempt failed with a transmission error and is rescheduled; **not** evidence of dishonesty |
| `ConnectionClosed` | EOF exactly between frames | normal disconnect |

Error replies never echo internal details: `REJECT.reason` is one of a fixed set.

## 3. Messages

Direction: C = coordinator, W = worker. Every header also contains `"type"` and `"v": 1`.
Unknown fields are ignored (forward compatibility) and logged at DEBUG.

| Type | Dir | Header fields | Payload |
|---|---|---|---|
| `CHALLENGE` | C→W | `nonce` (32 hex), `server_version` | — |
| `HELLO` | W→C | `worker_id`, `name`, `platform`, `device_class`, `cpu_count`, `slots`, `bench_ms_per_unit`, `battery`, `version`, `trusted_local`, `nonce_w` (32 hex), `auth` (64 hex) | — |
| `WELCOME` | C→W | `session_id`, `ping_interval_s`, `accepted_slots`, `proof` (64 hex), `limits` | — |
| `REJECT` | C→W | `reason` ∈ {`auth`, `revoked`, `busy`, `protocol`, `version`} | — |
| `PING` | C→W | `seq`, `t_c` (coordinator monotonic) | — |
| `PONG` | W→C | `seq`, `t_c` (echoed), `active_tasks`, `queued_tasks`, `battery` | — |
| `PROBE` | C→W | `probe_id`, `t_c` | N random bytes |
| `PROBE_ACK` | W→C | `probe_id`, `t_c` (echoed), `received_bytes` | — |
| `TASK_ASSIGN` | C→W | `task_id`, `attempt_id`, `kind`, `params`, array fields | input array (none for `synthetic`) |
| `TASK_RESULT` | W→C | `task_id`, `attempt_id`, `status` (`ok`/`error`), `error`, `compute_ms`, `queue_ms`, `t_recv_w`, `t_send_w`, array fields **or** `json_result` | output array or empty |
| `TASK_CANCEL` | C→W | `attempt_id` | — |
| `CONTROL` | C→W | `action` ∈ {`crash`, `leave`, `set_malicious`, `set_slowdown`, `set_drop`}, `value` | — |
| `BYE` | either | `reason` | — |

Ids (`worker_id`, `session_id`, `task_id`, `attempt_id`, `probe_id`) match `^[A-Za-z0-9_-]{1,64}$`;
names match `^[A-Za-z0-9_.-]{1,32}$`; other strings are at most 256 characters.

### Example headers

```json
{"type":"CHALLENGE","v":1,"nonce":"9f0c3a...32 hex...","server_version":"0.1.0"}

{"type":"HELLO","v":1,"worker_id":"w_3f9a0c1d2e4b5a69","name":"phone-a","platform":"android",
 "device_class":"phone","cpu_count":8,"slots":1,"bench_ms_per_unit":41.7,
 "battery":{"pct":87.0,"charging":false,"current_ma":-412.0,"voltage_v":3.85,"source":"termux","ts_w":1234.5},
 "version":"0.1.0","trusted_local":false,"nonce_w":"1c2d...32 hex...","auth":"7e41...64 hex..."}

{"type":"WELCOME","v":1,"session_id":"s_a1b2c3d4e5f60718","ping_interval_s":1.0,"accepted_slots":1,
 "proof":"c09b...64 hex...","limits":{"blur_radius":5,"blur_iterations":64,"chain_len":5000000,"max_tile_pixels":4194304}}

{"type":"REJECT","v":1,"reason":"auth"}

{"type":"PING","v":1,"seq":42,"t_c":5123.875}
{"type":"PONG","v":1,"seq":42,"t_c":5123.875,"active_tasks":1,"queued_tasks":0,"battery":{...}}

{"type":"PROBE","v":1,"probe_id":"p_01","t_c":5124.0}            + 262144 random bytes
{"type":"PROBE_ACK","v":1,"probe_id":"p_01","t_c":5124.0,"received_bytes":262144}

{"type":"TASK_ASSIGN","v":1,"task_id":"s1_r03_c05","attempt_id":"att_9d8c7b6a5f4e3d2c",
 "kind":"img_tile","params":{"r":2,"iters":8},
 "shape":[162,290],"dtype":"uint8","comp":"zlib","raw_len":46980,"sha256":"5be1...64 hex..."}

{"type":"TASK_RESULT","v":1,"task_id":"s1_r03_c05","attempt_id":"att_9d8c7b6a5f4e3d2c",
 "status":"ok","error":null,"compute_ms":812.4,"queue_ms":0.3,"t_recv_w":991.2,"t_send_w":992.1,
 "shape":[128,256],"dtype":"uint8","comp":"zlib","raw_len":32768,"sha256":"a3f0...64 hex..."}

{"type":"TASK_RESULT","v":1,"task_id":"s2_r03","attempt_id":"att_...","status":"ok","error":null,
 "compute_ms":3.1,"queue_ms":0.0,"t_recv_w":1.0,"t_send_w":1.01,
 "json_result":{"hist":[...256 ints...],"edge_pixels":5120,"pixels":32768}}

{"type":"TASK_CANCEL","v":1,"attempt_id":"att_9d8c7b6a5f4e3d2c"}
{"type":"CONTROL","v":1,"action":"set_slowdown","value":3.0}
{"type":"BYE","v":1,"reason":"shutting down"}
```

### Battery object

```json
{"pct": 87.0, "charging": false, "current_ma": -412.0, "voltage_v": 3.85,
 "source": "termux" | "sysfs" | "mock", "ts_w": 1234.5}
```

`current_ma` is negative while discharging. Any field may be `null`. Self-reported values only
steer estimates and are clamped (`pct` to [0, 100], `slots` to ≤ 8, `bench_ms_per_unit` to
[0.01, 1e6]).

### Task kinds and parameters

Workers execute only these kinds — no code, pickles or module names ever cross the network:

| kind | `params` | input | output |
|---|---|---|---|
| `img_tile` | `r` (1..`blur_radius`), `iters` (1..`blur_iterations`) | 2-D uint8, each side > 2·halo, ≤ `max_tile_pixels` | uint8 array |
| `row_stats` | `edge_threshold` (0..255) | 2-D uint8 row strip | `{"hist": [256 ints summing to pixels], "edge_pixels", "pixels"}` |
| `synthetic` | `seed` (≥ 0), `chain_len` (1..`chain_len` limit) | none | `{"digest": 64 hex}` |

`halo = r * iters + 1`. Both sides enforce the limits; the worker uses its own copy.

## 4. Handshake (inside TLS)

```
 Worker                                                     Coordinator
   |---- TCP connect + TLS handshake ------------------------->|
   |  check SHA-256(server cert) == fp from join code          |
   |  (mismatch: close, log pin_mismatch, send nothing)        |
   |<--------------------------------- CHALLENGE{nonce_c} -----|  nonce_c = 16 random bytes
   |  nonce_w = 16 random bytes                                |
   |  auth = HMAC(secret, "MDC-W"|nonce_c|nonce_w|id|fp)       |
   |---- HELLO{worker_id, ..., nonce_w, auth} ---------------->|
   |                                   look up worker_id:      |
   |                                     unknown -> REJECT auth|
   |                                     revoked -> REJECT revoked
   |                                   verify auth (constant time)
   |                                     bad -> REJECT auth, count failure for the IP
   |                                   non-HELLO first -> REJECT protocol
   |      proof = HMAC(secret, "MDC-C"|nonce_w|nonce_c|id|fp)  |
   |<------------------------------ WELCOME{session_id, proof}-|
   |  verify proof (bad: close, log coordinator_proof_failed)  |
   |                                                           |
   |  ... PROBE / PING / TASK_* for the rest of the session ...|
```

`|` is the length-prefixed concatenation `uint32_be(len) || bytes` of each field. The whole
handshake must finish within `network.handshake_timeout_s` (5 s).

Why it is safe:

* **No replay:** each connection uses a fresh `nonce_c`, so a recorded HELLO is worthless.
* **No reflection:** the labels `MDC-W` and `MDC-C` differ, so a coordinator proof can never be
  accepted as a worker auth (or vice versa).
* **Channel binding:** both MACs include the pinned certificate fingerprint; a
  man-in-the-middle with a different certificate cannot relay a valid MAC.
* **Mutual:** a rogue coordinator that somehow presents the right certificate still cannot
  produce `proof` without the device secret.

After WELCOME the coordinator registers the worker, sends a bandwidth `PROBE` (Phase 6), and
the worker becomes eligible for tasks when the probe completes (or after 3 s). If the same
`worker_id` connects again while an old session exists, the old session is closed and its
in-flight attempts are reassigned; reputation is keyed by `worker_id`, so it carries over.
