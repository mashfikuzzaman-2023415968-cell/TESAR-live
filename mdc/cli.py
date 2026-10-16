"""Helper command line: ``python -m mdc.cli <command>``.

Enrollment and certificate commands (Phase 1, spec S.4.4):

* ``enroll --name NAME [--out FILE] [--qr] [--host H] [--port P]`` — enroll a
  device and write its join file (mode 0600). ``--qr`` prints an ASCII QR code
  of the join code instead (for phones). The code itself is never echoed.
* ``revoke --name NAME`` — the device's secret stops working immediately.
* ``rotate-secret --name NAME --out FILE`` — new secret, same worker id.
* ``list-devices`` — table of devices; never prints secrets.
* ``show-fingerprint`` — the pinned certificate fingerprint.
* ``rotate-cert --yes [--reissue-dir DIR]`` — new certificate; all join codes
  must be re-issued (written to DIR if given).

Compute commands (Phase 2):

* ``reference --image IMG --r 2 --iters 8 --out ref.png`` — whole-image ground truth.
* ``compare a.png b.png`` — "identical" or the number of differing pixels
  (exit status 0 if identical, 1 otherwise).
* ``bench`` — this machine's milliseconds per work unit.

Measurement (Phase 12):

* ``calibrate-energy --runs DIR`` — suggest energy constants from measured runs.

Common options: ``--config FILE``, ``--set key=value`` (repeatable). The
state directory is ``security.state_dir`` from the configuration.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from mdc import calibrate
from mdc.common.config import Config, ConfigError, load_config
from mdc.common.credentials import JoinCode, encode_join_code, write_join_file
from mdc.common.validation import TaskLimits, ValidationError
from mdc.coordinator.beacon import detect_lan_ip
from mdc.coordinator.certs import CertificateError, ensure_certificate, rotate_certificate
from mdc.coordinator.enrollment import DeviceRegistry, EnrollmentError
from mdc.coordinator.job_builder import ImageLoadError, load_image, save_image
from mdc.coordinator.security_log import SecurityLog
from mdc.tasks.benchmark import run_benchmark
from mdc.tasks.image_pipeline import reference


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")


def _endpoint(cfg: Config, args: argparse.Namespace) -> tuple[str, int]:
    host = args.host or cfg.network.advertise_host or detect_lan_ip()
    port = args.port or cfg.network.tcp_port
    return host, port


class _Resources:
    """Lazily opened registry + security log, closed when the command ends."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._sec_log: SecurityLog | None = None
        self._registry: DeviceRegistry | None = None

    @property
    def security_log(self) -> SecurityLog:
        if self._sec_log is None:
            self._sec_log = SecurityLog.for_state_dir(
                self.cfg.security.state_dir, self.cfg.security.security_log
            )
        return self._sec_log

    @property
    def registry(self) -> DeviceRegistry:
        if self._registry is None:
            self._registry = DeviceRegistry(self.cfg.security.state_dir, self.security_log)
        return self._registry

    def close(self) -> None:
        if self._sec_log is not None:
            self._sec_log.close()


def _print_qr(code: JoinCode) -> None:
    try:
        import qrcode  # noqa: PLC0415 - optional dependency
    except ImportError:
        raise SystemExit("--qr needs the optional 'qrcode' package (pip install qrcode)") from None
    qr = qrcode.QRCode(border=1)
    qr.add_data(encode_join_code(code))
    qr.print_ascii(invert=True)


def _write_or_show(code: JoinCode, out: str | None, qr: bool) -> None:
    if qr:
        _print_qr(code)
        print("Scan this QR code on the phone; keep it private (it contains a secret).")
    if out or not qr:
        path = write_join_file(out or f"{code.name}.join", code)
        print(f"Join file written: {path} (mode 0600). Transfer it privately (QR/USB).")


def cmd_enroll(res: _Resources, args: argparse.Namespace) -> int:
    """Enroll a new device."""
    cfg = res.cfg
    cert = ensure_certificate(cfg.security.state_dir, cfg.security.cert_days)
    host, port = _endpoint(cfg, args)
    registry = res.registry
    code = registry.enroll(args.name, host, port, cert.fingerprint)
    print(f"Enrolled {code.name} as {code.worker_id} -> {host}:{port}")
    _write_or_show(code, args.out, args.qr)
    return 0


def cmd_revoke(res: _Resources, args: argparse.Namespace) -> int:
    """Revoke a device."""
    registry = res.registry
    device = registry.revoke(args.name)
    print(
        f"Revoked {device.name} ({device.worker_id}). A running coordinator closes its "
        "session within 2 s."
    )
    return 0


def cmd_rotate_secret(res: _Resources, args: argparse.Namespace) -> int:
    """Give a device a new secret."""
    cfg = res.cfg
    cert = ensure_certificate(cfg.security.state_dir, cfg.security.cert_days)
    host, port = _endpoint(cfg, args)
    registry = res.registry
    code = registry.rotate_secret(args.name, host, port, cert.fingerprint)
    print(f"New secret for {code.name}; the old join code no longer works.")
    _write_or_show(code, args.out, args.qr)
    return 0


def cmd_list_devices(res: _Resources, args: argparse.Namespace) -> int:
    """Print enrolled devices (no secrets)."""
    registry = res.registry
    rows = registry.list_devices()
    if not rows:
        print("No devices enrolled.")
        return 0
    print(f"{'name':<20} {'worker_id':<20} {'status':<8} {'created':<26} last_seen")
    for row in rows:
        status = "revoked" if row["revoked"] else "active"
        print(
            f"{row['name']:<20} {row['worker_id']:<20} {status:<8} {row['created']:<26} "
            f"{row['last_seen'] or '-'}"
        )
    return 0


def cmd_show_fingerprint(res: _Resources, args: argparse.Namespace) -> int:
    """Print the coordinator certificate fingerprint."""
    cfg = res.cfg
    cert = ensure_certificate(cfg.security.state_dir, cfg.security.cert_days)
    print(cert.fingerprint)
    return 0


def cmd_rotate_cert(res: _Resources, args: argparse.Namespace) -> int:
    """Replace the certificate (all join codes must be re-issued)."""
    cfg = res.cfg
    if not args.yes:
        print(
            "rotate-cert replaces the coordinator certificate: EVERY join code stops "
            "working and must be re-issued. Re-run with --yes to proceed."
        )
        return 2
    cert = rotate_certificate(cfg.security.state_dir, cfg.security.cert_days)
    registry = res.registry
    res.security_log.emit("certificate_rotated", fingerprint=cert.fingerprint)
    print(f"New certificate fingerprint: {cert.fingerprint}")
    if args.reissue_dir:
        host, port = _endpoint(cfg, args)
        out_dir = Path(args.reissue_dir)
        for code in registry.join_codes(host, port, cert.fingerprint):
            path = write_join_file(out_dir / f"{code.name}.join", code)
            print(f"Re-issued {path}")
    else:
        print("Re-issue join codes with: python -m mdc.cli rotate-cert ... --reissue-dir DIR")
    return 0


def cmd_reference(res: _Resources, args: argparse.Namespace) -> int:
    """Render the whole-image reference."""
    image = load_image(args.image)
    limits = res.cfg.security.max_task_params
    out = reference(
        image,
        args.r,
        args.iters,
        TaskLimits(
            limits.blur_radius, limits.blur_iterations, limits.chain_len, limits.max_tile_pixels
        ),
    )
    path = save_image(args.out, out)
    print(
        f"Reference {image.shape[1]}x{image.shape[0]} (r={args.r}, iters={args.iters}) "
        f"written to {path}"
    )
    return 0


def cmd_compare(res: _Resources, args: argparse.Namespace) -> int:
    """Compare two images pixel by pixel."""
    a, b = load_image(args.a), load_image(args.b)
    if a.shape != b.shape:
        print(f"different shapes: {a.shape[1]}x{a.shape[0]} vs {b.shape[1]}x{b.shape[0]}")
        return 1
    differing = int(np.count_nonzero(a != b))
    if differing == 0:
        print("identical")
        return 0
    print(f"{differing} differing pixels ({100.0 * differing / a.size:.4f} %)")
    return 1


def cmd_bench(res: _Resources, args: argparse.Namespace) -> int:
    """Print this machine's speed score."""
    print(f"{run_benchmark(args.runs):.4f} ms per work unit")
    return 0


def cmd_calibrate_energy(res: _Resources, args: argparse.Namespace) -> int:
    """Print suggested energy constants fitted to measured runs."""
    runs = Path(args.runs)
    if not runs.is_dir():
        print(f"error: {runs} is not a directory", file=sys.stderr)
        return 1
    samples = calibrate.load_samples(runs)
    fits = calibrate.fit(samples)
    print(calibrate.report_text(fits, len(samples)))
    return 0 if fits else 1


def build_parser() -> argparse.ArgumentParser:
    """The argument parser (exposed for tests)."""
    parser = argparse.ArgumentParser(prog="python -m mdc.cli", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str, endpoint: bool = False) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        _common(p)
        if endpoint:
            p.add_argument("--host", default=None, help="coordinator address for the join code")
            p.add_argument("--port", type=int, default=None)
        return p

    p = add("enroll", "enroll a new worker device", endpoint=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", default=None, help="join file to write (default NAME.join)")
    p.add_argument("--qr", action="store_true", help="print an ASCII QR code")
    p = add("revoke", "revoke a device")
    p.add_argument("--name", required=True)
    p = add("rotate-secret", "issue a new secret for a device", endpoint=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--qr", action="store_true")
    add("list-devices", "list enrolled devices (no secrets)")
    add("show-fingerprint", "print the certificate fingerprint")
    p = add("rotate-cert", "replace the TLS certificate", endpoint=True)
    p.add_argument("--yes", action="store_true", help="confirm")
    p.add_argument("--reissue-dir", default=None, help="write new join files here")
    p = add("reference", "render the whole-image reference")
    p.add_argument("--image", required=True)
    p.add_argument("--r", type=int, default=2, help="blur radius")
    p.add_argument("--iters", type=int, default=8, help="blur iterations")
    p.add_argument("--out", required=True)
    p = add("compare", "compare two images pixel by pixel")
    p.add_argument("a")
    p.add_argument("b")
    p = add("bench", "measure this machine's ms per work unit")
    p.add_argument("--runs", type=int, default=5)
    p = add("calibrate-energy", "suggest energy constants from measured runs")
    p.add_argument("--runs", required=True, help="directory with report.json files")
    return parser


COMMANDS: dict[str, Callable[[_Resources, argparse.Namespace], int]] = {
    "enroll": cmd_enroll,
    "revoke": cmd_revoke,
    "rotate-secret": cmd_rotate_secret,
    "list-devices": cmd_list_devices,
    "show-fingerprint": cmd_show_fingerprint,
    "rotate-cert": cmd_rotate_cert,
    "reference": cmd_reference,
    "compare": cmd_compare,
    "bench": cmd_bench,
    "calibrate-energy": cmd_calibrate_energy,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Errors are reported as one line, without tracebacks."""
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, args.overrides)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    resources = _Resources(cfg)
    try:
        return COMMANDS[args.command](resources, args)
    except (
        EnrollmentError,
        CertificateError,
        ImageLoadError,
        ValidationError,
        ValueError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        resources.close()


if __name__ == "__main__":
    sys.exit(main())
