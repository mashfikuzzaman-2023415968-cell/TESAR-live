# Security policy — TESAR-Live

TESAR-Live aims for a production-grade security posture **for a LAN distributed-computing
research prototype**: secure defaults, defense in depth, fail-closed behaviour, and a test for
every control. The full threat model and controls are in
[docs/SECURITY_DESIGN.md](docs/SECURITY_DESIGN.md).

## Threat model summary

| ID | Adversary | Main controls |
|---|---|---|
| A1 | Passive LAN eavesdropper | TLS on all worker traffic |
| A2 | Active LAN attacker (spoofing, MITM, floods, malformed frames) | certificate pinning, mutual auth with channel binding, enrollment, strict validation, rate limits, timeouts |
| A3 | Compromised/malicious enrolled worker | result verification + quarantine, clamping self-reported values, attempt ownership, loopback-only trusted local worker |
| A4 | Malicious coordinator (from a phone's view) | whitelisted task kinds only, local parameter limits, CONTROL gated by lab mode, no inbound ports, no shell/file access |
| A5 | Web attacker via the operator's browser | dashboard auth, CSRF tokens, Origin/Host checks, CSP, safe DOM rendering |
| A6 | Other local users on the laptop | state dir 0700, secret files 0600, secrets never in argv/logs |
| A7 | Supply chain | minimal deps, hash-pinned locks, pip-audit, bandit |

## Status

Every control of the specification's Section S is implemented and covered by a test (most in
`tests/security/`): transport (TLS, pinning, channel-bound mutual authentication), enrollment and
revocation, strict validation and resource limits, worker-device protection, result verification
with quarantine and collusion resistance, the dashboard (S.9), secrets handling, and supply chain
(hash-pinned locks, pip-audit, bandit, forbidden-pattern scan). The phase gate enforces them on
every change. `docs/SECURITY_DESIGN.md` maps each control to its code and tests.

Measured residual risk 1 (experiment E5, `docs/results/e5_malicious`): with the default
spot-check probability 0.05, a worker lying on 5 % of its results let a wrong tile reach the output
in 1 of 5 runs; at 0.2 or more, all runs were correct. For results that must be right, use
`--set verification.base_prob=1.0` (about 20 % compute overhead in E5).

## Reporting a problem

Report security problems privately to the project author rather than in a public issue.
Please include steps to reproduce.

## Residual risks

1. An enrolled, compromised device can still waste time (slow responses) and, with low
   verification probability, a wrong result can occasionally survive (quantified in experiment E5).
2. Device secrets are stored readable by the coordinator's OS user; anyone with that account
   controls the system.
3. Trust bootstrapping depends on keeping join codes confidential during transfer.
4. Workers necessarily see the image tiles they process — do not offload sensitive images to
   devices you don't trust.
5. Radio-level jamming/deauthentication attacks on WiFi cannot be prevented by software.
6. The system has not had an independent security review; treat it as a well-defended research
   prototype, and get a review before any real deployment.
7. Without a configured trusted local worker, and with no other live worker, a verification check
   cannot run and is skipped (`verification_skipped` event). Keep the local worker enabled
   (the default).
8. Dashboard demo controls (crash, make malicious) exist for lab demonstrations. They need
   `security.lab_mode` **and** `dashboard.demo_controls` on the coordinator, and
   `--lab-mode --allow-demo-control` on each worker. Never enable them outside a lab.
