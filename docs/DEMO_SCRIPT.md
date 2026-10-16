# Five-minute live demo

**Setup before the audience arrives (10 min)**

- [ ] Laptop on the demo WiFi (or your own router / phone hotspot). Run
      `python -m mdc.coordinator --diag`: note the LAN address, check port 50010 is free and the
      beacon is sent.
- [ ] Phones enrolled (`python -m mdc.cli enroll --name phone-a --qr`), Termux open, battery
      optimization off, `termux-wake-lock`.
- [ ] Start the coordinator **with demo controls**:
      ```bash
      python -m mdc.coordinator --set security.lab_mode=true --set dashboard.demo_controls=true \
          --set network.max_connections_per_ip=8
      ```
      Open the printed dashboard URL and log in with the token from `~/.mdc/admin_token`.
- [ ] Start the phones' workers with `--lab-mode --allow-demo-control`, so the demo buttons work
      on them:
      `python -m mdc.worker --join-file phone-a.join --device-class phone --battery termux --lab-mode --allow-demo-control`
- [ ] Optional extra devices on the laptop:
      `python scripts/local_cluster.py configs/cluster_profiles/hetero5.yaml` (lab-mode profile;
      add `--allow-demo-control` to a profile's args if you want buttons for them).
- [ ] Have `docs/results/e2_alpha_sweep/e2_tradeoff_curve.png` open in a second tab.

## The script

| Time | Do | Say |
|---|---|---|
| 0:00 | Show the dashboard header ("Workers connect to …") and the worker table as phones appear (beacon discovery). | "The laptop is the coordinator. Phones find it by UDP beacon and connect over TLS. Each phone was enrolled with its own key, and the certificate is pinned." |
| 0:45 | Start a job with **policy `local`** (image `test_2048.png`, 8 iterations). Note the elapsed time. | "Baseline: the laptop alone." |
| 1:30 | Start the same job with **`tesar`, α = 0.5**. Point at the tile canvas coloured by device, the chunk sizes and the reputation bars. | "TESAR trades speed against reliability for every chunk: fast devices get bigger chunks, unreliable ones get replicas." |
| 2:30 | Start another job. Mid-job, press **Crash** on a phone (or walk it out of range). | "The phone goes SUSPECT, then DEAD; its tiles return to pending and other devices take them. The output is still byte-exact." Point at the ✓ badge. |
| 3:30 | Press **Make malicious** on another phone, then start a job. | "This phone now returns plausible but wrong tiles with valid checksums. The trusted laptop spot-checks results: the red flashes are caught lies. After two strikes it is quarantined, and everything it produced is recomputed (taint propagation). Output: still correct ✓." |
| 4:30 | Show the E2 tradeoff figure. | "Sweeping α moves us along the speed–reliability tradeoff, the central idea of the TESAR paper." |

## Fallback plans

* **Phones cannot reach the laptop** (campus WiFi isolates clients: `curl http://LAPTOP_IP:8000`
  fails while `python -m http.server 8000` runs): switch to a personal router or a phone hotspot.
  The hotspot phone can itself be a worker.
* **Beacons blocked:** start workers with `--coordinator LAPTOP_IP:50010`.
* **No phones at all:** run the laptop-only demo. Start the coordinator as above, then
  `python scripts/local_cluster.py configs/cluster_profiles/hetero_faulty6.yaml` for failures, or
  `hetero_malicious6.yaml` for liars. The demo buttons work on profile workers whose args include
  `--allow-demo-control`. `scripts/chaos.py --stop NAME@T --cont NAME@T` freezes a worker without
  the dashboard.
* **Dashboard trouble:** every step also works headless. Use `--headless --run-job`, watch
  `runs/latest/events.jsonl` with `tail -f`, and open `runs/latest/output.png`.
