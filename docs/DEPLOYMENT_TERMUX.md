# Deploying workers on Android (Termux)

This guide turns Android phones into TESAR-Live workers. The laptop runs the
coordinator; each phone runs `python -m mdc.worker` inside Termux. The code zip
contains **no secrets**; each phone's join code is a secret and travels only by
QR code or USB.

## 0. Before you start (laptop)

```bash
python -m mdc.coordinator --diag          # interfaces, ports, test beacon, fingerprint
sh scripts/package_worker.sh              # -> dist/mdc-worker.zip and .sha256
cat dist/mdc-worker.zip.sha256            # note the hash: you will check it on the phone
```

Phone and laptop must be on the same network. Campus or café WiFi often blocks
traffic between clients (see §6).

## 1. Install Termux and Termux:API

* Install **Termux** and **Termux:API** from **F-Droid**. The Play Store builds are
  outdated and cannot install packages. Both apps must come from the same source.
* Open Termux:API once and grant its permissions.
* Android Settings → Apps → Termux → Battery → **Unrestricted** (no battery
  optimization), and keep Termux's notification. Otherwise Android freezes the
  worker as soon as the screen turns off.

## 2. Get the code onto the phone

On the laptop, serve **only the zip**, then stop the server:

```bash
python -m http.server 8000 --directory dist      # Ctrl+C once the phone has the file
```

On the phone (Termux):

```bash
pkg install -y curl unzip
curl -O http://LAPTOP_IP:8000/mdc-worker.zip
curl -O http://LAPTOP_IP:8000/mdc-worker.zip.sha256
sha256sum -c mdc-worker.zip.sha256               # must print "mdc-worker.zip: OK"
unzip mdc-worker.zip -d mdc-worker && cd mdc-worker
sh scripts/termux_setup.sh                       # python, numpy, termux-api, wake lock
```

If `sha256sum -c` does not print `OK`, delete the file and download it again. Do
not run a zip whose hash differs from the one printed on the laptop.

## 3. Enroll the phone (join code by QR, never over HTTP)

On the laptop:

```bash
python -m mdc.cli enroll --name phone-rafi --qr
```

Scan the QR code with any scanner app, copy the text, and in Termux:

```bash
termux-clipboard-get > phone-rafi.join && chmod 600 phone-rafi.join
```

Alternative over USB: `python -m mdc.cli enroll --name phone-rafi --out /tmp/phone-rafi.join`,
then `adb push /tmp/phone-rafi.join /sdcard/` and move it into Termux's home directory
(`mv /sdcard/phone-rafi.join ~/mdc-worker/ && chmod 600 ~/mdc-worker/phone-rafi.join`). Delete
the copy in `/sdcard` and on the laptop afterwards.

The worker refuses a join file that other users can read.

## 4. Start the worker

```bash
termux-wake-lock
python -m mdc.worker --join-file phone-rafi.join --device-class phone --battery termux
```

The worker finds the coordinator by UDP beacon. If discovery fails, give the address explicitly:
`--coordinator LAPTOP_IP:50010`. The pinned certificate fingerprint in the join code still
protects the connection, so a typo or a rogue host cannot impersonate the coordinator.

Diagnostics (prints no secrets): `python -m mdc.worker --join-file phone-rafi.join --diag
--coordinator LAPTOP_IP:50010`. It shows the platform, the battery reading, a benchmark,
whether a beacon was heard in 5 s, and a TLS connection test.

## 5. Lost or stolen phone

```bash
python -m mdc.cli revoke --name phone-rafi
```

Revocation takes effect immediately, and an active session is closed. Enroll the phone again
(new secret) if it comes back.

## 6. Network troubleshooting checklist

1. **Laptop firewall.** Allow inbound TCP 50010 and UDP from the LAN. On Windows Defender
   Firewall, allow Python on Private networks. On Linux: `sudo ufw allow 50010/tcp`.
2. **Client isolation.** Campus and public WiFi often block device-to-device traffic. Test it
   from the phone with `curl http://LAPTOP_IP:8000` while `python -m http.server 8000` runs on
   the laptop. If that fails, use a personal router or a phone hotspot (the hotspot phone itself
   can be a worker).
3. **Beacons blocked.** Some networks drop broadcast and multicast. Start the worker with
   `--coordinator LAPTOP_IP:50010`.
4. **Wrong interface on the laptop.** A VPN or virtual adapter (Docker, VirtualBox) can make the
   coordinator advertise the wrong address. Check `python -m mdc.coordinator --diag` and set
   `--set network.advertise_host=192.168.x.y`.

## 7. Measuring energy

* Battery current is read from Termux:API (`termux-battery-status`). The phone must be
  **unplugged**: while charging, the coordinator does not count measured energy.
* The screen dominates drain, so keep screens off during runs and note the screen state in
  `OBSERVATIONS.md`.
* `report.json` gives each worker's measured energy (`energy_meas_j`), the idle baseline before
  the job (`idle_baseline_w`), and the measured compute and transfer seconds. After a few runs,
  `python -m mdc.cli calibrate-energy --runs runs/` suggests `energy.p_compute_w_phone` and
  `energy.p_tx_w_phone`. Adopt values deliberately and record them in `DECISIONS.md`.

## 8. Real-device acceptance checklist (Phase 12)

- [ ] `python -m mdc.coordinator --diag` shows the LAN address and a free port 50010.
- [ ] Two phones run `--diag` successfully: beacon heard (or `--coordinator` connects) and TLS ok.
- [ ] Run the coordinator with the local worker:
      `python -m mdc.coordinator --run-job --wait-workers 2 --set job.image_path=data/images/test_1024.png`.
      The job finishes with `output correct ✓`.
- [ ] During a second job, take one phone out of WiFi range (or switch on airplane mode). It goes
      SUSPECT → DEAD in the log, its tiles are reassigned, and the output is still correct.
- [ ] Write what happened (WiFi type, phone models, Android versions, any surprises) in
      `OBSERVATIONS.md` using its template.
