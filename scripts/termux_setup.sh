#!/data/data/com.termux/files/usr/bin/sh
# One-time setup of a TESAR-Live worker inside Termux (Android). Run from the
# unzipped mdc-worker directory:  sh scripts/termux_setup.sh
set -e
pkg update -y && pkg upgrade -y
pkg install -y python python-numpy termux-api unzip
termux-wake-lock            # keep the CPU awake while the worker runs
echo "Install the Termux:API app (F-Droid) if termux-battery-status fails"
termux-battery-status || true
python -c "import numpy, sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__)"
echo "Next: get your join code by QR scan, then"
echo "  termux-clipboard-get > phone.join && chmod 600 phone.join"
echo "  python -m mdc.worker --join-file phone.join --device-class phone --battery termux"
