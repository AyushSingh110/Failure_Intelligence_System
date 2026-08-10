#!/usr/bin/env bash
#
# One-shot provisioning for an Oracle Cloud Always Free ARM instance.
#
# Target: VM.Standard.A1.Flex — 4 OCPU / 24 GB RAM, free permanently, no credit
# expiry. This is deliberately NOT another trial-credit host: the last deploy
# died when GCP credits ran out, and the point of this move is that there is
# nothing left to expire.
#
# FIE needs well under 1 GB since torch was removed, so a single free instance
# has enormous headroom.
#
# Usage (on a fresh Ubuntu 22.04 ARM instance):
#   curl -fsSL https://raw.githubusercontent.com/AyushSingh110/Failure_Intelligence_System/main/deploy/oracle/setup.sh | bash
# or clone the repo and run:
#   bash deploy/oracle/setup.sh
#
set -euo pipefail

REPO_URL="https://github.com/AyushSingh110/Failure_Intelligence_System.git"
APP_DIR="${APP_DIR:-$HOME/fie}"
SERVICE="fie"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ── Preconditions ─────────────────────────────────────────────────────────────
[[ $EUID -eq 0 ]] && die "Do not run as root. Run as the 'ubuntu' user; the script uses sudo where needed."

log "Detected: $(uname -m) | $(. /etc/os-release && echo "$PRETTY_NAME")"

# ── System packages ───────────────────────────────────────────────────────────
log "Installing system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3.11 python3.11-venv python3-pip git curl ufw

# ── Firewall ──────────────────────────────────────────────────────────────────
# Oracle images ship with a restrictive iptables config AND a cloud-level
# Security List. Both must allow the port or the instance looks dead from
# outside while working fine over SSH — the single most common Oracle gotcha.
log "Opening port 8080 (host firewall)"
sudo ufw allow OpenSSH   >/dev/null 2>&1 || true
sudo ufw allow 8080/tcp  >/dev/null 2>&1 || true
sudo ufw --force enable  >/dev/null 2>&1 || true
# Oracle's Ubuntu images also enforce iptables directly; ufw alone is not enough.
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8080 -j ACCEPT 2>/dev/null || true
sudo netfilter-persistent save >/dev/null 2>&1 || true

cat <<'REMINDER'

  !! ALSO required, in the Oracle Cloud console (this script cannot do it):
     Networking -> Virtual Cloud Networks -> your VCN -> Security Lists
     -> Add Ingress Rule:  Source 0.0.0.0/0,  IP Protocol TCP,  Dest port 8080

     Without this the port stays closed at the cloud edge no matter what the
     host firewall says.

REMINDER

# ── Application ───────────────────────────────────────────────────────────────
if [[ -d "$APP_DIR/.git" ]]; then
  log "Updating existing checkout at $APP_DIR"
  git -C "$APP_DIR" pull --ff-only
else
  log "Cloning into $APP_DIR"
  git clone --depth 1 "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

log "Creating virtualenv"
python3.11 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

log "Downloading model artifacts (SHA-256 verified)"
python scripts/download_models.py --strict

# ── Configuration ─────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
  log "Creating .env template"
  cat > .env <<'ENVEOF'
# ── Required ──────────────────────────────────────────────────────────────────
MONGODB_URI=mongodb+srv://USER:PASS@cluster.mongodb.net/
MONGODB_DB_NAME=fie_database
JWT_SECRET_KEY=CHANGE_ME_TO_32_PLUS_RANDOM_CHARS
ADMIN_EMAIL=you@example.com

# ── Optional ──────────────────────────────────────────────────────────────────
GROQ_API_KEY=                 # hallucination monitoring only
SERPER_API_KEY=
SENTRY_DSN=                   # never ships prompt text

# ── Recommended production settings ───────────────────────────────────────────
# Block instead of forwarding unscanned prompts when the scanner itself fails.
FIE_SCAN_FAILURE_MODE=closed
FIE_LAYER_POOL_SIZE=8
CORS_ALLOWED_ORIGINS=https://failure-intelligence-system.pages.dev
DEBUG=false
ENVEOF
  chmod 600 .env
  echo
  echo "  Edit $APP_DIR/.env before starting the service:"
  echo "    nano $APP_DIR/.env"
  echo
fi

# ── systemd unit ──────────────────────────────────────────────────────────────
log "Installing systemd service"
sudo tee /etc/systemd/system/${SERVICE}.service >/dev/null <<UNITEOF
[Unit]
Description=Failure Intelligence Engine API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
Restart=always
RestartSec=5

# Model warm-up takes ~2s; allow generous headroom on a cold ARM boot.
TimeoutStartSec=120

# Hardening. FIE writes only to ./storage and reads its own models.
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=read-only
ReadWritePaths=$APP_DIR

[Install]
WantedBy=multi-user.target
UNITEOF

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE} >/dev/null

log "Setup complete"
cat <<DONE

  Next steps:

    1. nano $APP_DIR/.env            # fill in MONGODB_URI and JWT_SECRET_KEY
    2. sudo systemctl start $SERVICE
    3. sudo systemctl status $SERVICE
    4. curl localhost:8080/health
       curl localhost:8080/ready     # 503 until models warm (~2s), then 200

  Logs:     journalctl -u $SERVICE -f
  Restart:  sudo systemctl restart $SERVICE

  Remember the Security List ingress rule in the Oracle console (see above).

DONE
