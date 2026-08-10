#!/usr/bin/env bash
#
# Give the FIE API a free, stable HTTPS URL — no domain purchase required.
#
# WHY THIS IS NOT OPTIONAL
# ------------------------
# The dashboard is served from Cloudflare Pages over HTTPS. Browsers refuse to
# let an HTTPS page call an HTTP endpoint (mixed-content blocking), so a bare
# http://<ip>:8080 API is invisible to the dashboard no matter how healthy it
# is. curl and the Python SDK work fine over plain HTTP; browsers do not.
#
# This script wires up:
#   DuckDNS   free subdomain  -> yourname.duckdns.org  (stable, survives reboots)
#   Caddy     automatic TLS   -> Let's Encrypt certificate, auto-renewing
#   reverse proxy             -> :443 public  ->  :8080 local
#
# Usage:
#   bash deploy/oracle/enable-https.sh <duckdns-subdomain> <duckdns-token>
#
# Get both from https://www.duckdns.org (sign in with GitHub, create a
# subdomain, copy the token shown at the top of the page).
#
set -euo pipefail

SUBDOMAIN="${1:-}"
TOKEN="${2:-}"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[[ -z "$SUBDOMAIN" || -z "$TOKEN" ]] && die "Usage: $0 <duckdns-subdomain> <duckdns-token>
  e.g. $0 fie-api 8f3c1a2b-...

  Create both free at https://www.duckdns.org"

[[ $EUID -eq 0 ]] && die "Run as the 'ubuntu' user, not root. The script uses sudo where needed."

DOMAIN="${SUBDOMAIN}.duckdns.org"

# ── 1. Open 80 and 443 ────────────────────────────────────────────────────────
# Port 80 is required for the Let's Encrypt HTTP-01 challenge, not just 443.
log "Opening ports 80 and 443"
sudo ufw allow 80/tcp  >/dev/null 2>&1 || true
sudo ufw allow 443/tcp >/dev/null 2>&1 || true
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80  -j ACCEPT 2>/dev/null || true
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT 2>/dev/null || true
sudo netfilter-persistent save >/dev/null 2>&1 || true

cat <<REMINDER

  !! ALSO required in the Oracle console (this script cannot do it):
     Networking -> Virtual Cloud Networks -> your VCN -> Security Lists
     -> Add Ingress Rules for BOTH:
          Source 0.0.0.0/0  TCP  port 80
          Source 0.0.0.0/0  TCP  port 443

     Port 80 is needed for certificate issuance. Without it, Caddy will fail
     to obtain a certificate and retry forever.

REMINDER
read -r -p "Press Enter once those ingress rules exist (or Ctrl-C to abort)... " _

# ── 2. Point DuckDNS at this instance ─────────────────────────────────────────
log "Pointing ${DOMAIN} at this instance"
PUBLIC_IP="$(curl -fsS https://api.ipify.org)"
echo "  public IP: ${PUBLIC_IP}"

RESULT="$(curl -fsS "https://www.duckdns.org/update?domains=${SUBDOMAIN}&token=${TOKEN}&ip=${PUBLIC_IP}")"
[[ "$RESULT" == "OK" ]] || die "DuckDNS update failed (response: '${RESULT}'). Check the subdomain and token."
echo "  DuckDNS: OK"

# Keep the record fresh. Oracle public IPs are stable, but a reboot or a
# reattached ephemeral IP would otherwise silently break the domain.
log "Installing DuckDNS refresh timer (every 30 min)"
sudo tee /usr/local/bin/duckdns-update >/dev/null <<UPDEOF
#!/bin/sh
curl -fsS "https://www.duckdns.org/update?domains=${SUBDOMAIN}&token=${TOKEN}&ip=" >/dev/null
UPDEOF
sudo chmod +x /usr/local/bin/duckdns-update
sudo tee /etc/systemd/system/duckdns.service >/dev/null <<'SVCEOF'
[Unit]
Description=Refresh DuckDNS record
After=network-online.target
[Service]
Type=oneshot
ExecStart=/usr/local/bin/duckdns-update
SVCEOF
sudo tee /etc/systemd/system/duckdns.timer >/dev/null <<'TMREOF'
[Unit]
Description=Refresh DuckDNS every 30 minutes
[Timer]
OnBootSec=2min
OnUnitActiveSec=30min
[Install]
WantedBy=timers.target
TMREOF
sudo systemctl daemon-reload
sudo systemctl enable --now duckdns.timer >/dev/null

# ── 3. Install Caddy ──────────────────────────────────────────────────────────
if ! command -v caddy >/dev/null 2>&1; then
  log "Installing Caddy"
  sudo apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq caddy
fi

# ── 4. Configure the reverse proxy ────────────────────────────────────────────
log "Configuring Caddy for ${DOMAIN}"
sudo tee /etc/caddy/Caddyfile >/dev/null <<CADDYEOF
${DOMAIN} {
    reverse_proxy localhost:8080

    # FIE emits structured key=value logs; keep Caddy's access log JSON so the
    # two are distinguishable in journalctl.
    log {
        output file /var/log/caddy/access.log
        format json
    }

    encode gzip
}
CADDYEOF

sudo mkdir -p /var/log/caddy && sudo chown caddy:caddy /var/log/caddy
sudo systemctl restart caddy

# ── 5. Verify ─────────────────────────────────────────────────────────────────
log "Waiting for the certificate (Let's Encrypt usually takes 10-30s)"
for i in $(seq 1 30); do
  if curl -fsS --max-time 5 "https://${DOMAIN}/health" >/dev/null 2>&1; then
    echo "  certificate issued"
    break
  fi
  sleep 3
  [[ $i -eq 30 ]] && {
    echo
    echo "  Certificate not issued yet. Check: sudo journalctl -u caddy -n 40"
    echo "  The usual cause is port 80 being closed in the Oracle Security List."
  }
done

echo
echo "──────────────────────────────────────────────────────────────"
echo "  Public API URL:  https://${DOMAIN}"
echo
curl -fsS --max-time 10 "https://${DOMAIN}/health" 2>/dev/null | head -c 300 || echo "  (not responding yet — see journalctl -u caddy)"
echo
echo "──────────────────────────────────────────────────────────────"
cat <<NEXTEOF

  Next:

    1. Set this in Cloudflare Pages -> Settings -> Environment variables:

         VITE_API_URL = https://${DOMAIN}/api/v1

       then redeploy the dashboard.

    2. Add the dashboard origin to CORS on the API:

         nano ~/fie/.env
         CORS_ALLOWED_ORIGINS=https://failure-intelligence-system.pages.dev
         sudo systemctl restart fie

  Checks:
    curl https://${DOMAIN}/health
    curl https://${DOMAIN}/ready
    curl https://${DOMAIN}/health/deep | python3 -m json.tool

NEXTEOF
