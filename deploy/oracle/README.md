# Deploying FIE on Oracle Cloud Always Free

Free permanently. No trial credits, nothing to expire — which is the point,
since the previous deployment died when GCP credits ran out.

**Time: ~30 minutes**, most of it waiting for Oracle to provision the instance.

---

## Why this host

| | Oracle Always Free | GCP free tier | Render / Fly free |
| --- | --- | --- | --- |
| CPU / RAM | **4 ARM cores / 24 GB** | 0.25 vCPU / 1 GB | 0.5 vCPU / 512 MB |
| Expiry | **never** | 90-day credits | never |
| Sleeps when idle | **no** | no | yes (Render) |
| Cost after trial | **$0** | billed | $0 |

FIE needs well under 1 GB now that torch is gone, so a single free instance has
roughly 20× the headroom it needs. You could run the API, MongoDB and the
frontend on one box.

---

## Step 1 — Create the instance

1. Sign up at [cloud.oracle.com](https://cloud.oracle.com) (a card is required
   for identity verification; Always Free resources are never charged).
2. **Compute → Instances → Create Instance**
3. Change the shape — the default is x86 and *not* the free ARM one:
   - **Edit → Change Shape → Ampere → VM.Standard.A1.Flex**
   - **4 OCPUs, 24 GB memory** (the full free allocation)
4. **Image**: Canonical Ubuntu 22.04
5. **Add SSH key** — upload your public key, or let Oracle generate one and
   download it. You cannot retrieve it later.
6. Create.

> **If you get "Out of host capacity"** — this is common for free ARM capacity in
> popular regions. Either retry periodically, or pick a less busy home region.
> Capacity frees up; it is not a permanent block.

---

## Step 2 — Open the port at the cloud edge

**This is the step everyone misses.** Oracle enforces networking in two places,
and the host firewall alone is not enough — the instance will look completely
dead from outside while working fine over SSH.

**Networking → Virtual Cloud Networks → your VCN → Security Lists → Default
Security List → Add Ingress Rules:**

| Field | Value |
| --- | --- |
| Source CIDR | `0.0.0.0/0` |
| IP Protocol | TCP |
| Destination Port Range | `8080` |

The `setup.sh` script handles the host-level firewall (ufw *and* iptables —
Oracle's Ubuntu images enforce both).

---

## Step 3 — Provision

SSH in and run the setup script:

```bash
ssh -i /path/to/key ubuntu@<YOUR_PUBLIC_IP>

curl -fsSL https://raw.githubusercontent.com/AyushSingh110/Failure_Intelligence_System/main/deploy/oracle/setup.sh | bash
```

It installs Python 3.11, clones the repo, creates a virtualenv, installs
dependencies, downloads and SHA-256-verifies the model artifacts, writes an
`.env` template, and installs a hardened systemd unit.

---

## Step 4 — Configure and start

```bash
nano ~/fie/.env          # MONGODB_URI and JWT_SECRET_KEY are required
sudo systemctl start fie
sudo systemctl status fie
```

Generate a proper secret rather than inventing one:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

MongoDB Atlas has a permanent free M0 tier — create a cluster, add a database
user, and allow access from `0.0.0.0/0` (or just your instance IP).

---

## Step 5 — Verify

```bash
curl localhost:8080/health     # {"status":"healthy",...}
curl localhost:8080/ready      # 503 while cold, 200 after ~2s
curl localhost:8080/health/deep | python3 -m json.tool
```

`/health/deep` should show `"detector": {"status": "ok", "mode": "full_pipeline"}`.
If it says `reduced_recall`, the model artifacts did not load — re-run
`python scripts/download_models.py --strict`.

Then from your own machine:

```bash
curl http://<YOUR_PUBLIC_IP>:8080/health
```

If that hangs, the Security List ingress rule from Step 2 is missing.

---

## Step 6 — Point the dashboard at it

In Cloudflare Pages → your project → **Settings → Environment variables**:

```
VITE_API_URL = http://<YOUR_PUBLIC_IP>:8080/api/v1
```

Redeploy. The demo banner disappears automatically once the API responds — the
dashboard falls back to bundled sample data only while the backend is
unreachable.

---

## Optional — HTTPS

Browsers block `https://` pages calling `http://` endpoints (mixed content), so
if your dashboard is on HTTPS you need TLS on the API too.

Free options, in order of effort:

1. **Cloudflare Tunnel** — no public IP exposure, no certificate management,
   free. Easiest.
2. **Caddy** — automatic Let's Encrypt, two lines of config, needs a domain
   pointed at the instance.
3. **nginx + certbot** — most manual.

With a domain in hand, Caddy is the quickest:

```bash
sudo apt install -y caddy
sudo tee /etc/caddy/Caddyfile <<'EOF'
api.yourdomain.com {
    reverse_proxy localhost:8080
}
EOF
sudo systemctl restart caddy
```

Then set `CORS_ALLOWED_ORIGINS` in `.env` to your dashboard origin and restart.

---

## Operations

```bash
journalctl -u fie -f            # follow logs
sudo systemctl restart fie      # restart
cd ~/fie && git pull && sudo systemctl restart fie   # deploy an update
```

Logs are structured `key=value`, so degradations are greppable:

```bash
journalctl -u fie | grep 'capability='          # what degraded, and its impact
journalctl -u fie | grep 'status=degraded'      # scans with reduced coverage
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Connection times out from outside, works via SSH | Security List ingress missing | Step 2 |
| `/ready` returns 503 forever | Models failed to download | `python scripts/download_models.py --strict` |
| `detector: reduced_recall` in `/health/deep` | PAIR artifacts missing | as above |
| Service restart-loops | Bad `.env` | `journalctl -u fie -n 50` |
| "Out of host capacity" at create time | Free ARM capacity exhausted in region | retry, or change home region |
| Dashboard shows the demo banner | API unreachable from the browser | check `VITE_API_URL`, CORS, and mixed content |

---

## Cost

Zero, permanently, provided you stay within Always Free limits:

- 4 ARM OCPUs and 24 GB RAM total across A1 instances
- 200 GB block storage
- 10 TB/month egress

FIE uses a fraction of each.
