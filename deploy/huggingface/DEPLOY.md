# Deploying FIE to Hugging Face Spaces

**~10 minutes. No credit card. Free HTTPS. No domain needed.**

You get one URL — e.g. `https://ayushsingh110-fie.hf.space` — serving both the
interactive demo and the full API.

---

## Why Spaces rather than a VPS

| | HF Spaces | Oracle Always Free |
| --- | --- | --- |
| Credit card | **not required** | required |
| HTTPS | **automatic** | DuckDNS + Caddy |
| Setup | **~10 min, git push** | ~45 min, SSH + systemd |
| Capacity | always available | "Out of host capacity" is common |
| Idle policy | sleeps, **wakes on request** | **instance reclaimed** after 7 idle days |
| Audience | ML researchers | none |

Oracle reclaims Always Free compute when CPU, network and memory all sit below
20% for 7 days — which is exactly what a low-traffic demo backend looks like.
Spaces sleeping and waking is a far better failure mode than the box being
deleted.

`deploy/oracle/` is still in this repo for when you want a dedicated always-on
backend.

---

## Step 1 — Create the Space

1. Sign in at [huggingface.co](https://huggingface.co) (free, no card)
2. **New → Space**
3. Fill in:

   | Field | Value |
   | --- | --- |
   | Owner | your username |
   | Space name | `fie` (or anything) |
   | License | Apache 2.0 |
   | SDK | **Docker** → **Blank** |
   | Hardware | **CPU basic — FREE** |
   | Visibility | Public |

4. Create.

---

## Step 2 — Push the code

The Space is a git repo. Push this project into it, with two Space-specific
files placed at the repo root.

```bash
cd ~/Desktop/Failure_Intelligence_System

# Add the Space as a second remote
git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/fie

# Spaces need the Dockerfile and README.md AT THE ROOT.
# Use a dedicated branch so main stays clean.
git checkout -b space

cp deploy/huggingface/Dockerfile      ./Dockerfile.space
cp deploy/huggingface/SPACE_README.md ./README.space.md

# Swap them into place on this branch only
mv Dockerfile.space  Dockerfile
mv README.space.md   README.md

git add -A
git commit -m "Hugging Face Space configuration"
git push space space:main

git checkout main       # back to normal
```

When prompted for a password, use a **write token** from
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), not
your account password.

The build takes ~4 minutes (mostly baking in the 90 MB ONNX model). Watch it
under the **Logs** tab.

---

## Step 3 — Verify

Once the Space shows **Running**:

```bash
curl https://<YOUR_USERNAME>-fie.hf.space/health
curl https://<YOUR_USERNAME>-fie.hf.space/ready
curl https://<YOUR_USERNAME>-fie.hf.space/health/deep | python3 -m json.tool
```

The line that matters in `/health/deep`:

```json
"detector": { "status": "ok", "mode": "full_pipeline" }
```

`"reduced_recall"` means the model artifacts did not bake in — check the build
log for the `download_models.py --strict` step.

Then open the Space in a browser and try the examples.

---

## Step 4 (optional) — Enable the dashboard endpoints

The demo needs no database. The dashboard endpoints do.

**Space → Settings → Variables and secrets → New secret:**

| Name | Value |
| --- | --- |
| `MONGODB_URI` | your Atlas connection string |
| `JWT_SECRET_KEY` | `python3 -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `ADMIN_EMAIL` | your email |
| `CORS_ALLOWED_ORIGINS` | `https://failure-intelligence-system.pages.dev` |

MongoDB Atlas has a permanent free M0 tier. Add `0.0.0.0/0` under Network Access
— Spaces do not have stable egress IPs.

Then point the dashboard at it, in **Cloudflare Pages → Settings → Environment
variables**:

```
VITE_API_URL = https://<YOUR_USERNAME>-fie.hf.space/api/v1
```

Redeploy. The dashboard's "Demo data" banner disappears once the API answers.

---

## Step 5 — Link it from the README

```markdown
[![Demo](https://img.shields.io/badge/🤗_Demo-Live-yellow)](https://huggingface.co/spaces/<YOUR_USERNAME>/fie)
```

---

## Updating

```bash
git checkout space
git merge main -m "sync from main"
git push space space:main
```

The Space rebuilds automatically.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Build fails on `download_models.py` | GitHub release unreachable | re-run the build; check `scripts/model_manifest.json` release tag exists |
| `detector: reduced_recall` | models not baked in | check the `--strict` step in the build log |
| Space shows "Configuration error" | `README.md` frontmatter missing at repo root | Step 2 — the Space README must be at the root |
| Port error / no response | `app_port` mismatch | frontmatter says `app_port: 7860`; the Dockerfile must listen there |
| Permission denied writing files | container running as root | the Dockerfile creates uid 1000 `user` — do not change that |
| Slow first request after idle | free Spaces sleep | expected; wakes in ~30 s |

---

## Cost

Zero. CPU Basic (2 vCPU, 16 GB RAM) is free indefinitely. FIE uses well under
1 GB now that torch is gone.
