from fastapi import FastAPI, Request
from datetime import datetime, timezone
import json, os

app = FastAPI()

LOG_FILE = "pings.jsonl"


@app.post("/ping")
async def ping(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "os": body.get("os", "unknown"),
        "python": body.get("python", "unknown"),
        "version": body.get("version", "unknown"),
        "ip": request.headers.get("x-forwarded-for", request.client.host),
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return {"ok": True}


@app.get("/stats")
async def stats():
    if not os.path.exists(LOG_FILE):
        return {"total": 0, "entries": []}

    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    os_counts = {}
    for e in entries:
        os_counts[e["os"]] = os_counts.get(e["os"], 0) + 1

    return {"total": len(entries), "by_os": os_counts, "recent": entries[-10:]}


@app.get("/")
async def root():
    return {"status": "fie telemetry server running"}
