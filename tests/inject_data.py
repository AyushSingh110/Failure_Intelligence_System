import requests
import sys
from datetime import datetime, timedelta
from random import Random

BASE    = "http://127.0.0.1:8000/api/v1"
TRACK   = f"{BASE}/track"
HEADERS = {"Content-Type": "application/json"}

# Model profiles 
MODELS = {
    "gpt-4": {
        "version":       "turbo-2024-04",
        "base_latency":  380,
        "latency_jitter": 120,
        "base_entropy":  0.15,   
        "base_agree":    0.88,   # High agreement
        "spike_prob":    0.15,   # Occasional spikes
        "spike_entropy": 0.82,
        "spike_agree":   0.30,
        "questions":     [
            "What is the capital of France?",
            "Explain quantum entanglement in one sentence.",
            "Who wrote Hamlet?",
            "What is the boiling point of water at sea level?",
            "Summarise the French Revolution in 3 words.",
        ],
        "answers": ["Paris", "Quantum entanglement is...", "Shakespeare", "100°C", "Liberty, Equality, Fraternity"],
    },
    "gpt-3.5-turbo": {
        "version":        "0125",
        "base_latency":   210,
        "latency_jitter": 80,
        "base_entropy":   0.30,   # More variable
        "base_agree":     0.72,
        "spike_prob":     0.28,   # Spikes more often
        "spike_entropy":  0.90,
        "spike_agree":    0.22,
        "questions":     [
            "What year did WW2 end?",
            "What is photosynthesis?",
            "Name the largest planet in our solar system.",
            "Who invented the telephone?",
            "What is the speed of light?",
        ],
        "answers": ["1945", "Photosynthesis is...", "Jupiter", "Alexander Graham Bell", "299,792,458 m/s"],
    },
    "claude-3-sonnet": {
        "version":        "20240229",
        "base_latency":   520,
        "latency_jitter": 200,
        "base_entropy":   0.12,   # Very consistent
        "base_agree":     0.91,
        "spike_prob":     0.10,   # Rarely spikes
        "spike_entropy":  0.75,
        "spike_agree":    0.40,
        "questions":     [
            "What is the mitochondria?",
            "Who painted the Mona Lisa?",
            "What is the Pythagorean theorem?",
            "Name the currency of Japan.",
            "What does DNA stand for?",
        ],
        "answers": ["The powerhouse of the cell", "Leonardo da Vinci", "a²+b²=c²", "Japanese Yen", "Deoxyribonucleic acid"],
    },
    "gemini-pro": {
        "version":        "1.5-pro",
        "base_latency":   290,
        "latency_jitter": 150,
        "base_entropy":   0.38,   # Most variable — newer model still calibrating
        "base_agree":     0.65,
        "spike_prob":     0.32,   # Highest spike probability
        "spike_entropy":  0.95,
        "spike_agree":    0.18,
        "questions":     [
            "What is the tallest mountain on Earth?",
            "Who discovered penicillin?",
            "What is Newton's first law?",
            "How many continents are there?",
            "What is the chemical symbol for gold?",
        ],
        "answers": ["Mount Everest", "Alexander Fleming", "An object in motion stays in motion", "7", "Au"],
    },
}


def entropy_multiplier(hour: int) -> float:
    if   9  <= hour < 12: return 1.0    # Normal
    elif 12 <= hour < 14: return 1.4    # Load spike
    elif 14 <= hour < 17: return 1.9    # Peak degradation
    elif 17 <= hour < 19: return 1.3    # Recovery starts
    else:                 return 0.8    # Evening stable


def latency_multiplier(hour: int) -> float:
    if   12 <= hour < 14: return 2.2
    elif 14 <= hour < 16: return 3.5
    elif 16 <= hour < 18: return 1.8
    else:                 return 1.0


def build_record(rng: Random, idx: int, model_name: str, ts: datetime) -> dict:
    m       = MODELS[model_name]
    hour    = ts.hour
    e_mult  = entropy_multiplier(hour)
    l_mult  = latency_multiplier(hour)

    # Is this a spike event?
    is_spike = rng.random() < m["spike_prob"] * e_mult

    entropy  = min(1.0, (m["spike_entropy"] if is_spike else m["base_entropy"]) * e_mult + rng.uniform(-0.05, 0.05))
    agree    = max(0.0, (m["spike_agree"] if is_spike else m["base_agree"]) - (e_mult - 1) * 0.2 + rng.uniform(-0.05, 0.05))
    fsd      = max(0.0, min(1.0, agree - 0.1 + rng.uniform(-0.05, 0.05)))
    latency  = max(50, m["base_latency"] * l_mult + rng.gauss(0, m["latency_jitter"]))
    temp     = rng.choice([0.1, 0.3, 0.7, 0.9])
    q_idx    = idx % len(m["questions"])
    is_correct = None if is_spike else (rng.random() > 0.08)

    return {
        "request_id":    f"{model_name.replace('-','').replace('.','')}-{idx:05d}",
        "timestamp":     ts.isoformat(),
        "model_name":    model_name,
        "model_version": m["version"],
        "temperature":   temp,
        "latency_ms":    round(latency, 1),
        "input_text":    m["questions"][q_idx],
        "output_text":   m["answers"][q_idx],
        "ground_truth":  m["answers"][q_idx] if rng.random() > 0.3 else None,
        "is_correct":    is_correct,
        "metrics": {
            "entropy":        round(min(1.0, max(0.0, entropy)), 4),
            "agreement_score": round(min(1.0, max(0.0, agree)),   4),
            "fsd_score":      round(min(1.0, max(0.0, fsd)),      4),
        }
    }


def main():
    rng = Random(42)  

    # Check connection
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=3)
        if r.status_code != 200:
            raise ConnectionError
    except Exception:
        print("ERROR: Backend not running. Start with:")
        print("  uvicorn app.main:app --reload")
        sys.exit(1)

    print("Connected to backend. Injecting records...\n")

    # Build schedule: 40 records per model = 160 total
    # Spread across 09:00 – 22:00 on 2024-01-15
    base_date  = datetime(2024, 1, 15, 9, 0, 0)
    total_hours = 13  # 09:00 → 22:00
    records_per_model = 40

    all_records = []
    for model_name in MODELS:
        for i in range(records_per_model):
            # Spread evenly across the day
            minutes_offset = int((i / records_per_model) * total_hours * 60)
            ts = base_date + timedelta(minutes=minutes_offset)
            all_records.append(build_record(rng, i + 1, model_name, ts))

    # Sort by timestamp so vault shows chronological order
    all_records.sort(key=lambda r: r["timestamp"])

    ok = err = 0
    for rec in all_records:
        try:
            resp = requests.post(TRACK, json=rec, headers=HEADERS, timeout=8)
            if resp.status_code == 200:
                ok += 1
                model_short = rec["model_name"][:12].ljust(12)
                e = rec["metrics"]["entropy"]
                a = rec["metrics"]["agreement_score"]
                flag = " ⚠ HIGH RISK" if e > 0.75 or a < 0.50 else ""
                print(f"  [{ok:03d}] {model_short} | E={e:.3f} A={a:.3f} | {rec['timestamp'][11:16]}{flag}")
            else:
                err += 1
                print(f"  [ERR] {rec['request_id']} → HTTP {resp.status_code}: {resp.text[:80]}")
        except Exception as ex:
            err += 1
            print(f"  [ERR] {rec['request_id']} → {ex}")

    print(f"\n{'─'*55}")
    print(f"  Injected: {ok} records across {len(MODELS)} models")
    print(f"  Failed:   {err} records")
    print(f"\nModels injected:")
    for name, m in MODELS.items():
        count = sum(1 for r in all_records if r["model_name"] == name)
        highrisk = sum(1 for r in all_records
                       if r["model_name"] == name
                       and (r["metrics"]["entropy"] > 0.75 or r["metrics"]["agreement_score"] < 0.50))
        print(f"  {name:<20} {count} records  |  {highrisk} high-risk  ({100*highrisk//count}%)")
    print(f"\nOpen http://localhost:8501 to see the dashboard.")


if __name__ == "__main__":
    main()