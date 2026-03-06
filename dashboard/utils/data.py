import pandas as pd


def _safe_metrics(record: dict) -> dict:
    """
    Safely extracts the metrics dict from a record.
    Handles: None, missing key, nested Pydantic object serialized as dict.
    """
    m = record.get("metrics")
    if m is None:
        return {}
    # Pydantic may serialize to a dict or the field may already be a dict
    if isinstance(m, dict):
        return m
    # Fallback: try to convert unknown object
    try:
        return dict(m)
    except Exception:
        return {}


def build_inference_dataframe(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    rows = []
    for r in records:
        m = _safe_metrics(r)
        rows.append({
            "timestamp":   r.get("timestamp", ""),
            "request_id":  r.get("request_id", ""),
            "model":       r.get("model_name", "—"),
            "version":     r.get("model_version", "—"),
            # metrics fields — use both key variants for safety
            "entropy":     m.get("entropy") or m.get("entropy_score"),
            "agreement":   m.get("agreement_score"),
            "fsd":         m.get("fsd_score"),
            "latency_ms":  r.get("latency_ms"),
            "temperature": r.get("temperature"),
            "is_correct":  r.get("is_correct"),
        })

    df = pd.DataFrame(rows)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def compute_kpi_summary(df: pd.DataFrame, records: list[dict]) -> dict:
    total         = len(records)
    avg_entropy   = df["entropy"].mean()    if not df.empty and "entropy"    in df.columns else None
    avg_agreement = df["agreement"].mean()  if not df.empty and "agreement"  in df.columns else None
    avg_latency   = df["latency_ms"].mean() if not df.empty and "latency_ms" in df.columns else None

    # pd.mean() returns NaN for all-None columns — treat NaN as None
    import math
    if avg_entropy   is not None and (isinstance(avg_entropy,   float) and math.isnan(avg_entropy)):
        avg_entropy = None
    if avg_agreement is not None and (isinstance(avg_agreement, float) and math.isnan(avg_agreement)):
        avg_agreement = None
    if avg_latency   is not None and (isinstance(avg_latency,   float) and math.isnan(avg_latency)):
        avg_latency = None

    high_risk_count = sum(
        1 for r in records
        if _safe_metrics(r).get("entropy") is not None
        and _safe_metrics(r).get("entropy") > 0.75
    )

    return {
        "total":           total,
        "avg_entropy":     round(avg_entropy,   4) if avg_entropy   is not None else None,
        "avg_agreement":   round(avg_agreement, 4) if avg_agreement is not None else None,
        "avg_latency":     round(avg_latency,   1) if avg_latency   is not None else None,
        "high_risk_count": high_risk_count,
        "risk_pct":        round(high_risk_count / total * 100, 1) if total else 0.0,
    }




