"""
Batch analysis — run FIE over a CSV of prompt/output pairs.
Useful for auditing existing model outputs offline.
"""
import os
import csv
from fie_sdk import FIEClient

fie = FIEClient(api_key=os.getenv("FIE_API_KEY", "fie-your-key"))


def analyze_csv(input_path: str, output_path: str) -> None:
    """
    Reads a CSV with columns: prompt, output1, output2, output3
    Writes results to output_path with archetype and risk columns added.
    """
    results = []

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt  = row.get("prompt", "")
            outputs = [row[k] for k in ("output1", "output2", "output3") if row.get(k)]

            if not outputs:
                continue

            try:
                r = fie.monitor(prompt=prompt, model_outputs=outputs)
                results.append({
                    **row,
                    "archetype":   r.archetype,
                    "high_risk":   r.high_failure_risk,
                    "entropy":     f"{r.failure_signal_vector.entropy_score:.3f}",
                    "agreement":   f"{r.failure_signal_vector.agreement_score:.3f}",
                    "summary":     r.failure_summary,
                })
                print(f"[{r.archetype}] {prompt[:60]}")
            except Exception as e:
                print(f"Error on row: {e}")
                results.append({**row, "archetype": "ERROR", "high_risk": "", "entropy": "", "agreement": "", "summary": str(e)})

    if results:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {output_path} ({len(results)} rows)")


if __name__ == "__main__":
    analyze_csv("prompts.csv", "fie_analysis_results.csv")
