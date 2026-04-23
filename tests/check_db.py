"""
Quick script to check what data is available in MongoDB for classifier training.
Run: python check_db.py
"""
import os
import sys

# Load .env
from dotenv import load_dotenv
load_dotenv()

from pymongo import MongoClient
from pymongo.server_api import ServerApi

MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME   = os.getenv("MONGODB_DB_NAME", "failure_intelligence")

if not MONGO_URI:
    print("ERROR: MONGODB_URI not found in .env")
    sys.exit(1)

client = MongoClient(
    MONGO_URI,
    server_api=ServerApi("1"),
    serverSelectionTimeoutMS=10000,
    tls=True,
    tlsAllowInvalidCertificates=True,
)

db = client[DB_NAME]

print("=" * 55)
print("  FIE — MongoDB Data Check")
print("=" * 55)

# 1. All collections
collections = db.list_collection_names()
print(f"\nCollections found: {collections}\n")

# 2. Inferences collection
inf_col   = db["inferences"]
inf_count = inf_col.count_documents({})
print(f"inferences       : {inf_count} records")

# 3. Signal logs collection
sig_col   = db["signal_logs"]
sig_count = sig_col.count_documents({})
print(f"signal_logs      : {sig_count} records")

# 4. Ground truth cache
gt_col   = db["ground_truth_cache"]
gt_count = gt_col.count_documents({})
print(f"ground_truth_cache: {gt_count} records")

# 5. Feedback
fb_col   = db["feedback"]
fb_count = fb_col.count_documents({})
print(f"feedback         : {fb_count} records")

print()

# 6. Signal logs breakdown — how many failures vs non-failures
if sig_count > 0:
    failures     = sig_col.count_documents({"high_failure_risk": True})
    non_failures = sig_col.count_documents({"high_failure_risk": False})
    print(f"signal_logs breakdown:")
    print(f"  high_failure_risk=True  : {failures}")
    print(f"  high_failure_risk=False : {non_failures}")

    # Check FSV fields are present
    sample = sig_col.find_one({})
    if sample:
        fsv_fields = [
            "agreement_score", "entropy_score", "fsd_score",
            "ensemble_disagreement", "high_failure_risk"
        ]
        print(f"\n  FSV fields present in a sample record:")
        for f in fsv_fields:
            val = sample.get(f, "MISSING")
            print(f"    {f:30s} = {val}")

# 7. Inferences breakdown
if inf_count > 0:
    print(f"\ninferences breakdown:")
    flagged     = inf_col.count_documents({"failure_detected": True})
    not_flagged = inf_col.count_documents({"failure_detected": False})
    print(f"  failure_detected=True  : {flagged}")
    print(f"  failure_detected=False : {not_flagged}")

    # Model breakdown
    pipeline = [
        {"$group": {"_id": "$model_name", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    models = list(inf_col.aggregate(pipeline))
    if models:
        print(f"\n  Records by model:")
        for m in models:
            print(f"    {str(m['_id']):35s} : {m['count']}")

print()
print("=" * 55)

# 8. Verdict for classifier training
print("\nClassifier Training Readiness:")
if sig_count >= 500:
    print(f"  GOOD — {sig_count} signal_logs records available")
    print("  Enough data to train XGBoost classifier directly.")
elif sig_count >= 100:
    print(f"  PARTIAL — {sig_count} signal_logs records")
    print("  Will need synthetic data to supplement.")
else:
    print(f"  NOT ENOUGH — only {sig_count} signal_logs records")
    print("  Need to generate synthetic data first.")
    print("  Run: python data/synthetic_generator.py")

client.close()
