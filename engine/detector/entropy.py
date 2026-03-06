import math
from collections import Counter


def compute_entropy(model_outputs: list[str]) -> float:
    if not model_outputs or len(model_outputs) == 1:
        return 0.0

    normalized_outputs = [o.strip().lower() for o in model_outputs]
    total_samples = len(normalized_outputs)
    answer_counts = Counter(normalized_outputs)

    raw_entropy = 0.0
    for count in answer_counts.values():
        probability = count / total_samples
        raw_entropy -= probability * math.log2(probability)

    max_entropy = math.log2(total_samples)
    normalized_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0

    return round(normalized_entropy, 4)
