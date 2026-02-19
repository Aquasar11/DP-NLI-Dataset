from typing import Set, Tuple

def compute_precision_recall(
    predicted: Set[Tuple[str, str]],
    actual: Set[Tuple[str, str]]
) -> Tuple[float, float]:

    # sort both sets to ensure consistent ordering
    predicted = set(tuple(sorted(pair)) for pair in predicted)
    actual = set(tuple(sorted(pair)) for pair in actual)

    print(f"Predicted pairs: {predicted}")
    print(f"Actual pairs: {actual}")


    tp = predicted & actual
    fp = predicted - actual
    fn = actual - predicted

    precision = len(tp) / (len(tp) + len(fp)) if predicted else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if actual else 0.0

    return precision, recall
