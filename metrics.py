from typing import Iterable, List, Tuple, Dict, Set

Edge = Tuple[str, str]


def _normalize_edges(edges: Iterable[Edge]) -> Set[Edge]:
    """Return a set of edges with sorted node tuples to ensure undirected equality."""
    normalized = set()
    for u, v in edges:
        normalized.add(tuple(sorted((u, v))))
    return normalized


def compute_coverage_metrics(predicted_edges: Iterable[Edge], ground_truth_edges: Iterable[Edge]) -> Dict[str, float]:
    """Compute coverage and redundancy metrics for graph quality assessment.

    Parameters
    ----------
    predicted_edges : iterable of tuple
        Edges predicted by the model.
    ground_truth_edges : iterable of tuple
        Gold-standard edges.
    """
    pred_edges = _normalize_edges(predicted_edges)
    gt_edges = _normalize_edges(ground_truth_edges)

    correct_edges = pred_edges & gt_edges
    num_correct = len(correct_edges)
    num_pred = len(pred_edges)
    num_gt = len(gt_edges)

    coverage = num_correct / num_gt if num_gt else 0.0
    redundancy = (num_pred - num_correct) / num_pred if num_pred else 0.0
    precision = num_correct / num_pred if num_pred else 0.0
    recall = coverage
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "coverage": coverage,
        "redundancy": redundancy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def compute_expression_level_metrics(predictions: List[Iterable[Edge]], ground_truth: List[Iterable[Edge]]) -> Dict[str, float]:
    """Expression-level evaluation across a dataset."""
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have the same length")

    precisions = []
    recalls = []
    f1s = []
    for pred, gt in zip(predictions, ground_truth):
        metrics = compute_coverage_metrics(pred, gt)
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1s.append(metrics["f1_score"])

    n = len(predictions)
    avg_precision = sum(precisions) / n if n else 0.0
    avg_recall = sum(recalls) / n if n else 0.0
    avg_f1 = sum(f1s) / n if n else 0.0

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": avg_f1,
    }


def compute_structure_accuracy(predicted_slt: List[Iterable[Edge]], ground_truth_slt: List[Iterable[Edge]]) -> float:
    """Compute structure recognition rate."""
    if len(predicted_slt) != len(ground_truth_slt):
        raise ValueError("predicted_slt and ground_truth_slt must have the same length")

    correct = 0
    total = len(predicted_slt)
    for pred, gt in zip(predicted_slt, ground_truth_slt):
        pred_edges = _normalize_edges(pred)
        gt_edges = _normalize_edges(gt)
        if pred_edges == gt_edges:
            correct += 1

    return correct / total if total else 0.0
