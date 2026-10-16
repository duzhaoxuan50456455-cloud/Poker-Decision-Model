"""Exact-candidate evaluation and reproducible mistake inspection."""

import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence

from poker_decision.data import DecisionExample, DecisionInputs, action_type

Predictor = Callable[[DecisionInputs], list[float]]


def probabilities(inputs: DecisionInputs, predict: Predictor) -> list[float]:
    values = predict(inputs)
    if len(values) != len(inputs.candidates) or any(
        not math.isfinite(p) or p < 0 or p > 1 for p in values
    ):
        raise ValueError("Invalid candidate probabilities")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-8):
        raise ValueError("Candidate probabilities must sum to one")
    return values


def predicted_index(inputs: DecisionInputs, values: Sequence[float]) -> int:
    """Break exact ties lexically by action, independent of candidate ordering."""
    return min(range(len(values)), key=lambda i: (-values[i], inputs.candidates[i]))


def evaluate(examples: Sequence[DecisionExample], predict: Predictor) -> dict[str, float]:
    if not examples:
        raise ValueError("Empty evaluation set")
    hits, expected_hits, losses, briers, multi_hits = [], [], [], [], []
    truth, predictions = [], []
    for e in examples:
        values = probabilities(e.inputs, predict)
        index = predicted_index(e.inputs, values)
        hit = int(index == e.target_index)
        hits.append(hit)
        ties = [i for i, p in enumerate(values) if p == max(values)]
        expected_hits.append(int(e.target_index in ties) / len(ties))
        losses.append(-math.log(max(values[e.target_index], 1e-15)))
        briers.append(sum((p - int(i == e.target_index)) ** 2 for i, p in enumerate(values)))
        if len(values) > 1:
            multi_hits.append(hit)
        truth.append(action_type(e.target_action))
        predictions.append(action_type(e.inputs.candidates[index]))
    labels = sorted(set(truth) | set(predictions))
    f1s = []
    for label in labels:
        tp = sum(t == p == label for t, p in zip(truth, predictions, strict=True))
        denominator = truth.count(label) + predictions.count(label)
        f1s.append(2 * tp / denominator if denominator else 0.0)
    metrics = {
        "n": float(len(examples)),
        "exact_accuracy": sum(hits) / len(hits),
        "tie_expected_accuracy": sum(expected_hits) / len(expected_hits),
        "action_type_macro_f1": sum(f1s) / len(f1s),
        "log_loss": sum(losses) / len(losses),
        "brier_score": sum(briers) / len(briers),
        "single_option_rate": 1 - len(multi_hits) / len(examples),
    }
    if multi_hits:
        metrics["multi_option_exact_accuracy"] = sum(multi_hits) / len(multi_hits)
    return metrics


def evaluate_by_street(
    examples: Sequence[DecisionExample], predict: Predictor
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[DecisionExample]] = defaultdict(list)
    for e in examples:
        groups[e.street].append(e)
    return {"overall": evaluate(examples, predict)} | {
        street: evaluate(rows, predict) for street, rows in sorted(groups.items())
    }


def inspect_mistakes(
    examples: Sequence[DecisionExample], predict: Predictor, *, seed: int = 42, limit: int = 5
) -> list[dict[str, object]]:
    if limit < 0:
        raise ValueError("Mistake limit must be nonnegative")
    mistakes = []
    for e in examples:
        values = probabilities(e.inputs, predict)
        index = predicted_index(e.inputs, values)
        if index != e.target_index:
            mistakes.append(
                e.to_record()
                | {
                    "target_action": e.target_action,
                    "predicted_action": e.inputs.candidates[index],
                    "probabilities": values,
                }
            )
    return random.Random(seed).sample(mistakes, min(limit, len(mistakes)))
