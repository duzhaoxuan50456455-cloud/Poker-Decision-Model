"""CPU baselines over variable candidate sets; prediction accepts answer-free inputs."""

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from poker_decision.data import DecisionExample, DecisionInputs, action_type


def uniform(inputs: DecisionInputs) -> list[float]:
    return [1 / len(inputs.candidates)] * len(inputs.candidates)


class OptionPrior:
    """Smoothed training-only action-type prior, conditional on the unordered option types.

    Multiple amounts of one type share that type's probability mass. Unseen menus
    use uniform probabilities. This diagnoses how much the source menus reveal.
    """

    def __init__(self) -> None:
        self.counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

    @staticmethod
    def _key(inputs: DecisionInputs) -> tuple[str, ...]:
        return tuple(sorted(action_type(c) for c in inputs.candidates))

    def fit(self, examples: Sequence[DecisionExample]) -> "OptionPrior":
        if not examples:
            raise ValueError("Empty training set")
        self.counts.clear()
        for example in examples:
            self.counts[self._key(example.inputs)][action_type(example.target_action)] += 1
        return self

    def predict(self, inputs: DecisionInputs) -> list[float]:
        counts = self.counts.get(self._key(inputs))
        if counts is None:
            return uniform(inputs)
        multiplicity = Counter(action_type(c) for c in inputs.candidates)
        total = sum(counts.values()) + len(multiplicity)
        return [
            (counts[action_type(c)] + 1) / total / multiplicity[action_type(c)]
            for c in inputs.candidates
        ]


def pair_features(inputs: DecisionInputs, candidate: str) -> list[str]:
    """Explicit context/action interactions let a linear model condition its ranking.

    Simply appending the same context features to every candidate would cancel
    those context terms when ranking. Preserve card case and action amounts.
    """
    tokens = re.findall(r"[2-9TJQKA][cdhs]|[A-Za-z]+|\d+(?:\.\d+)?", inputs.context)
    terms = set(tokens)
    terms.update(f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False))
    kind = action_type(candidate)
    return (
        [f"action={candidate}", f"kind={kind}"]
        + [f"{kind}|{term}" for term in sorted(terms)]
        + [f"{candidate}|{term}" for term in sorted(terms)]
    )


class LinearCandidateScorer:
    """Hashed context/candidate interactions + binary logistic regression.

    Candidate logits are normalized within each decision. These probabilities
    are a baseline's scores, not calibrated solver mixing frequencies.
    """

    def __init__(self, *, seed: int = 42) -> None:
        try:
            from sklearn.feature_extraction import FeatureHasher
            from sklearn.linear_model import LogisticRegression
        except ImportError as error:
            raise ImportError("Install the analysis extra to train the linear baseline") from error
        self.hasher: Any = FeatureHasher(n_features=2**18, input_type="string")
        self.model: Any = LogisticRegression(max_iter=500, solver="liblinear", random_state=seed)
        self.fitted = False

    def fit(self, examples: Sequence[DecisionExample]) -> "LinearCandidateScorer":
        features, targets, weights = [], [], []
        for example in examples:
            n = len(example.inputs.candidates)
            # Forced decisions provide no ranking information.
            if n == 1:
                continue
            for j, candidate in enumerate(example.inputs.candidates):
                features.append(pair_features(example.inputs, candidate))
                correct = j == example.target_index
                targets.append(int(correct))
                weights.append(0.5 if correct else 0.5 / (n - 1))
        if not features:
            raise ValueError("Need training decisions with at least two candidates")
        self.model.fit(self.hasher.transform(features), targets, sample_weight=weights)
        self.fitted = True
        return self

    def predict(self, inputs: DecisionInputs) -> list[float]:
        if not self.fitted:
            raise RuntimeError("Fit the candidate scorer before prediction")
        if len(inputs.candidates) == 1:
            return [1.0]
        features = [pair_features(inputs, c) for c in inputs.candidates]
        scores = self.model.decision_function(self.hasher.transform(features)).tolist()
        maximum = max(scores)
        weights = [math.exp(score - maximum) for score in scores]
        total = sum(weights)
        return [weight / total for weight in weights]
