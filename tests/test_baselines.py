"""Candidate scoring, probability metrics, and optional learned baseline."""

import math

import pytest

from poker_decision.baselines import LinearCandidateScorer, OptionPrior, uniform
from poker_decision.data import DecisionExample, DecisionInputs
from poker_decision.evaluation import evaluate, inspect_mistakes, probabilities


def example(context="a", candidates=("call", "fold"), target=0, source_id="a"):
    return DecisionExample(source_id, "preflop", DecisionInputs(context, candidates), target)


def test_prior_reorders_with_candidates_and_splits_type_mass():
    e = example(candidates=("fold", "raise 2bb", "raise 3bb"), target=1)
    model = OptionPrior().fit([e, e, e])
    p = model.predict(e.inputs)
    assert sum(p) == pytest.approx(1)
    assert p[1] == p[2]
    assert p[1] > p[0]
    reordered = DecisionInputs("a", tuple(reversed(e.inputs.candidates)))
    assert model.predict(reordered) == list(reversed(p))
    assert model.predict(DecisionInputs("new", ("check", "bet 4bb"))) == [0.5, 0.5]


def test_known_metrics_and_tie_semantics():
    e = example()
    metrics = evaluate([e], lambda inputs: [0.75, 0.25])
    assert metrics["exact_accuracy"] == 1
    assert metrics["log_loss"] == pytest.approx(-math.log(0.75))
    assert metrics["brier_score"] == pytest.approx(0.125)
    ties = evaluate([e], uniform)
    assert ties["tie_expected_accuracy"] == 0.5
    assert ties["exact_accuracy"] == 1  # lexical tie break picks call
    forced = evaluate([example(candidates=("fold",))], uniform)
    assert forced["exact_accuracy"] == forced["single_option_rate"] == 1
    assert "multi_option_exact_accuracy" not in forced


@pytest.mark.parametrize("values", [[1], [0, 0], [-1, 2], [float("nan"), 1], [float("inf"), 0]])
def test_invalid_probabilities(values):
    with pytest.raises(ValueError):
        probabilities(example().inputs, lambda inputs: values)


def test_five_or_fewer_mistakes_reproducibly():
    rows = [example(source_id=str(i)) for i in range(10)]
    predict = lambda inputs: [0.1, 0.9]  # noqa: E731
    mistakes = inspect_mistakes(rows, predict)
    assert len(mistakes) == 5
    assert mistakes == inspect_mistakes(rows, predict)
    assert len(inspect_mistakes(rows[:2], predict)) == 2
    assert inspect_mistakes(rows, lambda inputs: [1, 0]) == []


def test_learned_scorer_uses_context_and_handles_unseen_sizes():
    pytest.importorskip("sklearn")
    rows = [example("Hero strong AA", target=0), example("Hero weak 72", target=1)] * 30
    model = LinearCandidateScorer().fit(rows)
    strong = model.predict(rows[0].inputs)
    weak = model.predict(rows[1].inputs)
    assert strong[0] > strong[1]
    assert weak[1] > weak[0]
    reordered = DecisionInputs("Hero strong AA", ("fold", "call"))
    assert model.predict(reordered) == pytest.approx(list(reversed(strong)))
    unseen = DecisionInputs("Unseen context", ("fold", "raise 987bb", "raise 42bb"))
    assert sum(model.predict(unseen)) == pytest.approx(1)
    assert model.predict(DecisionInputs("forced", ("check",))) == [1.0]


def test_linear_rejects_only_forced_training():
    pytest.importorskip("sklearn")
    model = LinearCandidateScorer()
    with pytest.raises(RuntimeError):
        model.predict(example().inputs)
    with pytest.raises(ValueError, match="at least two"):
        model.fit([example(candidates=("fold",))])
