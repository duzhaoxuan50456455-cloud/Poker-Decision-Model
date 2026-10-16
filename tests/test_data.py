"""Offline regression tests for parsing, answer leakage, and grouped splitting."""

from dataclasses import replace
from pathlib import Path

import pytest

from poker_decision.data import (
    DecisionExample,
    DecisionInputs,
    build_examples,
    build_inputs,
    download_training,
    normalize_action,
    split_examples,
)


@pytest.fixture
def preflop():
    return {
        "prev_line": "",
        "hero_pos": "UTG",
        "hero_holding": "AhKd",
        "correct_decision": "2.0bb",
        "num_players": "6",
        "num_bets": "0",
        "available_moves": "['fold', '2.0bb', '3bb']",
        "pot_size": "1.5",
    }


@pytest.fixture
def postflop():
    return {
        "preflop_action": "HJ/2.0bb/BB/call",
        "board_flop": "JcJh4s",
        "board_turn": "4d",
        "board_river": "As",
        "aggressor_position": "OOP",
        "postflop_action": "OOP_CHECK",
        "evaluation_at": "Flop",
        "available_moves": "['Check', 'Bet 17']",
        "pot_size": "21",
        "hero_position": "IP",
        "holding": "AhKd",
        "correct_decision": "Bet 17",
    }


def test_exact_sizing_and_order(preflop):
    examples, failures = build_examples([preflop], "preflop")
    assert not failures
    assert examples[0].inputs.candidates == ("fold", "raise 2bb", "raise 3bb")
    assert examples[0].target_index == 1
    assert "Previous actions: none" in examples[0].inputs.context


@pytest.mark.parametrize("kind", ["preflop", "postflop"])
def test_inputs_do_not_need_labels(kind, preflop, postflop):
    row = preflop if kind == "preflop" else postflop
    original = build_inputs(row, kind)
    row.pop("correct_decision")
    row["answer"] = "INJECTED_ANSWER"
    row["future_result"] = "INJECTED_RESULT"
    assert build_inputs(row, kind) == original
    assert set(original.__dataclass_fields__) == {"context", "candidates"}
    assert "INJECTED" not in original.context


@pytest.mark.parametrize(
    "field,value",
    [
        ("available_moves", "[]"),
        ("available_moves", "['fold', 'FOLD']"),
        ("available_moves", "['2bb', '2.0bb']"),
        ("available_moves", "{'fold': 1}"),
        ("available_moves", "__import__('os').getcwd()"),
        ("available_moves", "[42]"),
        ("available_moves", "['']"),
        ("correct_decision", "call"),
        ("hero_holding", "AhAh"),
        ("hero_holding", "bad"),
        ("hero_pos", "unknown"),
        ("num_players", "nan"),
        ("num_players", "0"),
        ("num_players", "2.5"),
        ("num_bets", "-1"),
        ("pot_size", "inf"),
        ("pot_size", "-3"),
        ("prev_line", "UTG/2bb/BTN"),
        ("prev_line", None),
    ],
)
def test_failures_are_accounted_for(preflop, field, value):
    broken = preflop | {field: value}
    valid, failed = build_examples([preflop, broken], "preflop")
    assert len(valid) == len(failed) == 1
    assert failed[0].source_id == "preflop-train-1"
    assert failed[0].error


def test_missing_column_reports_failure(preflop):
    del preflop["hero_holding"]
    assert len(build_examples([preflop], "preflop")[1]) == 1


@pytest.mark.parametrize(
    "street,history,board",
    [
        ("Flop", "OOP_CHECK", "Jc Jh 4s"),
        ("Turn", "OOP_CHECK/IP_CHECK/dealcards/4d/OOP_CHECK", "Jc Jh 4s 4d"),
        (
            "River",
            "OOP_CHECK/IP_CHECK/dealcards/4d/OOP_CHECK/IP_CHECK/dealcards/As",
            "Jc Jh 4s 4d As",
        ),
    ],
)
def test_visible_board_only(postflop, street, history, board):
    postflop.update(evaluation_at=street, postflop_action=history)
    inputs = build_inputs(postflop, "postflop")
    assert f"Board: {board}." in inputs.context
    if street == "Flop":
        postflop.update(board_turn="INVALID_FUTURE", board_river="INVALID_FUTURE")
        assert build_inputs(postflop, "postflop") == inputs
    if street == "Turn":
        postflop["board_river"] = "INVALID_FUTURE"
        assert build_inputs(postflop, "postflop") == inputs


@pytest.mark.parametrize(
    "changes",
    [
        {"evaluation_at": "Turn"},
        {"holding": "JcKd"},
        {"postflop_action": "OOP_CHECK/dealcards"},
        {"evaluation_at": "Turn", "postflop_action": "dealcards/2s"},
        {"postflop_action": "OOP_UNKNOWN"},
        {"postflop_action": "IP_BET"},
    ],
)
def test_bad_postflop_is_reported(postflop, changes):
    valid, failures = build_examples([postflop | changes], "postflop")
    assert not valid
    assert len(failures) == 1


def test_split_groups_context_even_with_different_candidates():
    rows = [
        DecisionExample(str(i), "preflop", DecisionInputs(f"state {i}", ("fold", "call")), 0)
        for i in range(20)
    ]
    rows.append(
        replace(rows[0], source_id="duplicate", inputs=DecisionInputs("state 0", ("fold",)))
    )
    train, validation = split_examples(rows)
    assert {e.inputs.context for e in train}.isdisjoint(e.inputs.context for e in validation)
    assert len(train) + len(validation) == len(rows)
    assert split_examples(rows) == (train, validation)
    reversed_train, reversed_val = split_examples(list(reversed(rows)))
    assert {e.source_id for e in train} == {e.source_id for e in reversed_train}
    assert {e.source_id for e in validation} == {e.source_id for e in reversed_val}
    assert split_examples(rows, seed=1) != (train, validation)


@pytest.mark.parametrize("fraction", [0, 1, -0.1])
def test_bad_split_fraction(fraction):
    with pytest.raises(ValueError):
        split_examples([], validation_fraction=fraction)


def test_single_context_cannot_split():
    with pytest.raises(ValueError, match="two distinct"):
        split_examples([DecisionExample("0", "flop", DecisionInputs("a", ("check",)), 0)])


def test_checksum_rejects_corrupt_cache(tmp_path):
    path = tmp_path / "preflop_60k_train_set_game_scenario_information.csv"
    path.write_text("corrupt")
    with pytest.raises(ValueError, match="Cached checksum"):
        download_training("preflop", tmp_path)


def test_download_checks_before_saving(monkeypatch, tmp_path):
    import io

    monkeypatch.setattr("poker_decision.data.urlopen", lambda *a, **kw: io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="Downloaded checksum"):
        download_training("preflop", tmp_path)
    assert not list(Path(tmp_path).iterdir())


@pytest.mark.parametrize("action", ["raise", "bet", "0bb", "Bet -3", "Call 5", "nan"])
def test_invalid_action(action):
    with pytest.raises(ValueError):
        normalize_action(action)


def test_action_normalization_does_not_round_sizes():
    assert normalize_action("raise 12.34567890") == "raise 12.3456789bb"
    assert normalize_action("100.00bb") == "raise 100bb"
    assert normalize_action("0002bb") == "raise 2bb"
