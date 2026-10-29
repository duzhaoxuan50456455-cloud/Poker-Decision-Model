"""PokerBench training data, answer-free inputs, validation, and grouped splits."""

import ast
import csv
import hashlib
import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

Kind = Literal["preflop", "postflop"]
DATA_REVISION = "7ac61f961c81a50fc0f667820b2fb0e432dfec0d"
TRAIN_FILES: dict[Kind, str] = {
    "preflop": "preflop_60k_train_set_game_scenario_information.csv",
    "postflop": "postflop_500k_train_set_game_scenario_information.csv",
}
SHA256: dict[Kind, str] = {
    "postflop": "a26f166b6cfbeb4cec5ab497016a1efaae058cd7a955ab0089603af2efb0c0b3",
    "preflop": "5786fe9b2e34b593cd68e9c426cfce1a0ca845d0c45a27e8471ef3ad38657e69",
}
SEATS = {"UTG", "HJ", "CO", "BTN", "SB", "BB"}


@dataclass(frozen=True)
class DecisionInputs:
    """The complete model input contract. No IDs, labels, or target-derived features."""

    context: str
    candidates: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.context.strip():
            raise ValueError("Empty context")
        if not self.candidates or any(not c.strip() for c in self.candidates):
            raise ValueError("Candidates must be nonempty strings")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("Duplicate normalized candidates")


@dataclass(frozen=True)
class DecisionExample:
    source_id: str
    street: str
    inputs: DecisionInputs
    target_index: int

    def __post_init__(self) -> None:
        if type(self.target_index) is not int or not 0 <= self.target_index < len(
            self.inputs.candidates
        ):
            raise ValueError("Target index out of range")

    @property
    def target_action(self) -> str:
        return self.inputs.candidates[self.target_index]

    def to_record(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "street": self.street,
            **asdict(self.inputs),
            "target_index": self.target_index,
        }


@dataclass(frozen=True)
class ParseFailure:
    source_id: str
    error: str


def normalize_action(value: object) -> str:
    """Normalize syntax and numeric spelling without discarding sizing or action type."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Empty or non-string action")
    action = " ".join(value.lower().split())
    if action in {"fold", "check", "call"}:
        return action
    if action in {"allin", "all-in", "all in"}:
        return "all-in"
    match = re.fullmatch(r"(?:(bet|raise)(?: to)? )?(\d+(?:\.\d+)?)\s*(bb)?", action)
    if match and (match[1] or match[3]):
        amount = Decimal(match[2])
        if amount.is_finite() and amount > 0:
            spelling = (
                format(amount, "f").rstrip("0").rstrip(".")
                if "." in match[2]
                else match[2].lstrip("0")
            )
            return f"{match[1] or 'raise'} {spelling}bb"
    raise ValueError(f"Unknown action: {value!r}")


def action_type(action: str) -> str:
    return normalize_action(action).split()[0]


def _text(row: Mapping[str, object], field: str, *, blank: bool = False) -> str:
    value = row[field]
    if not isinstance(value, str) or (not value.strip() and not blank):
        raise ValueError(f"Missing or invalid {field}")
    return " ".join(value.split())


def _number(row: Mapping[str, object], field: str, *, integer: bool = False) -> str:
    value = float(str(row[field]))
    if not math.isfinite(value) or value < 0 or (integer and not value.is_integer()):
        raise ValueError(f"Invalid {field}: {row[field]!r}")
    if field == "num_players" and not 1 <= value <= 6:
        raise ValueError("num_players outside 1..6")
    return f"{value:g}"


def _cards(value: str, count: int) -> list[str]:
    if not re.fullmatch(r"(?:[2-9TJQKA][cdhs]){" + str(count) + "}", value):
        raise ValueError(f"Invalid cards: {value!r}")
    cards = [value[i : i + 2] for i in range(0, len(value), 2)]
    if len(set(cards)) != len(cards):
        raise ValueError("Repeated card")
    return cards


def _preflop_history(value: str) -> str:
    tokens = value.split("/") if value else []
    if len(tokens) % 2:
        raise ValueError("Incomplete preflop action pair")
    result = []
    for seat, action in zip(tokens[::2], tokens[1::2], strict=True):
        if seat not in SEATS:
            raise ValueError(f"Invalid history seat: {seat}")
        result.append(f"{seat}: {normalize_action(action)}")
    return "; ".join(result) or "none"


def _postflop_history(row: Mapping[str, object]) -> tuple[str, list[str], str]:
    street = _text(row, "evaluation_at").lower()
    if street not in {"flop", "turn", "river"}:
        raise ValueError(f"Unknown street: {street}")
    board = _cards(_text(row, "board_flop"), 3)
    tokens = _text(row, "postflop_action", blank=True).split("/")
    if tokens == [""]:
        tokens = []
    history = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "dealcards":
            if len(board) >= 5 or i + 1 >= len(tokens):
                raise ValueError("Invalid dealcards marker")
            card = _cards(tokens[i + 1], 1)[0]
            column = "board_turn" if len(board) == 3 else "board_river"
            if card != _text(row, column):
                raise ValueError(f"Dealt card differs from {column}")
            board.append(card)
            history.append(f"{'turn' if len(board) == 4 else 'river'}: {card}")
            i += 2
            continue
        match = re.fullmatch(r"(OOP|IP)_(CHECK|CALL|FOLD|BET|RAISE)(?:_(\d+(?:\.\d+)?))?", token)
        if not match:
            raise ValueError(f"Unknown postflop action: {token!r}")
        side, verb, amount = match.groups()
        action = normalize_action(verb + (f" {amount}" if amount else ""))
        history.append(f"{side}: {action}")
        i += 1
    if len(board) != {"flop": 3, "turn": 4, "river": 5}[street]:
        raise ValueError("Street and dealt board length disagree")
    return street, board, "; ".join(history) or "none"


def build_inputs(row: Mapping[str, object], kind: Kind) -> DecisionInputs:
    """Allowlisted state fields only; this function works with the answer removed."""
    moves = row["available_moves"]
    if not isinstance(moves, str):
        raise ValueError("available_moves must be a serialized list")
    parsed = ast.literal_eval(moves)
    if not isinstance(parsed, list):
        raise ValueError("available_moves must be a list")
    candidates = tuple(normalize_action(move) for move in parsed)
    pot = _number(row, "pot_size")
    if kind == "preflop":
        position = _text(row, "hero_pos")
        if position not in SEATS:
            raise ValueError(f"Unknown hero position: {position}")
        cards = _cards(_text(row, "hero_holding"), 2)
        context = (
            f"Street: preflop. Hero position: {position}. Hero cards: {' '.join(cards)}. "
            f"Previous actions: {_preflop_history(_text(row, 'prev_line', blank=True))}. "
            f"Source num_players: {_number(row, 'num_players', integer=True)}. "
            f"Source num_bets: {_number(row, 'num_bets', integer=True)}. Pot: {pot} bb."
        )
    elif kind == "postflop":
        street, board, history = _postflop_history(row)
        position = _text(row, "hero_position")
        if position not in {"IP", "OOP"}:
            raise ValueError(f"Unknown postflop position: {position}")
        cards = _cards(_text(row, "holding"), 2)
        if len(set(cards + board)) != len(cards + board):
            raise ValueError("Repeated card between Hero and visible board")
        context = (
            f"Street: {street}. Hero position: {position}. Hero cards: {' '.join(cards)}. "
            f"Board: {' '.join(board)}. "
            f"Preflop actions: {_preflop_history(_text(row, 'preflop_action', blank=True))}. "
            f"Postflop actions: {history}. Pot: {pot} bb."
        )
    else:
        raise ValueError(f"Unsupported dataset kind: {kind}")
    return DecisionInputs(context, candidates)


def build_examples(
    rows: Iterable[Mapping[str, object]], kind: Kind
) -> tuple[list[DecisionExample], list[ParseFailure]]:
    """Account for every source row. Callers must stop on failures before training."""
    examples, failures = [], []
    for index, row in enumerate(rows):
        source_id = f"{kind}-train-{index}"
        try:
            inputs = build_inputs(row, kind)
            target = normalize_action(row["correct_decision"])
            if inputs.candidates.count(target) != 1:
                raise ValueError(f"Target {target!r} must match exactly one candidate")
            street = "preflop" if kind == "preflop" else _text(row, "evaluation_at").lower()
            examples.append(
                DecisionExample(source_id, street, inputs, inputs.candidates.index(target))
            )
        except (ValueError, TypeError, SyntaxError, KeyError, OverflowError) as error:
            failures.append(ParseFailure(source_id, str(error)))
    return examples, failures


def split_examples(
    examples: Sequence[DecisionExample], *, seed: int = 42, validation_fraction: float = 0.2
) -> tuple[list[DecisionExample], list[DecisionExample]]:
    """Shuffle sorted distinct contexts; keep every identical context on the same side."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if len({e.source_id for e in examples}) != len(examples):
        raise ValueError("Duplicate source IDs")
    contexts = sorted({e.inputs.context for e in examples})
    if len(contexts) < 2:
        raise ValueError("Need at least two distinct contexts")
    random.Random(seed).shuffle(contexts)
    n = max(1, min(len(contexts) - 1, round(len(contexts) * validation_fraction)))
    held_out = set(contexts[:n])
    train = [e for e in examples if e.inputs.context not in held_out]
    validation = [e for e in examples if e.inputs.context in held_out]
    return train, validation


def download_training(kind: Kind, directory: Path) -> Path:
    """Fetch only a named training CSV at a pinned revision; verify cached bytes too."""
    filename = TRAIN_FILES[kind]
    path = directory / filename
    expected = SHA256[kind]
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Cached checksum mismatch: {path}")
        return path
    url = f"https://huggingface.co/datasets/RZ412/PokerBench/resolve/{DATA_REVISION}/{filename}"
    with urlopen(url, timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Downloaded checksum mismatch: {filename}")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    """Preserve blanks: empty preflop history means no prior recorded action."""
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))
