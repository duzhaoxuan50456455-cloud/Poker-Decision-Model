"""Offline neural contract checks with a tiny randomly initialized BERT."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from poker_decision.data import DecisionExample, DecisionInputs  # noqa: E402
from poker_decision.models.encoder import (  # noqa: E402
    PokerDecisionModel,
    collate_decisions,
    encode_inputs,
    model_input,
)
from poker_decision.training import (  # noqa: E402
    encode_examples,
    load_checkpoint,
    load_prepared,
    predict_encoded,
    save_checkpoint,
    seed_everything,
    train_epoch,
)


@pytest.fixture
def tokenizer(tmp_path):
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "state",
        ":",
        "question",
        "decisions",
        "what",
        "should",
        "hero",
        "do",
        "?",
        "strong",
        "weak",
        "call",
        "fold",
        "raise",
        "12",
        "##bb",
        "check",
    ]
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocabulary) + "\n")
    return transformers.BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)


def make_model(tokenizer):
    config = transformers.BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=128,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )
    return PokerDecisionModel(transformers.BertModel(config))


def rows():
    return [
        DecisionExample("a", "preflop", DecisionInputs("strong", ("call", "fold")), 0),
        DecisionExample("b", "flop", DecisionInputs("weak", ("fold", "call", "raise 12bb")), 0),
        DecisionExample("c", "turn", DecisionInputs("weak", ("check",)), 0),
    ]


def test_answer_free_serialization_and_complete_spans(tokenizer):
    e = rows()[1]
    assert set(model_input(e.inputs)) == {"state", "question", "decisions"}
    encoded = encode_inputs(e.inputs, tokenizer, max_length=128)
    changed_target = DecisionExample(e.source_id, e.street, e.inputs, 1)
    assert encode_inputs(changed_target.inputs, tokenizer) == encoded
    for action, (start, end) in zip(e.inputs.candidates, encoded.candidate_spans, strict=True):
        assert list(encoded.input_ids[start:end]) == tokenizer.encode(
            action, add_special_tokens=False
        )
    with pytest.raises(ValueError, match="No truncation"):
        encode_inputs(e.inputs, tokenizer, max_length=8)
    with pytest.raises(ValueError, match="tokenization failures"):
        encode_examples([e], tokenizer, max_length=8)


def test_variable_candidates_padding_gradients_and_shared_head(tokenizer):
    seed_everything(42)
    model = make_model(tokenizer)
    examples = rows()
    encoded = encode_examples(examples, tokenizer, max_length=128)
    batch = collate_decisions(encoded, pad_token_id=tokenizer.pad_token_id)
    logits = model(**batch)
    assert logits.shape == (3, 3)
    probs = logits.softmax(-1)
    assert probs[0, 2].item() == 0
    assert probs[2].tolist() == [1, 0, 0]
    torch.testing.assert_close(probs.sum(-1), torch.ones(3))
    alone = model(**collate_decisions([encoded[0]], pad_token_id=tokenizer.pad_token_id))
    torch.testing.assert_close(logits[0, :2], alone[0])
    # Reordering extraction slots must apply the same head to each representation.
    swapped = dict(batch)
    swapped["candidate_token_mask"] = batch["candidate_token_mask"][:, [1, 0, 2]]
    torch.testing.assert_close(model(**swapped), logits[:, [1, 0, 2]])
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 0, 0]))
    loss.backward()
    assert model.head.weight.grad.abs().sum() > 0
    gradient = model.encoder.embeddings.word_embeddings.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    batch["candidate_token_mask"][0, 0, -1] = True
    with pytest.raises(ValueError, match="padding"):
        model(**batch)


def test_learning_and_local_checkpoint_roundtrip(tokenizer, tmp_path):
    torch.set_num_threads(1)
    seed_everything(42)
    model = make_model(tokenizer)
    examples = rows()[:2]
    encoded = encode_examples(examples, tokenizer, max_length=128)
    batch = collate_decisions(encoded, pad_token_id=tokenizer.pad_token_id)
    targets = torch.tensor([0, 0])
    initial = torch.nn.functional.cross_entropy(model(**batch), targets).item()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for step in range(20):
        train_epoch(model, encoded, [0, 0], pad_token_id=0, optimizer=optimizer, seed=step)
    model.eval()
    final = torch.nn.functional.cross_entropy(model(**batch), targets).item()
    assert final < initial * 0.5
    before = predict_encoded(model, encoded, pad_token_id=0)
    path = tmp_path / "checkpoint"
    save_checkpoint(path, model, tokenizer, {"question": "What should Hero do?", "seed": 42})
    restored_tokenizer, restored_model, metadata = load_checkpoint(path)
    assert metadata["seed"] == 42
    assert encode_examples(examples, restored_tokenizer, max_length=128) == encoded
    after = predict_encoded(restored_model, encoded, pad_token_id=0)
    torch.testing.assert_close(torch.tensor(before[0]), torch.tensor(after[0]))
    torch.testing.assert_close(torch.tensor(before[1]), torch.tensor(after[1]))
    with pytest.raises(ValueError, match="Target outside"):
        train_epoch(model, encoded, [2, 0], pad_token_id=0, optimizer=optimizer)


def test_saved_membership_and_provenance(tmp_path, monkeypatch):
    from test_pipeline import source

    from poker_decision.pipeline import prepare

    source(tmp_path, monkeypatch)
    expected = prepare(tmp_path, kinds=("preflop",), sample_per_kind=None)
    assert load_prepared(tmp_path) == expected
    path = tmp_path / "data/processed/pokerbench_meeting1/train.jsonl"
    data = path.read_text().splitlines()
    data.reverse()
    path.write_text("\n".join(data) + "\n")
    with pytest.raises(ValueError, match="membership"):
        load_prepared(tmp_path)
    path.write_text("\n".join(reversed(data)) + "\n")
    manifest_path = path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["official_test_used"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="training-only"):
        load_prepared(tmp_path)


def test_notebook_is_cleared_and_valid():
    path = Path(__file__).parents[1] / "notebooks/02_pokerbench_decision_model.ipynb"
    nb = json.loads(path.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), str(path), "exec")
