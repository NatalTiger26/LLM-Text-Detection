"""Unit tests for tokenization and dataset preparation utilities."""

from typing import Any, Dict, List
import pytest
import torch

from model import tokenize_and_encode


class DummyTokenizer:
    """Mock fast tokenizer conforming to HuggingFace tokenizer interface."""

    def __init__(self, vocab_size: int = 1000) -> None:
        self.vocab_size = vocab_size

    def __call__(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: int = 128,
        padding: str = "max_length",
        truncation: bool = True,
        return_attention_mask: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(texts)
        input_ids = torch.ones((batch_size, max_length), dtype=torch.long)
        attention_mask = torch.ones((batch_size, max_length), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_tokenize_and_encode_with_labels() -> None:
    """Tests tokenization with binary labels."""
    tokenizer = DummyTokenizer()
    texts = ["Sample human text", "Sample machine generated text", "Another text"]
    labels = [0, 1, 0]

    input_ids, attention_masks, labels_tensor = tokenize_and_encode(
        tokenizer, texts, labels=labels, max_length=64, batch_size=2
    )

    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_masks, torch.Tensor)
    assert isinstance(labels_tensor, torch.Tensor)

    assert input_ids.shape == (3, 64)
    assert attention_masks.shape == (3, 64)
    assert labels_tensor.shape == (3,)
    assert labels_tensor.dtype == torch.long
    assert labels_tensor.tolist() == [0, 1, 0]


def test_tokenize_and_encode_without_labels() -> None:
    """Tests tokenization without labels for pure inference."""
    tokenizer = DummyTokenizer()
    texts = ["First document", "Second document"]

    result = tokenize_and_encode(tokenizer, texts, labels=None, max_length=32)
    assert isinstance(result, tuple)
    assert len(result) == 2
    input_ids, attention_masks = result
    assert input_ids.shape == (2, 32)
    assert attention_masks.shape == (2, 32)


def test_tokenize_and_encode_empty() -> None:
    """Tests tokenization on empty input lists."""
    tokenizer = DummyTokenizer()
    input_ids, attention_masks = tokenize_and_encode(tokenizer, [], labels=None, max_length=16)
    assert input_ids.shape == (0, 16)
    assert attention_masks.shape == (0, 16)

    res_with_labels = tokenize_and_encode(tokenizer, [], labels=[], max_length=16)
    assert len(res_with_labels) == 3
    assert res_with_labels[0].shape == (0, 16)
    assert res_with_labels[2].shape == (0,)
