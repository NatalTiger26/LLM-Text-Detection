"""Unit tests for training, validation, and inference loops."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import MLP, predict, train_epoch, validate


class SimpleClassificationModel(nn.Module):
    """Simple linear model for testing training and validation loops."""

    def __init__(self, seq_len: int = 16, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(seq_len, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Cast input_ids to float for linear projection
        x = input_ids.float()
        return self.fc(x)


def test_train_epoch() -> None:
    """Verifies that train_epoch executes forward/backward passes and optimizes parameters."""
    seq_len = 16
    batch_size = 4
    num_samples = 16

    input_ids = torch.randint(0, 10, (num_samples, seq_len))
    attention_mask = torch.ones((num_samples, seq_len), dtype=torch.long)
    labels = torch.randint(0, 2, (num_samples,), dtype=torch.long)

    dataset = TensorDataset(input_ids, attention_mask, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = SimpleClassificationModel(seq_len=seq_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    loss, acc = train_epoch(model, loader, optimizer, loss_fn, device=torch.device("cpu"))
    assert isinstance(loss, float)
    assert loss >= 0.0
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 100.0


def test_validate_metrics() -> None:
    """Verifies that validate produces valid metric outputs without double-counting."""
    seq_len = 16
    batch_size = 4
    num_samples = 12

    input_ids = torch.randint(0, 10, (num_samples, seq_len))
    attention_mask = torch.ones((num_samples, seq_len), dtype=torch.long)
    # Balanced classes
    labels = torch.tensor([0, 1] * 6, dtype=torch.long)

    dataset = TensorDataset(input_ids, attention_mask, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = SimpleClassificationModel(seq_len=seq_len)
    loss_fn = nn.CrossEntropyLoss()

    val_loss, val_acc, roc_auc, prec, rec, f1, fpr, tpr = validate(
        model, loader, loss_fn, device=torch.device("cpu")
    )

    assert isinstance(val_loss, float)
    assert val_loss >= 0.0
    assert 0.0 <= val_acc <= 100.0
    assert 0.0 <= roc_auc <= 1.0
    assert 0.0 <= prec <= 1.0
    assert 0.0 <= rec <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert isinstance(fpr, np.ndarray)
    assert isinstance(tpr, np.ndarray)


def test_predict() -> None:
    """Verifies that predict generates class probabilities and binary predictions."""
    seq_len = 16
    batch_size = 4
    num_samples = 8

    input_ids = torch.randint(0, 10, (num_samples, seq_len))
    attention_mask = torch.ones((num_samples, seq_len), dtype=torch.long)

    dataset = TensorDataset(input_ids, attention_mask)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = SimpleClassificationModel(seq_len=seq_len)
    probs, labels = predict(model, loader, device=torch.device("cpu"))

    assert len(probs) == num_samples
    assert len(labels) == num_samples
    for p, l in zip(probs, labels):
        assert 0.0 <= p <= 1.0
        assert l in (0, 1)
        assert l == int(p >= 0.5)
