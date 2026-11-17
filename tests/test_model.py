"""Unit tests for MLP and CombinedModel architectures in model.py."""

from typing import Tuple
import pytest
import torch
from torch import nn

from model import MLP, CombinedModel, freeze_bert_layers


class MockBertEncoderLayer(nn.Module):
    """Mock BERT encoder layer for testing parameter freezing."""

    def __init__(self, hidden_dim: int = 768) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MockBertEncoder(nn.Module):
    """Mock BERT encoder containing multiple layers."""

    def __init__(self, num_layers: int = 12, hidden_dim: int = 768) -> None:
        super().__init__()
        self.layer = nn.ModuleList([MockBertEncoderLayer(hidden_dim) for _ in range(num_layers)])


class MockBertModel(nn.Module):
    """Mock BERT backbone returning last_hidden_state with shape (batch_size, seq_len, 768)."""

    def __init__(self, num_layers: int = 12, hidden_dim: int = 768) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = MockBertEncoder(num_layers, hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, seq_len = input_ids.shape
        # Return a namespace-like object mimicking BaseModelOutput
        class MockOutput:
            def __init__(self, tensor: torch.Tensor) -> None:
                self.last_hidden_state = tensor

        dummy_features = torch.randn(batch_size, seq_len, self.hidden_dim)
        return MockOutput(dummy_features)


def test_mlp_initialization() -> None:
    """Verifies that MLP layers and normalizations are properly instantiated."""
    mlp = MLP(input_dim=768, hidden_dim1=512, hidden_dim2=256, hidden_dim3=128, output_dim=2)
    assert isinstance(mlp.fc1, nn.Linear)
    assert mlp.fc1.in_features == 768
    assert mlp.fc1.out_features == 512

    assert isinstance(mlp.fc2, nn.Linear)
    assert mlp.fc2.in_features == 512
    assert mlp.fc2.out_features == 256

    assert isinstance(mlp.fc3, nn.Linear)
    assert mlp.fc3.in_features == 256
    assert mlp.fc3.out_features == 128

    assert isinstance(mlp.fc4, nn.Linear)
    assert mlp.fc4.in_features == 128
    assert mlp.fc4.out_features == 2

    assert isinstance(mlp.relu, nn.ReLU)
    assert isinstance(mlp.dropout, nn.Dropout)
    assert isinstance(mlp.batchnorm1, nn.BatchNorm1d)
    assert isinstance(mlp.batchnorm2, nn.BatchNorm1d)
    assert isinstance(mlp.batchnorm3, nn.BatchNorm1d)


def test_mlp_forward_pass_shape() -> None:
    """Verifies output shape and batch dimension of MLP."""
    mlp = MLP(input_dim=768, hidden_dim1=512, hidden_dim2=256, hidden_dim3=128, output_dim=2)
    mlp.eval()
    batch_size = 8
    dummy_input = torch.randn(batch_size, 768)
    logits = mlp(dummy_input)
    assert logits.shape == (batch_size, 2)


def test_mlp_eval_single_sample() -> None:
    """Verifies that a single evaluation sample passes through BatchNorm without error."""
    mlp = MLP(input_dim=768, hidden_dim1=512, hidden_dim2=256, hidden_dim3=128, output_dim=2)
    mlp.eval()
    single_input = torch.randn(1, 768)
    output = mlp(single_input)
    assert output.shape == (1, 2)


def test_combined_model_forward() -> None:
    """Verifies that CombinedModel passes input_ids through BERT and MLP classification head."""
    bert = MockBertModel(num_layers=4, hidden_dim=768)
    mlp = MLP(input_dim=768, hidden_dim1=512, hidden_dim2=256, hidden_dim3=128, output_dim=2)
    combined = CombinedModel(bert, mlp)
    combined.eval()

    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    logits = combined(input_ids, attention_mask)
    assert logits.shape == (batch_size, 2)


def test_freeze_bert_layers() -> None:
    """Tests selective layer freezing in BERT backbone."""
    bert = MockBertModel(num_layers=12, hidden_dim=768)
    mlp = MLP()
    combined = CombinedModel(bert, mlp)

    # Freeze first 6 layers
    frozen_count = freeze_bert_layers(combined, num_layers_to_freeze=6)
    assert frozen_count == 6

    # Verify first 6 layers require_grad is False
    for i in range(6):
        for param in combined.bert.encoder.layer[i].parameters():
            assert not param.requires_grad

    # Verify remaining 6 layers require_grad is True
    for i in range(6, 12):
        for param in combined.bert.encoder.layer[i].parameters():
            assert param.requires_grad

    # Freeze all layers
    freeze_bert_layers(combined, num_layers_to_freeze=None)
    for i in range(12):
        for param in combined.bert.encoder.layer[i].parameters():
            assert not param.requires_grad
