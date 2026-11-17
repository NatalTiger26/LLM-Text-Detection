"""Unit tests for model serialization and checkpointing in model.py."""

from pathlib import Path
import pytest
import torch
from torch import nn

from model import MLP, load_checkpoint, save_checkpoint


def test_save_and_load_checkpoint(tmp_path: Path) -> None:
    """Verifies that model parameters and training state are accurately serialized and restored."""
    checkpoint_file = tmp_path / "test_checkpoint.pth"

    model_src = MLP(input_dim=64, hidden_dim1=32, hidden_dim2=16, hidden_dim3=8, output_dim=2)
    optimizer_src = torch.optim.AdamW(model_src.parameters(), lr=0.001)

    save_checkpoint(
        path=checkpoint_file,
        model=model_src,
        optimizer=optimizer_src,
        epoch=5,
        val_acc=98.5,
    )

    assert checkpoint_file.exists()

    model_dest = MLP(input_dim=64, hidden_dim1=32, hidden_dim2=16, hidden_dim3=8, output_dim=2)
    optimizer_dest = torch.optim.AdamW(model_dest.parameters(), lr=0.001)

    checkpoint = load_checkpoint(
        path=checkpoint_file,
        model=model_dest,
        optimizer=optimizer_dest,
        device=torch.device("cpu"),
    )

    assert checkpoint["epoch"] == 5
    assert checkpoint["val_acc"] == 98.5

    # Check that model parameters match exactly
    for p1, p2 in zip(model_src.parameters(), model_dest.parameters()):
        assert torch.allclose(p1, p2)
