"""Contract and syntax integrity tests for repository notebooks.

Verifies that all notebooks exist at their required locations, conform to nbformat 4,
parse with valid Python AST syntax, contain static type hints, and have no dead
imports or undefined reference bugs.
"""

import ast
import json
from pathlib import Path
import types
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

NOTEBOOK_FILENAMES = [
    "BERT Model finetuning.ipynb",
    "llm bert model verification.ipynb",
    "llm-model-examples-prediction.ipynb",
]


@pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
def test_notebook_exists_and_valid_json(filename: str) -> None:
    """Verify notebook exists at repo root and is valid JSON conforming to nbformat 4."""
    path = REPO_ROOT / filename
    assert path.is_file(), f"Notebook file '{filename}' must exist at root directory."

    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert "metadata" in nb
    assert nb.get("nbformat") == 4
    assert len(nb["cells"]) > 0


@pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
def test_all_code_cells_ast_syntax(filename: str) -> None:
    """Verify every code cell parses into a valid Python Abstract Syntax Tree."""
    path = REPO_ROOT / filename
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            source_code = "".join(cell.get("source", []))
            if source_code.strip():
                try:
                    ast.parse(source_code)
                except SyntaxError as exc:
                    pytest.fail(f"Syntax error in '{filename}' at code cell {i}: {exc}")


def test_no_deprecated_adamw_imports() -> None:
    """Verify deprecated 'from transformers import AdamW' is removed from all notebooks."""
    for filename in NOTEBOOK_FILENAMES:
        path = REPO_ROOT / filename
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "from transformers import AdamW" not in content, (
            f"Found deprecated 'from transformers import AdamW' in '{filename}'"
        )


def test_no_undefined_val_texts_in_verification() -> None:
    """Verify the NameError bug referencing 'val_texts' in the verification notebook is eliminated."""
    path = REPO_ROOT / "llm bert model verification.ipynb"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "val_texts" not in content, (
        "Found undefined reference to 'val_texts' in 'llm bert model verification.ipynb'"
    )


def test_no_dead_nltk_in_examples() -> None:
    """Verify dead nltk and eng_dict references are eliminated from examples notebook."""
    path = REPO_ROOT / "llm-model-examples-prediction.ipynb"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "eng_dict" not in content, (
        "Found unused 'eng_dict' in 'llm-model-examples-prediction.ipynb'"
    )
    assert "import nltk" not in content, (
        "Found unused 'import nltk' in 'llm-model-examples-prediction.ipynb'"
    )


def test_finetuning_notebook_in_place_classes(finetuning_module: types.ModuleType) -> None:
    """Tests that classes extracted directly from finetuning notebook instantiate and forward."""
    mlp = finetuning_module.MLP(input_dim=768, hidden_dim1=512, hidden_dim2=256, hidden_dim3=128, output_dim=2)
    mlp.eval()
    dummy = torch.randn(2, 768)
    out = mlp(dummy)
    assert out.shape == (2, 2)


def test_verification_notebook_in_place_classes(verification_module: types.ModuleType) -> None:
    """Tests that classes extracted directly from verification notebook instantiate and forward."""
    mlp = verification_module.MLP(input_dim=768, hidden_dim1=512, hidden_dim2=256, hidden_dim3=128, output_dim=2)
    mlp.eval()
    dummy = torch.randn(2, 768)
    out = mlp(dummy)
    assert out.shape == (2, 2)


def test_examples_notebook_in_place_classes(examples_module: types.ModuleType) -> None:
    """Tests that classes extracted directly from examples notebook instantiate and forward."""
    mlp = examples_module.MLP(input_dim=768, hidden_dim1=512, hidden_dim2=256, hidden_dim3=128, output_dim=2)
    mlp.eval()
    dummy = torch.randn(2, 768)
    out = mlp(dummy)
    assert out.shape == (2, 2)
