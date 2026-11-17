"""Pytest test configuration and fixtures for LLM-Text-Detection.

Loads model and pipeline definitions directly from the modernized notebooks
to verify their implementations, type contracts, and execution behavior.
"""

import json
from pathlib import Path
import types
from typing import Any, Dict, List, Tuple
import pytest
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parent.parent


def extract_module_from_notebook(nb_filename: str, cell_indices: List[int]) -> types.ModuleType:
    """Extract and execute specified code cells from a notebook into a module namespace.

    Args:
        nb_filename: Notebook file name relative to repository root.
        cell_indices: Indices of the code cells to extract.

    Returns:
        types.ModuleType: Dynamically executed module containing the notebook's symbols.
    """
    nb_path = REPO_ROOT / nb_filename
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    mod = types.ModuleType(f"nb_{Path(nb_filename).stem}")
    mod.__file__ = str(nb_path)

    # Pre-inject standard modules
    mod.__dict__["torch"] = torch
    mod.__dict__["nn"] = nn

    combined_code = "\n\n".join(
        "".join(nb["cells"][idx]["source"])
        for idx in cell_indices
        if nb["cells"][idx]["cell_type"] == "code"
    )

    exec(combined_code, mod.__dict__)
    return mod


@pytest.fixture(scope="session")
def finetuning_module() -> types.ModuleType:
    """Module containing definitions from 'BERT Model finetuning.ipynb'."""
    # Cell 1: imports, Cell 4: tokenize_and_encode, Cell 10: MLP & CombinedModel, Cell 12: freeze_bert_layers
    return extract_module_from_notebook("BERT Model finetuning.ipynb", [1, 4, 10, 12])


@pytest.fixture(scope="session")
def verification_module() -> types.ModuleType:
    """Module containing definitions from 'llm bert model verification.ipynb'."""
    # Cell 1: imports, Cell 4: tokenize_and_encode, Cell 8: MLP & CombinedModel
    return extract_module_from_notebook("llm bert model verification.ipynb", [1, 4, 8])


@pytest.fixture(scope="session")
def examples_module() -> types.ModuleType:
    """Module containing definitions from 'llm-model-examples-prediction.ipynb'."""
    # Cell 0: imports, Cell 1: MLP, CombinedModel, tokenize_and_encode, testing
    return extract_module_from_notebook("llm-model-examples-prediction.ipynb", [0, 1])
