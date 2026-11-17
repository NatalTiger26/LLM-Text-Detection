"""Unit tests validating notebook integrity, syntax, and schema."""

import ast
import json
from pathlib import Path
import pytest

NOTEBOOK_PATHS = [
    Path("BERT Model finetuning.ipynb"),
    Path("llm bert model verification.ipynb"),
    Path("llm-model-examples-prediction.ipynb"),
]


@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS)
def test_notebook_exists(nb_path: Path) -> None:
    """Verifies that each notebook exists at its designated relative path."""
    assert nb_path.exists(), f"Notebook {nb_path} was moved or removed!"


@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS)
def test_notebook_valid_json_and_nbformat(nb_path: Path) -> None:
    """Verifies that each notebook is valid JSON adhering to nbformat version 4."""
    with open(nb_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "cells" in data
    assert "metadata" in data
    assert data.get("nbformat") == 4
    assert len(data["cells"]) > 0


@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS)
def test_notebook_code_cells_ast_valid(nb_path: Path) -> None:
    """Verifies that all Python code cells in the notebooks parse without syntax errors."""
    with open(nb_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, cell in enumerate(data.get("cells", [])):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            # Filter out IPython magic commands if any
            lines = [
                line for line in source.split("\n")
                if not line.strip().startswith(("%", "!"))
            ]
            clean_code = "\n".join(lines)
            try:
                ast.parse(clean_code)
            except SyntaxError as e:
                pytest.fail(f"SyntaxError in {nb_path} cell {idx}: {e}\nCode:\n{clean_code}")


@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS)
def test_notebook_no_unhandled_error_outputs(nb_path: Path) -> None:
    """Verifies that notebooks do not retain unhandled exception tracebacks in output cells."""
    with open(nb_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, cell in enumerate(data.get("cells", [])):
        if cell.get("cell_type") == "code":
            for output in cell.get("outputs", []):
                assert output.get("output_type") != "error", (
                    f"Notebook {nb_path} cell {idx} contains an unhandled error: "
                    f"{output.get('ename')}: {output.get('evalue')}"
                )
