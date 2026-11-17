"""BERT-based LLM Text Detection Model and Pipeline Utilities.

This module provides the core deep learning architecture and utilities for
detecting AI-generated text using a pretrained BERT backbone coupled with a
multi-layer perceptron (MLP) classification head.

Classes:
    MLP: Multi-layer perceptron classification head.
    CombinedModel: Composite architecture combining BERT and MLP.

Functions:
    tokenize_and_encode: Vectorized batch tokenization for sequences.
    freeze_bert_layers: Selective freezing of BERT encoder transformer layers.
    train_epoch: Executes one training epoch over a DataLoader.
    validate: Evaluates model performance and computes validation metrics.
    predict: Generates prediction probabilities and binary class labels.
    calculate_tp_tn_fp_fn: Derives confusion matrix counts from summary metrics.
    save_checkpoint: Persists model state and metrics to disk.
    load_checkpoint: Loads model weights safely with device mapping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) classification head.

    Applies sequential linear transformations with batch normalization and
    ReLU activations to project BERT CLS representations into class logits.

    Architecture:
        Linear(input_dim -> hidden_dim1) -> BatchNorm1d -> ReLU ->
        Linear(hidden_dim1 -> hidden_dim2) -> BatchNorm1d -> ReLU ->
        Linear(hidden_dim2 -> hidden_dim3) -> BatchNorm1d -> ReLU ->
        Linear(hidden_dim3 -> output_dim)

    Attributes:
        fc1 (nn.Linear): First linear layer.
        fc2 (nn.Linear): Second linear layer.
        fc3 (nn.Linear): Third linear layer.
        fc4 (nn.Linear): Final projection layer producing logits.
        relu (nn.ReLU): Non-linear activation function.
        dropout (nn.Dropout): Dropout regularization layer.
        batchnorm1 (nn.BatchNorm1d): Batch normalization after first linear layer.
        batchnorm2 (nn.BatchNorm1d): Batch normalization after second linear layer.
        batchnorm3 (nn.BatchNorm1d): Batch normalization after third linear layer.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim1: int = 512,
        hidden_dim2: int = 256,
        hidden_dim3: int = 128,
        output_dim: int = 2,
        dropout_rate: float = 0.2,
    ) -> None:
        """Initializes the MLP layers.

        Args:
            input_dim (int): Dimensionality of input features (default 768 for BERT base).
            hidden_dim1 (int): First hidden layer dimensionality (default 512).
            hidden_dim2 (int): Second hidden layer dimensionality (default 256).
            hidden_dim3 (int): Third hidden layer dimensionality (default 128).
            output_dim (int): Output dimensionality / number of classes (default 2).
            dropout_rate (float): Dropout probability (default 0.2).
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim1)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim2)
        self.batchnorm3 = nn.BatchNorm1d(hidden_dim3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input feature tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Raw output logits of shape (batch_size, output_dim).
        """
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


class CombinedModel(nn.Module):
    """End-to-end classifier coupling a pretrained BERT encoder with an MLP head.

    The model passes tokenized inputs through the BERT encoder, extracts the
    last hidden state corresponding to the [CLS] token (index 0), and passes
    it through the MLP head to produce binary classification logits.

    Attributes:
        bert (nn.Module): Pretrained BERT backbone (e.g. BertModel or AutoModel).
        mlp (nn.Module): Downstream classification head (e.g. MLP).
    """

    def __init__(self, bert_model: nn.Module, mlp: nn.Module) -> None:
        """Initializes the CombinedModel.

        Args:
            bert_model (nn.Module): Pretrained BERT model.
            mlp (nn.Module): MLP classification head.
        """
        super().__init__()
        self.bert = bert_model
        self.mlp = mlp

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through BERT and the MLP head.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Binary mask of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits of shape (batch_size, output_dim).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the representation of the [CLS] token at index 0
        if hasattr(outputs, "last_hidden_state"):
            cls_representation = outputs.last_hidden_state[:, 0, :]
        elif isinstance(outputs, (tuple, list)):
            cls_representation = outputs[0][:, 0, :]
        else:
            raise ValueError("Unsupported BERT output structure.")
        logits = self.mlp(cls_representation)
        return logits


def freeze_bert_layers(model: CombinedModel, num_layers_to_freeze: Optional[int] = None) -> int:
    """Selectively freezes bottom transformer encoder layers in the BERT backbone.

    Freezing lower layers reduces memory footprint and backpropagation compute
    time while retaining general linguistic features learned during pretraining.

    Args:
        model (CombinedModel): The combined model instance.
        num_layers_to_freeze (Optional[int]): Number of bottom encoder layers to
            freeze. If None or >= total layers, all encoder layers are frozen.
            If 0, all encoder layers remain trainable.

    Returns:
        int: Number of layers frozen.
    """
    if not hasattr(model.bert, "encoder") or not hasattr(model.bert.encoder, "layer"):
        logger.warning("Model backbone does not have bert.encoder.layer structure.")
        return 0

    total_layers = len(model.bert.encoder.layer)
    if num_layers_to_freeze is None:
        freeze_until = total_layers
    else:
        freeze_until = max(0, min(num_layers_to_freeze, total_layers))

    for idx, layer in enumerate(model.bert.encoder.layer):
        requires_grad = idx >= freeze_until
        for param in layer.parameters():
            param.requires_grad = requires_grad

    return freeze_until


def tokenize_and_encode(
    tokenizer: Any,
    comments: Sequence[str],
    labels: Optional[Sequence[Union[int, float]]] = None,
    max_length: int = 128,
    batch_size: int = 1024,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Vectorized tokenization of text sequences into PyTorch tensors.

    Processes sequences in vectorized chunks using the fast tokenizer backend,
    avoiding slow one-by-one Python iteration while preventing high memory peaks.

    Args:
        tokenizer (Any): Hugging Face PreTrainedTokenizerFast or AutoTokenizer.
        comments (Sequence[str]): List or sequence of raw text strings.
        labels (Optional[Sequence[Union[int, float]]]): Optional binary class labels.
        max_length (int): Maximum token sequence length (padding & truncation bound).
        batch_size (int): Chunk size for vectorized tokenization (default 1024).

    Returns:
        Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
            If labels are provided: (input_ids, attention_masks, labels_tensor).
            If labels are None: (input_ids, attention_masks).
    """
    comments_list = [str(c) if c is not None else "" for c in comments]
    all_input_ids: List[torch.Tensor] = []
    all_attention_masks: List[torch.Tensor] = []

    if len(comments_list) == 0:
        empty_ids = torch.empty((0, max_length), dtype=torch.long)
        empty_mask = torch.empty((0, max_length), dtype=torch.long)
        if labels is not None:
            return empty_ids, empty_mask, torch.empty((0,), dtype=torch.long)
        return empty_ids, empty_mask

    # Vectorized chunked tokenization
    for start_idx in range(0, len(comments_list), batch_size):
        chunk = comments_list[start_idx : start_idx + batch_size]
        encoded = tokenizer(
            chunk,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        all_input_ids.append(encoded["input_ids"])
        all_attention_masks.append(encoded["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_masks = torch.cat(all_attention_masks, dim=0)

    if labels is not None:
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return input_ids, attention_masks, labels_tensor

    return input_ids, attention_masks


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Trains the model for one epoch.

    Args:
        model (nn.Module): Neural network model to train.
        train_loader (DataLoader): DataLoader providing (input_ids, attention_mask, labels).
        optimizer (torch.optim.Optimizer): Optimizer instance (e.g. AdamW).
        loss_fn (nn.Module): Loss function (e.g. CrossEntropyLoss).
        device (Optional[torch.device]): Target compute device. Defaults to auto-detect.

    Returns:
        Tuple[float, float]: (average_loss, accuracy_percentage).
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.train()
    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = [t.to(device) for t in batch]
        labels = labels.to(dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(outputs.detach(), dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = (100.0 * total_correct / total_samples) if total_samples > 0 else 0.0
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: Optional[torch.device] = None,
) -> Tuple[float, float, float, float, float, float, np.ndarray, np.ndarray]:
    """Evaluates the model on validation data without gradient tracking.

    Args:
        model (nn.Module): Neural network model to evaluate.
        val_loader (DataLoader): DataLoader providing (input_ids, attention_mask, labels).
        loss_fn (nn.Module): Loss function (e.g. CrossEntropyLoss).
        device (Optional[torch.device]): Target compute device. Defaults to auto-detect.

    Returns:
        Tuple[float, float, float, float, float, float, np.ndarray, np.ndarray]:
            (val_loss, val_accuracy, roc_auc, precision, recall, f1, fpr, tpr).
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0
    y_pred_probs: List[float] = []
    y_true: List[int] = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            labels = labels.to(dtype=torch.long)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

            # Extract class-1 probability (AI-generated class)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            y_pred_probs.extend(probs.cpu().tolist())
            y_true.extend(labels.cpu().tolist())

    y_pred_arr = np.asarray(y_pred_probs, dtype=np.float64)
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_labels = (y_pred_arr >= 0.5).astype(np.int64)

    val_loss = total_loss / total_samples if total_samples > 0 else 0.0
    val_accuracy = (100.0 * total_correct / total_samples) if total_samples > 0 else 0.0

    # Calculate metrics with zero-division safety
    if len(np.unique(y_true_arr)) > 1:
        roc_auc = float(roc_auc_score(y_true_arr, y_pred_arr))
        fpr, tpr, _ = roc_curve(y_true_arr, y_pred_arr)
    else:
        roc_auc = 0.5
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])

    precision = float(precision_score(y_true_arr, y_pred_labels, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred_labels, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_labels, zero_division=0))

    return val_loss, val_accuracy, roc_auc, precision, recall, f1, fpr, tpr


def predict(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[List[float], List[int]]:
    """Runs pure inference across a test DataLoader without requiring ground-truth labels.

    Args:
        model (nn.Module): Trained neural network model.
        test_loader (DataLoader): DataLoader yielding (input_ids, attention_mask).
        device (Optional[torch.device]): Target compute device. Defaults to auto-detect.

    Returns:
        Tuple[List[float], List[int]]:
            (prediction_probabilities, predicted_labels) for class 1.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    pred_probs: List[float] = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            pred_probs.extend(probs.cpu().tolist())

    pred_labels = [int(p >= 0.5) for p in pred_probs]
    return pred_probs, pred_labels


def calculate_tp_tn_fp_fn(
    accuracy: float,
    precision: float,
    recall: float,
    total_entries: int,
) -> Tuple[float, float, float, float]:
    """Derives confusion matrix components (TP, FN, FP, TN) from summary rates.

    Maintained for legacy compatibility with zero-division and rounding protection.
    For exact evaluation from predictions, prefer sklearn.metrics.confusion_matrix.

    Args:
        accuracy (float): Model accuracy normalized to [0, 1].
        precision (float): Precision score in [0, 1].
        recall (float): Recall score in [0, 1].
        total_entries (int): Total number of evaluation instances.

    Returns:
        Tuple[float, float, float, float]: (tp, fn, fp, tn).
    """
    denom = (1.0 / precision) + (1.0 / recall) - 2.0 if precision > 0 and recall > 0 else 0.0
    if abs(denom) < 1e-9:
        tp = 0.0
    else:
        tp = float(np.round((1.0 - accuracy) * total_entries / denom))

    fn = float(np.round((1.0 / recall - 1.0) * tp)) if recall > 0 else 0.0
    fp = float(np.round((1.0 / precision - 1.0) * tp)) if precision > 0 else 0.0
    tn = float(np.round(total_entries - tp - fp - fn))
    return tp, fn, fp, tn


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    **extra_metrics: Any,
) -> None:
    """Safely saves a model checkpoint and associated training state to disk.

    Args:
        path (Union[str, Path]): Destination file path.
        model (nn.Module): PyTorch model to serialize.
        optimizer (Optional[torch.optim.Optimizer]): Optimizer state dict.
        epoch (Optional[int]): Current training epoch.
        **extra_metrics (Any): Arbitrary metric history or metadata to serialize.
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    checkpoint.update(extra_metrics)
    torch.save(checkpoint, str(save_path))


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Safely loads a model checkpoint with device mapping and CPU fallback.

    Args:
        path (Union[str, Path]): File path to checkpoint.
        model (nn.Module): Model instance into which weights are loaded.
        optimizer (Optional[torch.optim.Optimizer]): Optional optimizer to restore.
        device (Optional[torch.device]): Target compute device. Defaults to auto-detect.
        strict (bool): Whether strictly matching state_dict keys is enforced.

    Returns:
        Dict[str, Any]: Loaded checkpoint dictionary.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint: Dict[str, Any] = torch.load(str(path), map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
