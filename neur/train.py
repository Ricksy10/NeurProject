"""
Training utilities for the Chlorella classification pipeline.

Handles:
- Training loop per fold
- Early stopping based on F0.5 metric
- Discriminative fine-tuning (freeze/unfreeze backbone)
- Checkpoint saving and loading
- Two-stage fine-tuning strategy
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from neur.eval import compute_fbeta_score


class EarlyStopping:
    """
    Early stopping to terminate training when validation metric stops improving.
    """

    def __init__(self, patience: int = 5, mode: str = "max", delta: float = 0.0):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            mode: 'max' to maximize metric, 'min' to minimize
            delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if should stop training.

        Args:
            score: Current metric value

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze backbone parameters (disable gradients).

    Args:
        model: Model with get_backbone_params method
    """
    if hasattr(model, "get_backbone_params"):
        for param in model.get_backbone_params():
            param.requires_grad = False
    else:
        # Fallback: freeze all but last layer
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreeze backbone parameters (enable gradients).

    Args:
        model: Model with get_backbone_params method
    """
    if hasattr(model, "get_backbone_params"):
        for param in model.get_backbone_params():
            param.requires_grad = True
    else:
        # Fallback: unfreeze all
        for param in model.parameters():
            param.requires_grad = True


def get_discriminative_optimizer(
    model: nn.Module, lr_head: float = 1e-3, lr_backbone: float = 1e-4, weight_decay: float = 1e-4
) -> optim.Optimizer:
    """
    Create optimizer with discriminative learning rates.

    Args:
        model: Model with get_backbone_params and get_classifier_params methods
        lr_head: Learning rate for classifier head
        lr_backbone: Learning rate for backbone
        weight_decay: Weight decay

    Returns:
        Adam optimizer with parameter groups
    """
    if hasattr(model, "get_backbone_params") and hasattr(model, "get_classifier_params"):
        optimizer = optim.Adam(
            [
                {"params": model.get_backbone_params(), "lr": lr_backbone},
                {"params": model.get_classifier_params(), "lr": lr_head},
            ],
            weight_decay=weight_decay,
        )
    else:
        # Fallback: single LR for all parameters
        optimizer = optim.Adam(model.parameters(), lr=lr_head, weight_decay=weight_decay)

    return optimizer


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        verbose: Whether to show progress bar

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    iterator = tqdm(dataloader, desc="Training") if verbose else dataloader

    for inputs, labels in iterator:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Skip samples with invalid labels (e.g., test data with label -1)
        valid_mask = labels >= 0
        if not valid_mask.any():
            continue

        inputs = inputs[valid_mask]
        labels = labels[valid_mask]

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate model.

    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        verbose: Whether to show progress bar

    Returns:
        Tuple of (loss, accuracy, chlorella_f0_5, y_true, y_pred, y_probs)
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    iterator = tqdm(dataloader, desc="Validation") if verbose else dataloader

    with torch.no_grad():
        for inputs, labels in iterator:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Skip samples with invalid labels
            valid_mask = labels >= 0
            if not valid_mask.any():
                continue

            inputs = inputs[valid_mask]
            labels = labels[valid_mask]

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            # Collect results
            total_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    # Compute metrics
    avg_loss = total_loss / len(y_true) if len(y_true) > 0 else 0.0
    accuracy = (y_true == y_pred).mean() if len(y_true) > 0 else 0.0
    chlorella_f0_5 = compute_fbeta_score(y_true, y_pred, beta=0.5, class_id=0)

    return avg_loss, accuracy, chlorella_f0_5, y_true, y_pred, y_probs


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold_id: int,
    config: Dict,
    device: torch.device,
    output_dir: str,
    verbose: bool = False,
) -> Dict:
    """
    Train model for one fold with two-stage fine-tuning and early stopping.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        fold_id: Fold identifier
        config: Configuration dictionary
        device: Device to train on
        output_dir: Directory to save checkpoints
        verbose: Whether to show progress bars

    Returns:
        Dictionary with best metrics and validation predictions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training parameters
    epochs = config["training"]["epochs"]
    unfreeze_epoch = config["training"]["unfreeze_epoch"]
    patience = config["training"]["patience"]
    lr_head = config["training"]["lr_head"]
    lr_backbone = config["training"]["lr_backbone"]
    weight_decay = config["training"]["weight_decay"]

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mode="max")

    # Best metrics tracking
    best_f0_5 = 0.0
    best_epoch = 0
    best_val_predictions = None

    # Stage 1: Train head only (backbone frozen)
    print(f"\n[Fold {fold_id}] Stage 1: Training classifier head (backbone frozen)")
    freeze_backbone(model)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head, weight_decay=weight_decay
    )

    for epoch in range(min(unfreeze_epoch, epochs)):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, verbose
        )

        # Validate
        val_loss, val_acc, chlorella_f0_5, y_true, y_pred, y_probs = validate(
            model, val_loader, criterion, device, verbose
        )

        print(
            f"[Fold {fold_id}] Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
            f"F0.5(chlorella): {chlorella_f0_5:.4f}"
        )

        # Save best model
        if chlorella_f0_5 > best_f0_5:
            best_f0_5 = chlorella_f0_5
            best_epoch = epoch + 1

            # Save checkpoint
            checkpoint_path = output_dir / f"fold_{fold_id}_best.pth"
            save_checkpoint(model, checkpoint_path, epoch, chlorella_f0_5, config)

            # Cache validation predictions
            best_val_predictions = {"y_true": y_true, "y_pred": y_pred, "y_probs": y_probs}

            print(
                f"[Fold {fold_id}] New best F0.5(chlorella): {chlorella_f0_5:.4f} → Saved checkpoint"
            )

        # Early stopping check
        if early_stopping(chlorella_f0_5):
            print(f"[Fold {fold_id}] Early stopping triggered at epoch {epoch+1}")
            break

    # Stage 2: Fine-tune entire network (if not stopped early)
    if epoch + 1 >= unfreeze_epoch and not early_stopping.should_stop:
        print(f"\n[Fold {fold_id}] Stage 2: Fine-tuning entire network (backbone unfrozen)")
        unfreeze_backbone(model)
        optimizer = get_discriminative_optimizer(
            model, lr_head=lr_head, lr_backbone=lr_backbone, weight_decay=weight_decay
        )

        # Reset early stopping for stage 2
        early_stopping = EarlyStopping(patience=patience, mode="max")

        for epoch in range(unfreeze_epoch, epochs):
            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, verbose
            )

            # Validate
            val_loss, val_acc, chlorella_f0_5, y_true, y_pred, y_probs = validate(
                model, val_loader, criterion, device, verbose
            )

            print(
                f"[Fold {fold_id}] Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                f"F0.5(chlorella): {chlorella_f0_5:.4f}"
            )

            # Save best model
            if chlorella_f0_5 > best_f0_5:
                best_f0_5 = chlorella_f0_5
                best_epoch = epoch + 1

                # Save checkpoint
                checkpoint_path = output_dir / f"fold_{fold_id}_best.pth"
                save_checkpoint(model, checkpoint_path, epoch, chlorella_f0_5, config)

                # Cache validation predictions
                best_val_predictions = {"y_true": y_true, "y_pred": y_pred, "y_probs": y_probs}

                print(
                    f"[Fold {fold_id}] New best F0.5(chlorella): {chlorella_f0_5:.4f} → Saved checkpoint"
                )

            # Early stopping check
            if early_stopping(chlorella_f0_5):
                print(f"[Fold {fold_id}] Early stopping triggered at epoch {epoch+1}")
                break

    print(
        f"\n[Fold {fold_id}] Training complete. Best F0.5(chlorella): {best_f0_5:.4f} at epoch {best_epoch}"
    )

    return {
        "best_f0_5": best_f0_5,
        "best_epoch": best_epoch,
        "val_predictions": best_val_predictions,
    }


def save_checkpoint(
    model: nn.Module, checkpoint_path: str, epoch: int, metric_value: float, config: Dict
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch
        metric_value: Metric value (F0.5)
        config: Configuration dictionary
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metric_value": metric_value,
        "config": config,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    model: nn.Module, checkpoint_path: str, device: torch.device
) -> Tuple[nn.Module, Dict]:
    """
    Load model checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        device: Device to load to

    Returns:
        Tuple of (model, checkpoint_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint
