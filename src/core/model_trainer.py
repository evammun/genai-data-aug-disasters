import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ..utils.callbacks import EarlyStopping, LRScheduler
from ..data.load_dali import get_dali_loaders
import numpy as np
import copy
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import json
from torch.utils.tensorboard import SummaryWriter
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import f1_score  # Importing f1_score from sklearn
import shutil

# Import centralized configuration defaults.
import config


def create_optimizer(model, learning_rate):
    """
    Create and configure an optimizer for the specified model using Adam optimizer.

    Parameters
    ----------
    model : torch.nn.Module
        The model for which the optimizer will be set up.
    learning_rate : float
        The learning rate parameter for the optimizer.

    Returns
    -------
    torch.optim.Optimizer
        Configured optimizer.
    """
    # Initialize the Adam optimizer with the specified learning rate and weight decay.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    return optimizer


def compute_metrics(outputs, labels):
    """
    Compute accuracy for each task based on model outputs and true labels.

    Parameters
    ----------
    outputs : dict
        Dictionary mapping task names to output logits.
    labels : dict
        Dictionary mapping task names to true labels.

    Returns
    -------
    dict
        Dictionary containing accuracy metrics per task.
    """
    accuracies = {}
    for task, output in outputs.items():
        _, preds = torch.max(output, 1)
        accuracies[task] = (preds == labels[task]).float().mean().item()
    return accuracies


def train_model(
    model,
    model_name,
    data_dir: str = None,
    criterion: dict = None,
    optimizer=None,
    learning_rate: float = None,
    initial_patience: int = None,
    num_epochs: int = None,
    device=None,
    batch_size: int = None,
    num_threads: int = None,
    callbacks: list = None,
    delete_tensorboard: bool = False,
    save_dir: str = None,
    num_classes: dict = None,
    dali: bool = True,
):
    """
    Train a model with specified parameters, employing validation, logging, and early stopping.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    model_name : str
        A unique name for the model (used for TensorBoard logging).
    data_dir : str, optional
        Directory where the training and validation data are located.
        Defaults to DATA_DIR from config.
    criterion : dict
        Dictionary of loss functions for each task.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    learning_rate : float, optional
        Learning rate for training. Defaults to LEARNING_RATE from config.
    initial_patience : int, optional
        Early stopping patience. Defaults to INITIAL_PATIENCE from config.
    num_epochs : int
        Total number of training epochs.
    device : torch.device, optional
        Device to use (CPU or GPU). If not provided, set from torch.cuda if available.
    batch_size : int, optional
        Number of samples per batch. Defaults to BATCH_SIZE from config.
    num_threads : int, optional
        Number of threads for data loading. Defaults to NUM_THREADS from config.
    callbacks : list, optional
        List of callback objects for early stopping and LR scheduling.
    delete_tensorboard : bool, optional
        Whether to delete existing TensorBoard logs.
    save_dir : str
        Directory to save the best model.
    num_classes : dict, optional
        Dictionary mapping task names to number of classes.
        Defaults to NUM_CLASSES from config.
    dali : bool, optional
        If True, use DALI data loaders.

    Returns
    -------
    tuple
        (trained model, best validation accuracy dictionary, best validation F1 dictionary)
    """

    if data_dir is None:
        data_dir = config.DATA_DIR
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_threads is None:
        num_threads = config.NUM_THREADS
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    if initial_patience is None:
        initial_patience = config.INITIAL_PATIENCE
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if save_dir is None:
        # Compute the project root (assuming this file is in WSLcode/src/core/)
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        models_base = os.path.join(project_root, "models")
        save_dir = os.path.join(models_base, config.DATASET_VERSION)

    # Retrieve the current CUDA device ID if available.
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
    model = model.to(device)

    # Load data using DALI loaders for train and validation phases.
    train_loader, train_batches, _ = get_dali_loaders(
        data_dir, batch_size, num_threads, device_id, phase="train"
    )[0]
    val_loader, val_batches, _ = get_dali_loaders(
        data_dir, batch_size, num_threads, device_id, phase="val"
    )[1]

    # Reset early stopping flags if callbacks are provided.
    if callbacks:
        for callback in callbacks:
            if hasattr(callback, "reset"):
                callback.reset()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    # Create dictionaries for best accuracy and F1 per task.
    best_val_accuracy = {task: 0.0 for task in criterion} if criterion else {}
    best_val_f1 = {task: 0.0 for task in criterion} if criterion else {}

    # Initialize callbacks.
    early_stopping = EarlyStopping(initial_patience, final_patience=10, verbose=False)
    lr_scheduler = LRScheduler(optimizer, factor=0.1)

    # TensorBoard setup.

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tensorboard_base = os.path.join(project_root, "tensorboard")
    tensorboard_dir = os.path.join(
        tensorboard_base, config.DATASET_VERSION, str(learning_rate)
    )
    if delete_tensorboard and os.path.exists(tensorboard_dir):
        shutil.rmtree(tensorboard_dir)
        os.makedirs(tensorboard_dir)  # Recreate it after deletion

    writer = SummaryWriter(tensorboard_dir)

    layout = {
        "Training": {
            "Accuracy": [
                "Multiline",
                [f"{model_name}/Training/Accuracy/{task}" for task in num_classes],
            ],
            "F1 Score": [
                "Multiline",
                [f"{model_name}/Training/F1/{task}" for task in num_classes],
            ],
            "Loss": [
                "Multiline",
                [f"{model_name}/Training/Loss/{task}" for task in num_classes],
            ],
        },
        "Validation": {
            "Accuracy": [
                "Multiline",
                [f"{model_name}/Validation/Accuracy/{task}" for task in num_classes],
            ],
            "F1 Score": [
                "Multiline",
                [f"{model_name}/Validation/F1/{task}" for task in num_classes],
            ],
            "Loss": [
                "Multiline",
                [f"{model_name}/Validation/Loss/{task}" for task in num_classes],
            ],
        },
    }
    writer.add_custom_scalars(layout)

    # Training loop.
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = {t: 0.0 for t in criterion} if criterion else {}
            running_total = {t: 0 for t in criterion} if criterion else {}
            running_corrects = {t: 0 for t in criterion} if criterion else {}
            true_labels = {task: [] for task in num_classes}
            predictions = {task: [] for task in num_classes}

            for batch in tqdm(
                loader,
                total=train_batches if phase == "train" else val_batches,
                desc=f"{phase.capitalize()} Phase, Epoch {epoch+1}",
            ):
                if dali:
                    inputs = batch[0]["data"].to(device)
                    combined_label = batch[0]["labels"].to(device)

                    # Remove the extra dimension BEFORE decoding
                    combined_label = combined_label.squeeze(1)

                    # Decode with debug prints
                    labels = {}

                    # Damage severity
                    damage = combined_label // 1000

                    # Humanitarian
                    humanitarian = (combined_label % 1000) // 100

                    # Informative
                    informative = (combined_label % 100) // 10

                    # Disaster types
                    disaster = combined_label % 10

                    labels = {
                        "damage_severity": damage,
                        "humanitarian": humanitarian,
                        "informative": informative,
                        "disaster_types": disaster,
                    }

                    for key in labels:
                        labels[key] = labels[key].view(-1).long()
                else:
                    data = batch[0]
                    inputs = data["data"].to(device)
                    labels = {
                        task: data[task].to(device).long().squeeze(1)
                        for task in num_classes
                    }

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    losses = {}
                    total_loss = torch.tensor(
                        0.0, requires_grad=True, dtype=torch.float, device=device
                    )
                    total_loss.to(device)
                    for task, task_output in outputs.items():
                        _, preds = torch.max(task_output, 1)

                        predictions[task].extend(preds.cpu().numpy())
                        true_labels[task].extend(labels[task].cpu().numpy())
                        running_corrects[task] += (preds == labels[task]).sum().item()

                        valid_idx = labels[task] != -1
                        task_loss = criterion[task](
                            outputs[task][valid_idx], labels[task][valid_idx]
                        ).to(device)

                        losses[task] = task_loss.item()
                        total_loss = total_loss + task_loss
                        running_loss[task] += task_loss.item() * valid_idx.sum().item()
                        running_total[task] += valid_idx.sum().item()

                    if phase == "train":
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

            accuracy_dict = {}
            f1_dict = {}
            for task in num_classes:
                epoch_accuracy = accuracy_score(true_labels[task], predictions[task])
                epoch_f1 = f1_score(
                    true_labels[task], predictions[task], average="weighted"
                )
                accuracy_dict[task] = epoch_accuracy
                f1_dict[task] = epoch_f1
                if phase == "val":
                    best_val_accuracy[task] = max(
                        best_val_accuracy.get(task, 0), epoch_accuracy
                    )
                    best_val_f1[task] = max(best_val_f1.get(task, 0), epoch_f1)

            epoch_loss = sum(running_loss.values()) / sum(running_total.values())
            task_losses = {t: running_loss[t] / running_total[t] for t in running_loss}
            task_acc = {
                t: running_corrects[t] / running_total[t] for t in running_corrects
            }
            epoch_acc = sum(task_acc.values()) / len(task_acc)
            task_losses["overall"] = epoch_loss
            task_acc["overall"] = epoch_acc

            writer.add_scalars(
                f"{model_name}/{phase.capitalize()}/Accuracy",
                accuracy_dict,
                global_step=epoch,
            )
            writer.add_scalars(
                f"{model_name}/{phase.capitalize()}/F1", f1_dict, global_step=epoch
            )
            writer.add_scalars(
                f"{model_name}/{phase.capitalize()}/Loss",
                task_losses,
                global_step=epoch,
            )
            writer.add_scalar(
                f"{model_name}/{phase.capitalize()}/Total_Loss",
                epoch_loss,
                global_step=epoch,
            )

            if phase == "val":
                status = early_stopping(epoch_loss)
                if status == "reduce_lr":
                    lr_scheduler.step()
                elif status == "stop":
                    print("Early stopping triggered.")
                    model.load_state_dict(best_model_wts)
                    writer.close()
                    return model, best_val_accuracy, best_val_f1
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    save_path = os.path.join(save_dir, f"best_model_{model_name}.pth")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(model, save_path)

    model.load_state_dict(best_model_wts)
    writer.close()
    return model, best_val_accuracy, best_val_f1
