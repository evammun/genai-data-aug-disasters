# model_tester.py

import torch
from ..data.load_dali import get_dali_loaders
import numpy as np
from .model_setup import initialize_model
import config
import os


def load_model(model_name, model_path=None):
    """
    Load a saved model from a checkpoint file. If model_path is None or matches
    the legacy config.MODEL_PATHS entry, automatically swap to config.get_model_paths().
    """
    from config import (
        NUM_CLASSES,
        DEVICE,
        PROJECT_ROOT,
        MODEL_PATHS,  # The old legacy dictionary
        get_model_paths,  # The version-aware dictionary
    )
    from .model_setup import initialize_model

    # Use the centralized number of classes
    num_classes = NUM_CLASSES

    # Use centralized DEVICE if available.
    device = torch.device(DEVICE) if torch.cuda.is_available() else torch.device("cpu")

    # If no path is provided, or the caller is using the old model_path from config.MODEL_PATHS:
    # then automatically switch to config.get_model_paths().
    if model_path is None or (
        model_name in MODEL_PATHS and model_path == MODEL_PATHS[model_name]
    ):
        versioned_paths = get_model_paths()  # e.g. {"resnet50": "models/original/..."}
        model_path = versioned_paths[model_name]

    # Convert relative path to absolute path if needed
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, "WSLcode", model_path)

    print(f"[load_model] Loading checkpoint from: {model_path}")

    # Initialize model architecture.
    model = initialize_model(model_name, num_classes)

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, torch.nn.Module):
        # If the entire model was saved
        model = checkpoint
    else:
        # Otherwise, load the state dictionary
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()  # Set to evaluation mode.
    return model


def test_model(model, data_dir, batch_size, num_threads):
    """
    Evaluate the trained model on the test dataset and return predictions, true labels,
    and the ordered list of file paths for each sample. This function depends on
    get_dali_loaders returning a tuple (train_loader, val_loader, test_loader) where
    the third element corresponds to the test set.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model to evaluate.
    data_dir : str
        Directory path where the test data and MEDIC_test.tsv are located.
    batch_size : int
        Number of samples per batch during testing.
    num_threads : int
        Number of threads to use for data loading.

    Returns
    -------
    tuple:
      - true_labels_dict (dict): Mapping task -> list of true labels.
      - predictions_dict (dict): Mapping task -> list of predicted labels.
      - test_files (list): List of file paths in the same order as processed.
    """

    num_classes = config.NUM_CLASSES
    device = (
        torch.device(config.DEVICE)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = model.to(device)
    model.eval()

    # Use torch.cuda.current_device() if available for the DALI pipeline.
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0

    # Retrieve the test data loader (assumes get_dali_loaders returns a tuple with test loader as the third element)
    test_loader, test_batches, test_files = get_dali_loaders(
        data_dir, batch_size, num_threads, device_id, phase="test"
    )[2]

    # Initialize dictionaries to store true labels and predictions.
    true_labels_dict = {task: [] for task in num_classes}
    predictions_dict = {task: [] for task in num_classes}

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0]["data"].to(device)
            combined_label = batch[0]["labels"].to(device)

            # Decode labels for each task from the combined label.
            labels = {
                "damage_severity": combined_label // 1000,
                "humanitarian": (combined_label % 1000) // 100,
                "informative": (combined_label % 100) // 10,
                "disaster_types": (combined_label % 10),
            }

            outputs = model(inputs)

            # For each task, compute predictions and store results.
            for task in num_classes:
                _, preds = torch.max(outputs[task], 1)
                true_labels_dict[task].extend(labels[task].cpu().numpy())
                predictions_dict[task].extend(preds.cpu().numpy())

    return true_labels_dict, predictions_dict, test_files
