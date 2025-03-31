"""
embedding_analysis.py

This module provides functions for extracting and reducing penultimate embeddings
from a CNN, as well as utilities for decoding and visualizing multi-task labels.

"""

import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tueplots.constants.color import palettes

try:
    import umap.umap_ as umap
except ImportError:
    umap = None  # UMAP is optional

from ..core.model_tester import load_model
from ..data.load_dali import get_dali_loaders

# We no longer need the local default label mappings;
# import them from config.py instead.
from config import (
    DATA_DIR,
    BATCH_SIZE,
    NUM_THREADS,
    DEVICE,
    LABEL_MAPPINGS,
    TASKS,
    MODEL_PATHS,
)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def register_penultimate_hook(model):
    """
    Registers a forward-hook on the model's nn.Flatten layer to capture embeddings.

    Returns:
        hook_handle (RemovableHandle): handle for the registered hook
        container (list): container where the captured features are appended
    """
    container = []

    def _hook(module, input_tensor, output_tensor):
        container.append(output_tensor.detach().cpu())

    flatten_layer = None
    for m in model.modules():
        if isinstance(m, nn.Flatten):
            flatten_layer = m
            break

    if flatten_layer is None:
        raise RuntimeError(
            "No nn.Flatten layer found. Verify your CustomModel structure."
        )

    hook_handle = flatten_layer.register_forward_hook(_hook)
    return hook_handle, container


def extract_penultimate_embeddings(model, data_loader, device):
    """
    Runs the model on 'data_loader', capturing penultimate-layer embeddings.

    Args:
        model (nn.Module): the CNN
        data_loader (DALI loader): validation images
        device (str): "cpu" or "cuda"

    Returns:
        embeddings (np.ndarray): shape (N, embed_dim)
        all_labels (list): raw integer labels (multi-task encoding)
        all_paths (list): file paths, if provided
    """
    model.eval().to(device)
    hook_handle, feats_container = register_penultimate_hook(model)

    all_labels = []
    all_paths = []

    with torch.no_grad():
        for batch_data in data_loader:
            # Expect a single-element list of dict per batch.
            if not (isinstance(batch_data, list) and len(batch_data) == 1):
                raise ValueError("Expected a single-element list of dict per batch.")
            batch_dict = batch_data[0]
            images = batch_dict["data"].to(device)
            labels = batch_dict["labels"]
            file_paths = batch_dict.get("file_paths", [])

            # Forward pass; hook captures features.
            _ = model(images)

            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().tolist()  # shape (batch_size, 1)
            # Assume each label is a list/tuple; take the first element.
            labels = [l[0] for l in labels]

            all_labels.extend(labels)
            all_paths.extend(file_paths)

    embeddings = torch.cat(feats_container, dim=0).numpy()
    hook_handle.remove()

    return embeddings, all_labels, all_paths


def reduce_embeddings(embeddings, method="pca", n_components=2, **kwargs):
    """
    Dimensionality reduction on 'embeddings' using PCA, t-SNE, or UMAP.

    Args:
        embeddings (np.ndarray): shape (N, D)
        method (str): "pca", "tsne", or "umap"
        n_components (int): target dimension (e.g. 2 for plotting)
        **kwargs: additional arguments for the reduction algorithm

    Returns:
        reduced (np.ndarray): shape (N, n_components)
    """
    method = method.lower()
    if method == "pca":
        reducer = PCA(n_components=n_components, **kwargs)
        reduced = reducer.fit_transform(embeddings)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, **kwargs)
        reduced = reducer.fit_transform(embeddings)
    elif method == "umap":
        if umap is None:
            raise ImportError("UMAP not installed. Try pip install umap-learn.")
        reducer = umap.UMAP(n_components=n_components, **kwargs)
        reduced = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    return reduced


def prettify_label(label_str):
    """
    Replace underscores with spaces and convert to title case.
    E.g., "not_disaster" -> "Not Disaster"
    """
    return label_str.replace("_", " ").title()


def decode_task_labels(raw_labels, color_by_task, label_map_dict):
    """
    Decodes multi-task integer labels into textual labels for a single task,
    then converts them to title case.

    Args:
        raw_labels (list): List of raw integer labels.
        color_by_task (str): Task identifier to decode.
        label_map_dict (dict): Mapping from task to {class_id: class_label}.

    Returns:
        list: Decoded textual labels.
    """
    if color_by_task not in label_map_dict:
        return [str(lbl) for lbl in raw_labels]

    mapping_for_this_task = label_map_dict[color_by_task]
    decoded = []

    for c in raw_labels:
        if not isinstance(c, int):
            decoded.append(str(c))
            continue

        if color_by_task == "damage_severity":
            digit = c // 1000
        elif color_by_task == "humanitarian":
            digit = (c % 1000) // 100
        elif color_by_task == "informative":
            digit = (c % 100) // 10
        elif color_by_task == "disaster_types":
            digit = c % 10
        else:
            decoded.append(str(c))
            continue

        label_str = mapping_for_this_task.get(digit, f"Unknown({digit})")
        decoded.append(prettify_label(label_str))

    return decoded


def prettify_task_name(task_name):
    """
    Converts a task identifier like "disaster_types" into "Disaster Types".
    """
    return task_name.replace("_", " ").title()


def dynamic_alpha(p):
    """
    Given the proportion p of a class (in [0,1]), returns an alpha in [0.2, 1.0].

    If p <= 0.1  -> alpha = 1.0
    If p >= 0.65 -> alpha = 0.2
    Otherwise, linearly interpolate.
    """
    if p <= 0.1:
        return 1.0
    elif p >= 0.6:
        return 0.1
    else:
        frac = (p - 0.1) / (0.6 - 0.1)
        return 1.0 - 0.9 * frac


# -----------------------------------------------------------------------------
# Main Function: analyze_penultimate_embeddings
# -----------------------------------------------------------------------------


"""
embedding_analysis.py

This module provides functions for extracting and reducing penultimate embeddings
from a CNN, as well as utilities for decoding and visualizing multi-task labels.

"""

import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tueplots.constants.color import palettes

try:
    import umap.umap_ as umap
except ImportError:
    umap = None  # UMAP is optional

from ..core.model_tester import load_model
from ..data.load_dali import get_dali_loaders

# We no longer need the local default label mappings;
# import them from config.py instead.
from config import (
    DATA_DIR,
    BATCH_SIZE,
    NUM_THREADS,
    DEVICE,
    LABEL_MAPPINGS,
    TASKS,
    MODEL_PATHS,
)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def register_penultimate_hook(model):
    """
    Registers a forward-hook on the model's nn.Flatten layer to capture embeddings.

    Returns:
        hook_handle (RemovableHandle): handle for the registered hook
        container (list): container where the captured features are appended
    """
    container = []

    def _hook(module, input_tensor, output_tensor):
        container.append(output_tensor.detach().cpu())

    flatten_layer = None
    for m in model.modules():
        if isinstance(m, nn.Flatten):
            flatten_layer = m
            break

    if flatten_layer is None:
        raise RuntimeError(
            "No nn.Flatten layer found. Verify your CustomModel structure."
        )

    hook_handle = flatten_layer.register_forward_hook(_hook)
    return hook_handle, container


def extract_penultimate_embeddings(model, data_loader, device):
    """
    Runs the model on 'data_loader', capturing penultimate-layer embeddings.

    Args:
        model (nn.Module): the CNN
        data_loader (DALI loader): validation images
        device (str): "cpu" or "cuda"

    Returns:
        embeddings (np.ndarray): shape (N, embed_dim)
        all_labels (list): raw integer labels (multi-task encoding)
        all_paths (list): file paths, if provided
    """
    model.eval().to(device)
    hook_handle, feats_container = register_penultimate_hook(model)

    all_labels = []
    all_paths = []

    with torch.no_grad():
        for batch_data in data_loader:
            # Expect a single-element list of dict per batch.
            if not (isinstance(batch_data, list) and len(batch_data) == 1):
                raise ValueError("Expected a single-element list of dict per batch.")
            batch_dict = batch_data[0]
            images = batch_dict["data"].to(device)
            labels = batch_dict["labels"]
            file_paths = batch_dict.get("file_paths", [])

            # Forward pass; hook captures features.
            _ = model(images)

            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().tolist()  # shape (batch_size, 1)
            # Assume each label is a list/tuple; take the first element.
            labels = [l[0] for l in labels]

            all_labels.extend(labels)
            all_paths.extend(file_paths)

    embeddings = torch.cat(feats_container, dim=0).numpy()
    hook_handle.remove()

    return embeddings, all_labels, all_paths


def reduce_embeddings(embeddings, method="pca", n_components=2, **kwargs):
    """
    Dimensionality reduction on 'embeddings' using PCA, t-SNE, or UMAP.

    Args:
        embeddings (np.ndarray): shape (N, D)
        method (str): "pca", "tsne", or "umap"
        n_components (int): target dimension (e.g. 2 for plotting)
        **kwargs: additional arguments for the reduction algorithm

    Returns:
        reduced (np.ndarray): shape (N, n_components)
    """
    method = method.lower()
    if method == "pca":
        reducer = PCA(n_components=n_components, **kwargs)
        reduced = reducer.fit_transform(embeddings)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, **kwargs)
        reduced = reducer.fit_transform(embeddings)
    elif method == "umap":
        if umap is None:
            raise ImportError("UMAP not installed. Try pip install umap-learn.")
        reducer = umap.UMAP(n_components=n_components, **kwargs)
        reduced = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    return reduced


def prettify_label(label_str):
    """
    Replace underscores with spaces and convert to title case.
    E.g., "not_disaster" -> "Not Disaster"
    """
    return label_str.replace("_", " ").title()


def decode_task_labels(raw_labels, color_by_task, label_map_dict):
    """
    Decodes multi-task integer labels into textual labels for a single task,
    then converts them to title case.

    Args:
        raw_labels (list): List of raw integer labels.
        color_by_task (str): Task identifier to decode.
        label_map_dict (dict): Mapping from task to {class_id: class_label}.

    Returns:
        list: Decoded textual labels.
    """
    if color_by_task not in label_map_dict:
        return [str(lbl) for lbl in raw_labels]

    mapping_for_this_task = label_map_dict[color_by_task]
    decoded = []

    for c in raw_labels:
        if not isinstance(c, int):
            decoded.append(str(c))
            continue

        if color_by_task == "damage_severity":
            digit = c // 1000
        elif color_by_task == "humanitarian":
            digit = (c % 1000) // 100
        elif color_by_task == "informative":
            digit = (c % 100) // 10
        elif color_by_task == "disaster_types":
            digit = c % 10
        else:
            decoded.append(str(c))
            continue

        label_str = mapping_for_this_task.get(digit, f"Unknown({digit})")
        decoded.append(prettify_label(label_str))

    return decoded


def prettify_task_name(task_name):
    """
    Converts a task identifier like "disaster_types" into "Disaster Types".
    """
    return task_name.replace("_", " ").title()


def dynamic_alpha(p):
    """
    Given the proportion p of a class (in [0,1]), returns an alpha in [0.2, 1.0].

    If p <= 0.1  -> alpha = 1.0
    If p >= 0.65 -> alpha = 0.2
    Otherwise, linearly interpolate.
    """
    if p <= 0.1:
        return 1.0
    elif p >= 0.6:
        return 0.1
    else:
        frac = (p - 0.1) / (0.6 - 0.1)
        return 1.0 - 0.9 * frac


# -----------------------------------------------------------------------------
# Main Function: analyze_penultimate_embeddings
# -----------------------------------------------------------------------------


def analyze_penultimate_embeddings(
    data_dir: str = None,
    model_name: str = None,
    model_ckpt: str = None,
    tasks: list = None,
    reduction_method: str = "tsne",
    n_components: int = 2,
    batch_size: int = None,
    num_threads: int = None,
    device: str = None,
    random_seed: int = 42,
    label_mappings: dict = None,
):
    """
    Loads a single model and the validation set, extracts penultimate embeddings,
    reduces them to 2D (or n_components) with a specified method (e.g., t-SNE), then
    plots 2D scatterplots in a 2x2 grid, each coloured by a different task. Uses dynamic
    alpha per class so minority classes are not overshadowed.

    Args:
        data_dir (str): Path to the dataset. Defaults to DATA_DIR from config.
        model_name (str): Model architecture name (required).
        model_ckpt (str): Path to the model checkpoint (required).
        tasks (list): List of tasks to plot. Defaults to TASKS from config.
        reduction_method (str): "tsne", "pca", or "umap". Default is "tsne".
        n_components (int): Target dimension for reduction (typically 2).
        batch_size (int): Batch size for data loading. Defaults to BATCH_SIZE from config.
        num_threads (int): Number of data loading threads. Defaults to NUM_THREADS from config.
        device (str): Device for inference ("cuda" or "cpu"). Defaults to DEVICE from config.
        random_seed (int): Random seed for reproducibility. Default is 42.
        label_mappings (dict): Mapping for task labels. Defaults to LABEL_MAPPINGS from config.

    Returns:
        None. Displays scatterplots.
    """
    # Use centralized config values if parameters are not provided.
    from config import DATA_DIR, BATCH_SIZE, NUM_THREADS, DEVICE, LABEL_MAPPINGS, TASKS

    if data_dir is None:
        data_dir = DATA_DIR
    if batch_size is None:
        batch_size = BATCH_SIZE
    if num_threads is None:
        num_threads = NUM_THREADS
    if device is None:
        device = DEVICE
    if tasks is None:
        tasks = TASKS
    if label_mappings is None:
        label_mappings = LABEL_MAPPINGS

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Get the validation DALI loader (phase "val" is hard-coded here)
    _, val_tuple, _ = get_dali_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=0,
        phase="val",
    )
    val_loader, _, _ = val_tuple

    # Load the model and extract penultimate embeddings.
    model = load_model(model_name, model_ckpt)
    embeddings, raw_labels, _ = extract_penultimate_embeddings(
        model, val_loader, device
    )

    # Reset DALI loader if needed.
    try:
        val_loader.reset()
    except Exception:
        pass

    if n_components != 2:
        raise NotImplementedError("This example only supports 2D scatterplots.")

    # Perform dimensionality reduction.
    reduced_2d = reduce_embeddings(
        embeddings,
        method=reduction_method,
        n_components=n_components,
        random_state=(random_seed if reduction_method.lower() == "tsne" else None),
    )

    # Create a 2x2 grid of scatterplots.
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flat

    for ax, task_name in zip(axes, tasks):
        # Decode labels for this task.
        decoded_labels = decode_task_labels(raw_labels, task_name, label_mappings)

        # Build a DataFrame for plotting.
        df = pd.DataFrame(
            {"x": reduced_2d[:, 0], "y": reduced_2d[:, 1], "label": decoded_labels}
        )

        # Count frequency of each label.
        counts = df["label"].value_counts()
        total = len(df)
        unique_labels = sorted(counts.index)

        # Create a colour palette.
        color_list = sns.color_palette("tab10", n_colors=len(unique_labels))
        label_to_color = dict(zip(unique_labels, color_list))

        # Plot each class separately, ordering so minority classes appear on top.
        sorted_labels = sorted(unique_labels, key=lambda lbl: counts[lbl], reverse=True)
        for i, lbl in enumerate(sorted_labels):
            subset = df[df["label"] == lbl]
            p = counts[lbl] / total  # proportion for this class
            alpha_val = dynamic_alpha(p)
            z_val = len(sorted_labels) - i  # higher zorder for minority classes
            ax.scatter(
                subset["x"],
                subset["y"],
                facecolor=label_to_color[lbl],
                edgecolor="none",
                alpha=alpha_val,
                s=10,
                zorder=z_val,
                label=f"{lbl} (p={p:.2f})",
            )

        ax.set_title(
            f"{model_name}: {reduction_method} by {prettify_task_name(task_name)}",
            fontsize=14,
        )
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        legend = ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.03, 0.97),
            framealpha=0.6,
            markerscale=2.0,
        )
        for txt in legend.get_texts():
            txt.set_fontsize(10)

    plt.tight_layout()
    plt.show()


def analyze_penultimate_embeddings_synthetic(
    data_dir: str = None,
    model_name: str = None,
    model_ckpt: str = None,
    tasks: list = None,
    reduction_method: str = "tsne",
    n_components: int = 2,
    batch_size: int = None,
    num_threads: int = None,
    device: str = None,
    random_seed: int = 42,
    label_mappings: dict = None,
):
    """
    Loads a model and compares original vs synthetic data in a visualization.
    Original data is shown with low opacity and synthetic data with full opacity.
    """
    # Use centralized config values if parameters are not provided.
    from config import DATA_DIR, BATCH_SIZE, NUM_THREADS, DEVICE, LABEL_MAPPINGS, TASKS
    import pandas as pd
    import os

    if data_dir is None:
        data_dir = DATA_DIR
    if batch_size is None:
        batch_size = BATCH_SIZE
    if num_threads is None:
        num_threads = NUM_THREADS
    if device is None:
        device = DEVICE
    if tasks is None:
        tasks = TASKS
    if label_mappings is None:
        label_mappings = LABEL_MAPPINGS

    def map_label_to_int(label, label_mapping_dict):
        """
        Converts a single textual label (e.g. "mild") into an integer ID (e.g. 1)
        using the provided dictionary mapping.
        """
        for key, value in label_mapping_dict.items():
            if label == value:
                return key
        return None

    def map_labels(row, label_mappings):
        """
        Combines four task labels into a single integer by placing each task's integer ID
        in a different 'decimal place' of a base-10 number.
        """
        damage = (
            map_label_to_int(row["damage_severity"], label_mappings["damage_severity"])
            * 1000
        )
        humanitarian = (
            map_label_to_int(row["humanitarian"], label_mappings["humanitarian"]) * 100
        )
        informative = (
            map_label_to_int(row["informative"], label_mappings["informative"]) * 10
        )
        disaster = map_label_to_int(
            row["disaster_types"], label_mappings["disaster_types"]
        )
        return damage + humanitarian + informative + disaster

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    augmented_tsv = "/home/evammun/Thesis/Dataset/MEDIC_train_augmented.tsv"
    augmented_df = pd.read_csv(augmented_tsv, sep="\t")

    # Split into original and synthetic dataframes
    synthetic_df = augmented_df[
        augmented_df["event_name"].str.contains("synthetic", case=False, na=False)
    ]
    original_df = augmented_df[
        ~augmented_df["event_name"].str.contains("synthetic", case=False, na=False)
    ]

    # Only keep required columns
    required_columns = [
        "image_path",
        "damage_severity",
        "informative",
        "humanitarian",
        "disaster_types",
    ]

    original_df = original_df[required_columns]
    synthetic_df = synthetic_df[required_columns]

    # Prepare original dataframe for DALI
    original_df["full_img_path"] = original_df["image_path"].apply(
        lambda x: os.path.join(data_dir, x)
    )
    original_df["combined_label"] = original_df.apply(
        lambda row: map_labels(row, label_mappings), axis=1
    )

    # Prepare synthetic dataframe for DALI
    synthetic_df["full_img_path"] = synthetic_df["image_path"].apply(
        lambda x: os.path.join(data_dir, x)
    )
    synthetic_df["combined_label"] = synthetic_df.apply(
        lambda row: map_labels(row, label_mappings), axis=1
    )

    # Load the model
    model = load_model(model_name, model_ckpt)
    model.eval().to(device)

    # Import the DALI loader function
    from src.data.load_dali import get_dali_loader

    # Process original samples
    orig_loader, _, _ = get_dali_loader(
        original_df, batch_size, num_threads, device_id=0, phase="train"
    )
    orig_embeddings, orig_labels, _ = extract_penultimate_embeddings(
        model, orig_loader, device
    )

    # Process synthetic samples
    syn_loader, _, _ = get_dali_loader(
        synthetic_df, batch_size, num_threads, device_id=0, phase="train"
    )
    syn_embeddings, syn_labels, _ = extract_penultimate_embeddings(
        model, syn_loader, device
    )

    # Combine embeddings and labels
    all_embeddings = np.vstack([orig_embeddings, syn_embeddings])
    all_labels = orig_labels + syn_labels
    all_is_synthetic = [False] * len(orig_labels) + [True] * len(syn_labels)

    # Perform dimensionality reduction on combined data
    reduced_2d = reduce_embeddings(
        all_embeddings,
        method=reduction_method,
        n_components=n_components,
        random_state=(random_seed if reduction_method.lower() == "tsne" else None),
    )

    # Create a 2x2 grid of scatterplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flat

    for ax_idx, (ax, task_name) in enumerate(zip(axes, tasks)):
        # Decode labels for this task
        decoded_labels = decode_task_labels(all_labels, task_name, label_mappings)

        # Build a DataFrame for plotting
        df = pd.DataFrame(
            {
                "x": reduced_2d[:, 0],
                "y": reduced_2d[:, 1],
                "label": decoded_labels,
                "is_synthetic": all_is_synthetic,
            }
        )

        # Count frequency of each label
        counts = df["label"].value_counts()
        total = len(df)
        unique_labels = sorted(counts.index)

        # Create a colour palette
        color_list = sns.color_palette("tab10", n_colors=len(unique_labels))
        label_to_color = dict(zip(unique_labels, color_list))

        # First plot all original points
        for lbl in unique_labels:
            orig_subset = df[(df["label"] == lbl) & (~df["is_synthetic"])]
            if not orig_subset.empty:
                p = len(orig_subset) / len(original_df)
                ax.scatter(
                    orig_subset["x"],
                    orig_subset["y"],
                    facecolor=label_to_color[lbl],
                    edgecolor="none",
                    alpha=0.2,  # Low opacity for original
                    s=10,
                    marker="o",
                    label=f"{lbl} (p={p:.2f})",
                )

        # Then plot all synthetic points with same marker but full opacity
        for lbl in unique_labels:
            syn_subset = df[(df["label"] == lbl) & (df["is_synthetic"])]
            if not syn_subset.empty:
                ax.scatter(
                    syn_subset["x"],
                    syn_subset["y"],
                    facecolor=label_to_color[lbl],
                    edgecolor="none",  # No edge to match original points
                    alpha=1.0,  # Full opacity
                    s=10,  # Same size
                    marker="o",  # Same marker (circle)
                    label=f"{lbl} (p={p:.2f})",
                )

        ax.set_title(
            f"{model_name}: {reduction_method} visualisation of penultimate layer embeddings for {prettify_task_name(task_name)}",
            fontsize=14,
        )
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

        legend = ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.03, 0.97),
            framealpha=0.6,
            markerscale=2.0,
        )
        for txt in legend.get_texts():
            txt.set_fontsize(10)

    plt.tight_layout()
    plt.show()


def analyze_penultimate_embeddings_synthetic_density(
    data_dir: str = None,
    model_name: str = None,
    model_ckpt: str = None,
    tasks: list = None,
    reduction_method: str = "tsne",
    n_components: int = 2,
    batch_size: int = None,
    num_threads: int = None,
    device: str = None,
    random_seed: int = 42,
    label_mappings: dict = None,
):
    """
    Loads a model and compares original vs synthetic data using density-based visualization
    of penultimate-layer embeddings. Specifically:
      1) It reads in an augmented TSV containing original and synthetic images,
      2) Splits them into separate dataframes,
      3) Extracts embeddings from the penultimate layer of a given model,
      4) Projects these embeddings down to 2D (e.g. via t-SNE or PCA),
      5) Uses Seaborn's KDE plotting to show distribution overlaps for original (filled) vs
         synthetic (dashed, no fill) data, colour-coded by each task's classes.

    Parameters
    ----------
    data_dir : str, optional
        Directory path where image files are located. Uses config.DATA_DIR by default.
    model_name : str, optional
        Name of the model to load.
    model_ckpt : str, optional
        Path or identifier for the model checkpoint.
    tasks : list, optional
        List of tasks to visualise (e.g. ["damage_severity", "informative", ...]).
        Defaults to config.TASKS if None.
    reduction_method : str, optional
        Dimensionality reduction method: "tsne" or "pca". Defaults to "tsne".
    n_components : int, optional
        Dimensionality of the reduced space (2 for 2D visualisations).
    batch_size : int, optional
        Batch size for DALI loading. Defaults to config.BATCH_SIZE if None.
    num_threads : int, optional
        Number of threads for DALI. Defaults to config.NUM_THREADS if None.
    device : str, optional
        The compute device ("cuda" or "cpu"). Defaults to config.DEVICE if None.
    random_seed : int, optional
        RNG seed for reproducibility. Defaults to 42.
    label_mappings : dict, optional
        A dictionary that maps integer class indices to string labels for each task.
        Defaults to config.LABEL_MAPPINGS if None.

    Returns
    -------
    None
        Displays the 2x2 grid of density plots for each task.
    """
    # -- Imports and config fallback --
    from config import DATA_DIR, BATCH_SIZE, NUM_THREADS, DEVICE, LABEL_MAPPINGS, TASKS
    import pandas as pd
    import os
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    import numpy as np

    if data_dir is None:
        data_dir = DATA_DIR
    if batch_size is None:
        batch_size = BATCH_SIZE
    if num_threads is None:
        num_threads = NUM_THREADS
    if device is None:
        device = DEVICE
    if tasks is None:
        tasks = TASKS
    if label_mappings is None:
        label_mappings = LABEL_MAPPINGS

    # -- Helper functions for label handling --
    def map_label_to_int(label, label_mapping_dict):
        """
        Converts a single textual label (e.g. "mild") into an integer ID (e.g. 1)
        using the provided dictionary mapping.
        """
        for key, value in label_mapping_dict.items():
            if label == value:
                return key
        return None

    def map_labels(row, label_mappings):
        """
        Combines four task labels into a single integer by placing each task's integer ID
        in a different 'decimal place' of a base-10 number, e.g. (damage*1000 + hum*100 + ...).
        This is just a convenient way to keep a single 'combined' label for all tasks.
        """
        damage = (
            map_label_to_int(row["damage_severity"], label_mappings["damage_severity"])
            * 1000
        )
        humanitarian = (
            map_label_to_int(row["humanitarian"], label_mappings["humanitarian"]) * 100
        )
        informative = (
            map_label_to_int(row["informative"], label_mappings["informative"]) * 10
        )
        disaster = map_label_to_int(
            row["disaster_types"], label_mappings["disaster_types"]
        )
        return damage + humanitarian + informative + disaster

    # -- Set random seeds --
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # -- Load the augmented TSV and separate original vs synthetic --
    augmented_tsv = "/home/evammun/Thesis/Dataset/MEDIC_train_augmented.tsv"
    augmented_df = pd.read_csv(augmented_tsv, sep="\t")

    synthetic_df = augmented_df[
        augmented_df["event_name"].str.contains("synthetic", case=False, na=False)
    ]
    original_df = augmented_df[
        ~augmented_df["event_name"].str.contains("synthetic", case=False, na=False)
    ]

    # -- Keep only the required columns --
    required_columns = [
        "image_path",
        "damage_severity",
        "informative",
        "humanitarian",
        "disaster_types",
    ]
    original_df = original_df[required_columns]
    synthetic_df = synthetic_df[required_columns]

    # -- Add full paths and combined labels --
    original_df["full_img_path"] = original_df["image_path"].apply(
        lambda x: os.path.join(data_dir, x)
    )
    original_df["combined_label"] = original_df.apply(
        lambda row: map_labels(row, label_mappings), axis=1
    )

    synthetic_df["full_img_path"] = synthetic_df["image_path"].apply(
        lambda x: os.path.join(data_dir, x)
    )
    synthetic_df["combined_label"] = synthetic_df.apply(
        lambda row: map_labels(row, label_mappings), axis=1
    )

    # -- Load the model (must define or import these functions) --
    model = load_model(model_name, model_ckpt)
    model.eval().to(device)

    from src.data.load_dali import get_dali_loader

    # -- Extract embeddings for original samples --
    orig_loader, _, _ = get_dali_loader(
        original_df, batch_size, num_threads, device_id=0, phase="train"
    )
    orig_embeddings, orig_labels, _ = extract_penultimate_embeddings(
        model, orig_loader, device
    )

    # -- Extract embeddings for synthetic samples --
    syn_loader, _, _ = get_dali_loader(
        synthetic_df, batch_size, num_threads, device_id=0, phase="train"
    )
    syn_embeddings, syn_labels, _ = extract_penultimate_embeddings(
        model, syn_loader, device
    )

    # -- Combine for joint dimension reduction --
    all_embeddings = np.vstack([orig_embeddings, syn_embeddings])
    all_labels = orig_labels + syn_labels
    all_is_synthetic = [False] * len(orig_labels) + [True] * len(syn_labels)

    # -- Reduce to 2D (t-SNE or PCA) --
    reduced_2d = reduce_embeddings(
        all_embeddings,
        method=reduction_method,
        n_components=n_components,
        random_state=(random_seed if reduction_method.lower() == "tsne" else None),
    )

    # -- Create a 2x2 grid of density plots --
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flat

    # --------------------------------------------------------------------------
    #  STYLE: No grid, remove spines, use a plain white background.
    # --------------------------------------------------------------------------

    import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap

    # Increase the global font size
    plt.rcParams.update({"font.size": 14})

    sns.set_style("white")
    sns.despine(left=True, bottom=True)

    # New palette from your second image
    new_palette = [
        "#56641a",  # Fern frond (dark olive green)
        "#c0affb",  # Perfume (light purple/lavender)
        "#e6a176",  # Apricot (light orange/peach)
        "#00678a",  # Orient (dark blue/teal)
        "#984464",  # Vin rouge (burgundy/wine red)
        "#5eccab",  # Downy (mint green/teal)
        "#808080",  # Gray (trolley gray)
    ]

    # -- Define colour maps for each task --
    color_palettes = {
        # damage_severity: 3 classes - using maximally different colors
        "damage_severity": ListedColormap(
            [
                new_palette[4],
                new_palette[3],
                new_palette[2],
            ]  # Burgundy, Dark blue, Olive green
        ),
        # informative: 2 classes (binary) - high contrast pair
        "informative": ListedColormap(
            [new_palette[4], new_palette[3]]  # Burgundy, Dark blue
        ),
        # humanitarian: 4 classes
        "humanitarian": ListedColormap(
            [
                new_palette[4],
                new_palette[0],
                new_palette[3],
                new_palette[2],
            ]  # Burgundy, Olive green, Dark blue, Apricot
        ),
        # disaster_types: 7 classes - using all colors
        "disaster_types": ListedColormap(
            [
                new_palette[4],  # Burgundy (Vin rouge)
                new_palette[3],  # Dark blue (Orient)
                new_palette[0],  # Olive green (Fern frond)
                new_palette[2],  # Apricot
                new_palette[1],  # Light purple (Perfume)
                new_palette[5],  # Mint green (Downy)
                new_palette[6],  # Gray
            ]
        ),
    }

    # Enhance the title formatting
    def prettify_task_name(task_name):
        """Convert snake_case task names to Title Case with spaces."""
        return task_name.replace("_", " ").title()

    for ax_idx, (ax, task_name) in enumerate(zip(axes, tasks)):
        # Decode labels for this task (you must define decode_task_labels, prettify_task_name)
        decoded_labels = decode_task_labels(all_labels, task_name, label_mappings)

        # Build a DataFrame for plotting
        df = pd.DataFrame(
            {
                "x": reduced_2d[:, 0],
                "y": reduced_2d[:, 1],
                "label": decoded_labels,
                "is_synthetic": all_is_synthetic,
            }
        )

        # Count unique labels to adjust threshold if necessary
        counts = df["label"].value_counts()
        unique_labels = sorted(counts.index)
        n_categories = len(unique_labels)

        # Choose threshold and contour levels
        threshold = 0.05
        levels = 3
        if n_categories > 5:  # For tasks with many categories
            threshold = 0.1
            levels = 2

        # Select this task's colormap
        cmap = color_palettes[task_name]

        # We will store just the class colours in a separate list for legend
        class_legend_handles = []

        # ----------------------------------------------------------------------
        # Plot the distributions for each class:
        #   * Original = filled contour, no outline
        #   * Synthetic = dashed contour, no fill
        # ----------------------------------------------------------------------
        for i, lbl in enumerate(unique_labels):
            # Normalised position in the colormap
            color_pos = i / max(1, len(unique_labels) - 1)
            base_color = cmap(color_pos)

            # Build a patch handle for the legend (one per class)
            # (We won't add original/synthetic text here; just the colour.)
            class_patch = mpatches.Patch(color=base_color, label=lbl, alpha=0.5)
            class_legend_handles.append(class_patch)

            # Subsets
            orig_subset = df[(df["label"] == lbl) & (~df["is_synthetic"])]
            syn_subset = df[(df["label"] == lbl) & (df["is_synthetic"])]

            # Original data: fill-only
            if len(orig_subset) > 20:
                sns.kdeplot(
                    x=orig_subset["x"],
                    y=orig_subset["y"],
                    ax=ax,
                    fill=True,
                    alpha=0.25,
                    levels=levels,
                    color=base_color,
                    thresh=threshold,
                )

            # Synthetic data: dashed outline (no fill)
            if len(syn_subset) > 20:
                sns.kdeplot(
                    x=syn_subset["x"],
                    y=syn_subset["y"],
                    ax=ax,
                    fill=False,
                    alpha=1,
                    levels=levels,
                    color=base_color,
                    linewidths=2,
                    linestyles="--",
                    thresh=threshold,
                )

        # ----------------------------------------------------------------------
        # Build a separate pair of handles for "Original" vs "Synthetic"
        # (showing one solid line and one dashed line).
        # We use a black line to indicate style only, ignoring fill colour.
        # ----------------------------------------------------------------------
        original_handle = Line2D(
            [0], [0], color="black", lw=3, linestyle="-", label="Original (filled)"
        )
        synthetic_handle = Line2D(
            [0], [0], color="black", lw=3, linestyle="--", label="Synthetic (dashed)"
        )  # Using -- instead of : for clarity

        # ----------------------------------------------------------------------
        # Combine the class-colour patches + line-style handles in one legend:
        #   - We only need one patch per class, plus these two lines for style.
        # ----------------------------------------------------------------------
        # We must remove duplicates from class_legend_handles
        # (since we added one patch per label in the loop).
        seen = set()
        unique_class_handles = []
        for patch in class_legend_handles:
            if patch.get_label() not in seen:
                unique_class_handles.append(patch)
                seen.add(patch.get_label())

        # Combine them all
        all_handles = unique_class_handles + [original_handle, synthetic_handle]

        # Show x-axis labels for all plots with smaller font
        ax.set_xlabel("Dim 1", fontsize=10)

        # Show y-axis labels for all plots with smaller font
        ax.set_ylabel("Dim 2", fontsize=10)

        # Make tick labels smaller
        ax.tick_params(axis="both", which="major", labelsize=8)

        # Place legend inside the plot to save space
        ax.legend(
            handles=all_handles,
            loc="upper center",  # Position the legend relative to the bounding box anchor
            bbox_to_anchor=(
                0.5,
                -0.1,
            ),  # (x, y) coordinates: x=0.5 centres it, y=-0.1 moves it below the plot; adjust as needed
            frameon=False,  # Remove the box around the legend
            ncol=2,  # Organise legend entries in two columns
            fontsize=10,  # Use a smaller font size
        )

        # Simplified titles without bold
        ax.set_title(
            f"{prettify_task_name(task_name)}",
            fontsize=12,  # Smaller title
            fontweight="normal",  # Not bold
            pad=5,  # Less padding
        )

        # Remove internal spines
        sns.despine(ax=ax, left=True, bottom=True)

    # Make sure tight_layout happens BEFORE subplots_adjust
    plt.tight_layout(pad=0.5)

    # IMPORTANT: Apply these adjustments AFTER tight_layout
    plt.subplots_adjust(wspace=-0, hspace=0.3)  # Negative values to force overlap

    plt.show()
