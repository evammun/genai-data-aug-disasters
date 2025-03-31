"""
model_analyser.py

Module for analysing multi-task classification models. This module includes functions
for plotting confusion matrices, computing evaluation metrics (both at the task and class levels),
displaying aggregated performance tables, and performing cross-task error analysis.
Configuration defaults (e.g. DATA_DIR, MODEL_PATHS, BATCH_SIZE, NUM_THREADS, LABEL_MAPPINGS)
are imported from config.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)
import os
import pandas as pd
from IPython.display import display, HTML
from collections import defaultdict
from typing import Dict, List, Tuple

from ..core.model_tester import load_model, test_model

# Import centralized configuration defaults.
from config import DATA_DIR, MODEL_PATHS, BATCH_SIZE, NUM_THREADS, LABEL_MAPPINGS

# =============================================================================
# plot_confusion_matrices
# =============================================================================


def plot_confusion_matrices(true_labels_dict, predictions_dict, save_dir=None):
    """
    Plot confusion matrices for multiple classification tasks in a single row.

    This function creates normalized confusion matrices for multiple tasks,
    displaying them in a horizontal layout with clean formatting and proper labeling.

    Parameters
    ----------
    true_labels_dict : dict
        Dictionary mapping task names to lists/arrays of true labels.
    predictions_dict : dict
        Dictionary mapping task names to lists/arrays of predicted labels.
    save_dir : str, optional
        Directory path to save the confusion matrix plot. If None, the plot is only displayed.
    """
    # Define shortened class labels for each task.
    task_names = {
        "disaster_types": ["quake", "fire", "flood", "hurr.", "land.", "none", "other"],
        "informative": ["not inf", "inf"],
        "humanitarian": ["injured", "infra", "not hum", "rescue"],
        "damage_severity": ["none", "mild", "severe"],
    }

    # Dictionary to convert task names to proper case titles.
    task_titles = {
        "disaster_types": "Disaster Types",
        "informative": "Informative",
        "humanitarian": "Humanitarian",
        "damage_severity": "Damage Severity",
    }

    plt.style.use("default")
    sns.set_theme()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Create figure with four subplots in a row.
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for idx, (task, ax) in enumerate(zip(true_labels_dict, axes)):
        cm = confusion_matrix(true_labels_dict[task], predictions_dict[task])
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        custom_norm = np.zeros_like(cm_normalized)
        for i in range(len(custom_norm)):
            for j in range(len(custom_norm[i])):
                if i == j:
                    custom_norm[i, j] = cm_normalized[i, j]
                else:
                    custom_norm[i, j] = -cm_normalized[i, j]
        annotations = [
            [
                (
                    f'.{str(x).split(".")[1][:2]}'
                    if 0 < x < 1
                    else "1.00" if x == 1 else "0"
                )
                for x in row
            ]
            for row in cm_normalized
        ]
        sns.heatmap(
            custom_norm,
            annot=annotations,
            fmt="",
            cmap=sns.diverging_palette(10, 240, s=100, l=40, n=9),
            xticklabels=task_names[task],
            yticklabels=task_names[task],
            ax=ax,
            cbar=False,
            annot_kws={"fontsize": 12},
            vmin=-0.5,
            center=0,
            vmax=1.0,
        )
        ax.set_title(task_titles[task])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

    plt.tight_layout(w_pad=1.5, h_pad=1.5)
    if save_dir:
        plt.savefig(
            os.path.join(save_dir, "confusion_matrices.png"),
            bbox_inches="tight",
            dpi=300,
        )
    plt.show()
    plt.close()


# =============================================================================
# format_label
# =============================================================================


def format_label(label: str):
    """Convert snake_case to Title Case."""
    return " ".join(word.capitalize() for word in label.replace("_", " ").split())


# =============================================================================
# calculate_metrics
# =============================================================================


def calculate_metrics(true_labels_dict, predictions_dict, label_mappings: Dict = None):
    """
    Calculate comprehensive evaluation metrics for multi-task classification.

    Computes both task-level (weighted averages) and class-level (binary) metrics.

    Parameters
    ----------
    true_labels_dict : dict
        Mapping from task names to lists/arrays of true labels.
    predictions_dict : dict
        Mapping from task names to lists/arrays of predicted labels.
    label_mappings : dict, optional
        Mapping from task names to {class_id: class_name} dictionaries.
        Defaults to centralized LABEL_MAPPINGS if not provided.

    Returns
    -------
    dict
        Nested dictionary containing metrics organized by task and class.
    """
    if label_mappings is None:
        label_mappings = LABEL_MAPPINGS

    metrics = {}
    for task, classes in label_mappings.items():
        true_labels = np.array(true_labels_dict[task]).flatten()
        predictions = np.array(predictions_dict[task])
        task_accuracy = accuracy_score(true_labels, predictions)
        task_precision, task_recall, task_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="weighted"
        )
        metrics[task] = {
            "accuracy": task_accuracy * 100,
            "f1_score": task_f1 * 100,
            "precision": task_precision * 100,
            "recall": task_recall * 100,
            "classes": {},
        }
        for class_id, class_name in classes.items():
            class_true = true_labels == class_id
            class_pred = predictions == class_id
            # Calculate class-level accuracy (binary classification accuracy)
            class_accuracy = accuracy_score(class_true, class_pred)
            class_precision, class_recall, class_f1, _ = (
                precision_recall_fscore_support(
                    class_true, class_pred, average="binary"
                )
            )
            metrics[task]["classes"][class_name] = {
                "accuracy": class_accuracy * 100,  # Add class-level accuracy
                "f1_score": class_f1 * 100,
                "precision": class_precision * 100,
                "recall": class_recall * 100,
            }
    return metrics


# =============================================================================
# display_metrics_table
# =============================================================================


def display_metrics_table(metrics, label_mappings: Dict = None):
    """Display metrics in a formatted table using Pandas Styler."""
    if label_mappings is None:
        label_mappings = LABEL_MAPPINGS

    data = []
    for task, task_metrics in metrics.items():
        data.append(
            [
                format_label(task),
                f"{task_metrics['accuracy']:.1f}%",
                f"{task_metrics['f1_score']:.1f}%",
                f"{task_metrics['precision']:.1f}%",
                f"{task_metrics['recall']:.1f}%",
            ]
        )
        for class_name, class_metrics in task_metrics["classes"].items():
            data.append(
                [
                    f"    {format_label(class_name)}",
                    "",
                    f"{class_metrics['f1_score']:.1f}%",
                    f"{class_metrics['precision']:.1f}%",
                    f"{class_metrics['recall']:.1f}%",
                ]
            )
    df = pd.DataFrame(
        data, columns=["Task/Class", "Accuracy", "F1 Score", "Precision", "Recall"]
    )
    styled_df = df.style.set_properties(**{"text-align": "center"})
    styled_df = styled_df.set_table_styles(
        [
            {
                "selector": "th",
                "props": [("text-align", "center"), ("font-weight", "bold")],
            },
            {"selector": "td", "props": [("text-align", "center")]},
            {"selector": "td:nth-child(1)", "props": [("text-align", "left")]},
            {"selector": ".index_name", "props": "display: none;"},
            {"selector": ".index_col", "props": "display: none;"},
        ]
    )
    task_rows = [i for i, row in enumerate(data) if not row[0].startswith("    ")]
    styled_df = styled_df.set_properties(
        subset=pd.IndexSlice[task_rows, :], **{"font-weight": "bold"}
    )
    return styled_df


# =============================================================================
# analyse_all_models
# =============================================================================


def analyse_all_models(
    data_dir: str = None,
    batch_size: int = None,
    num_threads: int = None,
    models_config: list = None,
):
    """
    Analyse multiple models and display their metrics side by side.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing the data. Defaults to DATA_DIR from config.
    batch_size : int, optional
        Batch size for testing. Defaults to BATCH_SIZE from config.
    num_threads : int, optional
        Number of threads for data loading. Defaults to NUM_THREADS from config.
    models_config : list, optional
        List of tuples (model_name, model_path). Defaults to list(MODEL_PATHS.items()).

    Returns
    -------
    dict
        Dictionary containing metrics and raw results for each model.
    """
    from config import DATA_DIR, BATCH_SIZE, NUM_THREADS, MODEL_PATHS

    if data_dir is None:
        data_dir = DATA_DIR
    if batch_size is None:
        batch_size = BATCH_SIZE
    if num_threads is None:
        num_threads = NUM_THREADS
    if models_config is None:
        models_config = list(MODEL_PATHS.items())

    all_results = {}
    styled_tables = []

    for model_name, model_path in models_config:

        model = load_model(model_name, model_path)
        true_labels, predictions, _ = test_model(
            model, data_dir, batch_size, num_threads
        )
        metrics_result = calculate_metrics(true_labels, predictions, LABEL_MAPPINGS)
        all_results[model_name] = {
            "metrics": metrics_result,
            "true_labels": true_labels,
            "predictions": predictions,
        }
        styled_df = display_metrics_table(metrics_result, LABEL_MAPPINGS)
        styled_tables.append((model_name, styled_df))

    html_content = """
        <div style="display: flex; justify-content: space-between; width: 100%;">
    """
    for model_name, styled_df in styled_tables:
        html_content += f"""
            <div style="flex: 1; margin: 10px;">
                <h3 style="text-align: center;">{model_name}</h3>
                {styled_df.to_html(index=False)}
            </div>
        """
    html_content += "</div>"
    display(HTML(html_content))
    return all_results


# =============================================================================
# display_comparative_table
# =============================================================================


def display_comparative_table(all_results):
    """
    Create a comparative table showing accuracy and F1 scores for all models side by side.

    Parameters
    ----------
    all_results : dict
        Dictionary containing metrics for each model.

    Returns
    -------
    Styler
        A Pandas Styler object for the comparative table.
    """
    tasks = []
    class_dict = {}
    for model_name, results in all_results.items():
        for task, task_metrics in results["metrics"].items():
            if task not in tasks:
                tasks.append(task)
                class_dict[task] = list(task_metrics["classes"].keys())

    data = []
    model_names = list(all_results.keys())

    for task in tasks:
        row = [format_label(task)]
        for model_name in model_names:
            row.append(f"{all_results[model_name]['metrics'][task]['accuracy']:.1f}%")
        for model_name in model_names:
            row.append(f"{all_results[model_name]['metrics'][task]['f1_score']:.1f}%")
        data.append(row)
        for class_name in class_dict[task]:
            row = [f"    {format_label(class_name)}"]
            # Add class-level accuracy values for each model
            for model_name in model_names:
                class_accuracy = all_results[model_name]["metrics"][task]["classes"][
                    class_name
                ]["accuracy"]
                row.append(f"{class_accuracy:.1f}%")
            # Add class-level F1 score values for each model
            for model_name in model_names:
                f1_score = all_results[model_name]["metrics"][task]["classes"][
                    class_name
                ]["f1_score"]
                row.append(f"{f1_score:.1f}%")
            data.append(row)

    top_level = [""] + ["Accuracy"] * len(model_names) + ["F1 Score"] * len(model_names)
    bottom_level = ["Task/Class"] + model_names + model_names
    columns = pd.MultiIndex.from_arrays([top_level, bottom_level])
    df = pd.DataFrame(data, columns=columns)

    def percentage_to_float(x):
        try:
            return float(x.rstrip("%"))
        except (AttributeError, ValueError):
            return None

    def highlight_max(s):
        values = [percentage_to_float(v) for v in s]
        if not any(v is not None for v in values):
            return [""] * len(s)
        max_val = max((v for v in values if v is not None), default=None)
        return [
            "color: #2E8B57" if percentage_to_float(v) == max_val else "" for v in s
        ]

    styled_df = df.style.set_properties(**{"text-align": "center"})
    n_models = len(model_names)
    acc_cols = df.columns[1 : n_models + 1]
    f1_cols = df.columns[n_models + 1 :]
    styled_df = styled_df.apply(highlight_max, axis=1, subset=acc_cols)
    styled_df = styled_df.apply(highlight_max, axis=1, subset=f1_cols)
    separator_col = n_models + 2
    styled_df = styled_df.set_table_styles(
        [
            {
                "selector": "th.col_heading.level0",
                "props": [
                    ("text-align", "center"),
                    ("font-weight", "bold"),
                    ("color", "black"),
                    ("background-color", "#f0f0f0"),
                ],
            },
            {
                "selector": "th.col_heading.level1",
                "props": [
                    ("text-align", "center"),
                    ("font-weight", "bold"),
                    ("color", "black"),
                    ("background-color", "#f8f8f8"),
                ],
            },
            {"selector": "td", "props": [("text-align", "center")]},
            {"selector": "td:nth-child(1)", "props": [("text-align", "left")]},
            {"selector": ".index_name", "props": "display: none;"},
            {"selector": ".index_col", "props": "display: none;"},
            {
                "selector": f"td:nth-child({separator_col})",
                "props": [("border-right", "2px solid black")],
            },
            {
                "selector": f"th:nth-child({separator_col})",
                "props": [("border-right", "2px solid black")],
            },
        ]
    )
    task_rows = [i for i, row in enumerate(data) if not row[0].startswith("    ")]
    styled_df = styled_df.set_properties(
        subset=pd.IndexSlice[task_rows, :], **{"font-weight": "bold"}
    )
    return styled_df


# =============================================================================
# analyze_cross_task_confusion
# =============================================================================


def analyze_cross_task_confusion(
    true_labels_dict: Dict[str, List], predictions_dict: Dict[str, List]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze error correlations between individual class labels across all tasks.

    This function computes binary error masks for each class in each task (using the centralized
    LABEL_MAPPINGS), then combines these into task-level error indicators for the purpose of
    detecting error patterns across tasks.

    Parameters
    ----------
    true_labels_dict : dict
        Mapping from task names to lists/arrays of true labels.
    predictions_dict : dict
        Mapping from task names to lists/arrays of predicted labels.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        A DataFrame of error correlations and a dictionary of error patterns.
    """
    # Create binary error masks for each individual class.
    error_masks = {}
    for task in true_labels_dict:
        true_array = np.array(true_labels_dict[task]).squeeze()
        pred_array = np.array(predictions_dict[task]).squeeze()
        for class_val in LABEL_MAPPINGS[task]:
            key = f"{task}_{LABEL_MAPPINGS[task][class_val]}"
            true_mask = (true_array == class_val).astype(int)
            pred_mask = (pred_array == class_val).astype(int)
            error_masks[key] = (true_mask != pred_mask).astype(int)

    # Compute task-level error masks by combining class error masks via logical OR.
    task_error = {}
    for task in LABEL_MAPPINGS:
        # Get masks for all classes in the task.
        masks = [
            error_masks[f"{task}_{LABEL_MAPPINGS[task][class_val]}"]
            for class_val in LABEL_MAPPINGS[task]
        ]
        task_error[task] = np.any(np.array(masks), axis=0).astype(int)

    # Calculate error correlations between the task-level error masks.
    error_df = pd.DataFrame(task_error)
    error_correlation = error_df.corr()

    # Compute error patterns.
    error_patterns = defaultdict(int)
    n_samples = len(next(iter(true_labels_dict.values())))
    for i in range(n_samples):
        pattern = []
        for task in LABEL_MAPPINGS:
            if task_error[task][i]:
                true_val = true_labels_dict[task][i]
                pred_val = predictions_dict[task][i]
                if hasattr(true_val, "item"):
                    true_val = true_val.item()
                if hasattr(pred_val, "item"):
                    pred_val = pred_val.item()
                pattern.append(f"{task}:{true_val}->{pred_val}")
        if len(pattern) > 1:
            error_patterns[tuple(sorted(pattern))] += 1

    return error_correlation, error_patterns


# =============================================================================
# plot_cross_task_errors
# =============================================================================


def plot_cross_task_errors(
    models_results: List[Tuple[str, Dict, Dict]], save_dir: str = None
):
    """
    Plot cross-task error correlation matrices for multiple models side by side.

    Parameters:
    -----------
    models_results : List[Tuple[str, Dict, Dict]]
        List of tuples containing (model_name, true_labels_dict, predictions_dict)
    save_dir : str, optional
        Directory to save visualization outputs
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Create figure with subplots for each model
    fig, axes = plt.subplots(1, len(models_results), figsize=(20, 6))

    # Ensure axes is always an array, even for a single model
    if len(models_results) == 1:
        axes = [axes]

    plt.subplots_adjust(
        wspace=0.3, bottom=0.2
    )  # Adjust spacing between subplots and bottom margin

    # Create mask for upper triangle including diagonal
    mask = np.triu(np.ones((4, 4)), k=0).astype(bool)

    # Define proper case mapping
    label_map = {
        "disaster_types": "Disaster Types",
        "informative": "Informative",
        "humanitarian": "Humanitarian",
        "damage_severity": "Damage Severity",
    }

    # Plot correlation matrix for each model
    for idx, (model_name, true_labels, predictions) in enumerate(models_results):
        error_correlation, error_patterns = analyze_cross_task_confusion(
            true_labels, predictions
        )

        # Rename index and columns with proper case
        error_correlation.index = [label_map[idx] for idx in error_correlation.index]
        error_correlation.columns = [
            label_map[col] for col in error_correlation.columns
        ]

        # Plot correlation matrix
        sns.heatmap(
            error_correlation,
            annot=True,
            cmap=sns.diverging_palette(240, 10, s=100, l=40, n=9),
            vmin=-1,
            vmax=1,
            center=0.1,
            fmt=".2f",
            mask=mask,
            ax=axes[idx],
            cbar=False,
        )  # Remove color bar

        # Set background color for masked areas to white
        axes[idx].patch.set_facecolor("white")

        # Adjust labels
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=0, ha="center")
        axes[idx].set_yticklabels(axes[idx].get_yticklabels(), rotation=0)

        # Set title with model name in proper case
        model_title = model_name.replace("_", " ").title()
        axes[idx].set_title(f"{model_title}\nCross-Task Error Correlation")

    plt.tight_layout()

    if save_dir:
        plt.savefig(
            os.path.join(save_dir, "cross_task_error_matrices.png"),
            bbox_inches="tight",
            dpi=300,
        )

    plt.show()
    plt.close()


# =============================================================================
# analyze_class_errors
# =============================================================================


def analyze_class_errors(
    true_labels_dict: Dict[str, List], predictions_dict: Dict[str, List]
) -> pd.DataFrame:
    """
    Analyze error correlations between individual class labels across all tasks.

    Returns a DataFrame of error correlations between class-specific error masks.
    """
    class_names = {
        "disaster_types": {
            0: "Earthquake",
            1: "Fire",
            2: "Flood",
            3: "Hurricane",
            4: "Landslide",
            5: "None",
            6: "Other",
        },
        "informative": {0: "Not Informative", 1: "Informative"},
        "humanitarian": {
            0: "Injured",
            1: "Infrastructure",
            2: "Not Humanitarian",
            3: "Rescue",
        },
        "damage_severity": {0: "No Damage", 1: "Mild Damage", 2: "Severe Damage"},
    }
    error_masks = {}
    for task in true_labels_dict:
        true_array = np.array(true_labels_dict[task]).squeeze()
        pred_array = np.array(predictions_dict[task]).squeeze()
        for class_val in class_names[task]:
            key = f"{task}_{class_names[task][class_val]}"
            true_mask = (true_array == class_val).astype(int)
            pred_mask = (pred_array == class_val).astype(int)
            error_masks[key] = (true_mask != pred_mask).astype(int)
    error_df = pd.DataFrame.from_dict(error_masks)
    error_correlation = error_df.corr()
    return error_correlation


# =============================================================================
# analyze_cross_task_errors
# =============================================================================


def analyze_cross_task_errors(
    true_labels_dict: Dict[str, List], predictions_dict: Dict[str, List]
) -> pd.DataFrame:
    """
    Analyze error correlations between individual class labels across all tasks.

    Returns a DataFrame of error correlations between class-specific error masks.
    """
    class_names = {
        "disaster_types": {
            0: "Earthquake",
            1: "Fire",
            2: "Flood",
            3: "Hurricane",
            4: "Landslide",
            5: "None",
            6: "Other",
        },
        "informative": {0: "Not\nInformative", 1: "Informative"},
        "humanitarian": {
            0: "Injured",
            1: "Infrastructure",
            2: "Not\nHumanitarian",
            3: "Rescue",
        },
        "damage_severity": {0: "No\nDamage", 1: "Mild\nDamage", 2: "Severe\nDamage"},
    }
    error_masks = {}
    for task in true_labels_dict:
        true_array = np.array(true_labels_dict[task]).squeeze()
        pred_array = np.array(predictions_dict[task]).squeeze()
        for class_val in class_names[task]:
            key = f"{task}_{class_names[task][class_val]}"
            true_mask = (true_array == class_val).astype(int)
            pred_mask = (pred_array == class_val).astype(int)
            error_masks[key] = (true_mask != pred_mask).astype(int)
    error_df = pd.DataFrame.from_dict(error_masks)
    error_correlation = error_df.corr()
    return error_correlation


# =============================================================================
# plot_class_error_matrix
# =============================================================================


def plot_class_error_matrix(
    model_name: str,
    true_labels: Dict[str, List],
    predictions: Dict[str, List],
    save_dir: str = None,
):
    """
    Plot error correlation matrix showing relationships between classification errors.

    Parameters
    ----------
    model_name : str
        Name of the model.
    true_labels : dict
        True labels for each task.
    predictions : dict
        Predicted labels for each task.
    save_dir : str, optional
        Directory to save the plot.
    """
    error_correlation = analyze_cross_task_errors(true_labels, predictions)
    y_tasks = {
        "Informative": ["not inf", "inf"],
        "Humanitarian": ["injured", "infra", "not hum", "rescue"],
        "Damage\nSeverity": ["none", "mild", "severe"],
    }
    x_tasks = {
        "Disaster\nTypes": [
            "quake",
            "fire",
            "flood",
            "hurr.",
            "land.",
            "none",
            "other",
        ],
        "Informative": ["not inf", "inf"],
        "Humanitarian": ["injured", "infra", "not hum", "rescue"],
    }
    full_tasks = {**x_tasks, "Damage\nSeverity": y_tasks["Damage\nSeverity"]}
    indices = {}
    current_idx = 0
    for task, classes in full_tasks.items():
        indices[task] = (current_idx, current_idx + len(classes))
        current_idx += len(classes)
    y_indices = []
    for task in y_tasks:
        start, end = indices[task]
        y_indices.extend(range(start, end))
    x_indices = []
    for task in x_tasks:
        start, end = indices[task]
        x_indices.extend(range(start, end))
    error_correlation = error_correlation.iloc[y_indices, x_indices]
    mask = np.zeros_like(error_correlation, dtype=bool)
    y_start = 0
    for y_task in y_tasks:
        x_start = 0
        for x_task in x_tasks:
            if y_task == x_task:
                task_size_y = len(y_tasks[y_task])
                task_size_x = len(x_tasks[x_task])
                mask[
                    y_start : y_start + task_size_y, x_start : x_start + task_size_x
                ] = True
            x_start += len(x_tasks[x_task])
        y_start += len(y_tasks[y_task])
    if not plt.get_fignums():
        plt.figure(figsize=(24, 8))
        plt.subplots_adjust(wspace=0.0, bottom=0.3)
        plt.suptitle(
            "Class-Level Error Correlation Matrix",
            y=1.02,
            fontsize=15,
            fontweight="bold",
        )
    current_plot = len(plt.gcf().axes)
    ax = plt.subplot(1, 3, current_plot + 1)
    ax.patch.set_facecolor("white")
    formatted_annot = error_correlation.map(
        lambda x: f".{str(abs(x))[2:4]}" if abs(x) < 1 else f"{x:.2f}"
    )
    sns.heatmap(
        error_correlation,
        annot=formatted_annot,
        fmt="",
        cmap=sns.diverging_palette(240, 10, s=100, l=40, n=9),
        vmin=-1,
        vmax=1,
        center=0.07,
        mask=mask,
        cbar=False,
        square=False,
        linewidths=0,
        linecolor="white",
        annot_kws={"size": 12},
        ax=ax,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    x_labels = []
    for task in x_tasks:
        x_labels.extend(x_tasks[task])
    y_labels = []
    for task in y_tasks:
        y_labels.extend(y_tasks[task])
    current_pos = 0
    for task in x_tasks:
        current_pos += len(x_tasks[task])
        if current_pos < len(x_labels):
            ax.axvline(x=current_pos, color="black", linewidth=2)
    current_pos = 0
    for task in y_tasks:
        current_pos += len(y_tasks[task])
        if current_pos < len(y_labels):
            ax.axhline(y=current_pos, color="black", linewidth=2)
    plt.xticks(
        np.arange(len(x_labels)) + 0.5, x_labels, rotation=45, ha="right", fontsize=14
    )
    if current_plot == 0:
        plt.yticks(
            np.arange(len(y_labels)) + 0.5,
            y_labels,
            rotation=0,
            ha="right",
            fontsize=14,
        )
    else:
        plt.yticks(np.arange(len(y_labels)) + 0.5, [], rotation=0)
    task_start = 0
    for task in x_tasks:
        middle = task_start + len(x_tasks[task]) / 2
        plt.text(
            middle,
            len(y_labels) + 2,
            task,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        task_start += len(x_tasks[task])
    if current_plot == 0:
        task_start = 0
        for task in y_tasks:
            middle = task_start + len(y_tasks[task]) / 2
            plt.text(
                -2,
                middle,
                task,
                ha="right",
                va="center",
                fontsize=14,
                fontweight="bold",
            )
            task_start += len(y_tasks[task])
    plt.title(
        f'{model_name.replace("_", " ").title()}',
        pad=10,
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.2)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"{model_name}_class_errors.png"),
            bbox_inches="tight",
            dpi=300,
        )
    if len(plt.gcf().axes) == 3:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, "class_error_matrices.png"),
                bbox_inches="tight",
                dpi=300,
            )
        plt.show()
        plt.close()


# --------------------------------------------------------------------
# Global placeholders for the last full CM and labels
# (Populated by plot_full_confusion_matrices, read by analyze_full_confusion_matrix)
# --------------------------------------------------------------------
_LATEST_FULL_CM = None
_LATEST_BIG_LABEL_LIST = None


def plot_full_confusion_matrices(true_labels, predictions, save_dir=None):
    """
    Plot a 'full' confusion matrix for each model, combining all four tasks
    (disaster_types, informative, humanitarian, damage_severity) into a single
    label-space of size 7*2*4*3 = 168 possible label combinations.

    Each axis in the resulting matrix is labeled with a string such as:
        "quake|inf|injured|mild"
    indicating the combination of class labels for the four tasks:
        disaster_types|informative|humanitarian|damage_severity

    This version handles the possibility that some label arrays might be
    0-dimensional (single scalar) or empty by wrapping them in np.atleast_1d.

    Parameters
    ----------
    true_labels : dict
        Mapping task -> list/array of integers, e.g.:
            true_labels["disaster_types"] = [0,2,1,...]
            ...
    predictions : dict
        Same structure as true_labels but with predicted labels.
    save_dir : str, optional
        Directory in which to save the resulting PNGs. If None, only displays plots inline.
    """
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # Short labels consistent with plot_confusion_matrices
    task_names = {
        "disaster_types": ["quake", "fire", "flood", "hurr.", "land.", "none", "other"],
        "informative": ["not inf", "inf"],
        "humanitarian": ["injured", "infra", "not hum", "rescue"],
        "damage_severity": ["none", "mild", "severe"],
    }
    tasks_order = ["disaster_types", "informative", "humanitarian", "damage_severity"]
    task_class_sizes = [len(task_names[t]) for t in tasks_order]  # [7, 2, 4, 3]
    n_classes = np.prod(task_class_sizes)  # 168

    # Build a list of label strings like "quake|inf|injured|mild"
    big_label_list = []
    for dt_lbl in task_names["disaster_types"]:
        for inf_lbl in task_names["informative"]:
            for hum_lbl in task_names["humanitarian"]:
                for dmg_lbl in task_names["damage_severity"]:
                    combo_str = f"{dt_lbl}|{inf_lbl}|{hum_lbl}|{dmg_lbl}"
                    big_label_list.append(combo_str)

    def encode_label(dt_val, inf_val, hum_val, dmg_val):
        """
        Maps (disaster_type, informative, humanitarian, damage_severity)
        to a unique index among 168 possibilities.
        """
        return (
            dt_val * (task_class_sizes[1] * task_class_sizes[2] * task_class_sizes[3])
            + inf_val * (task_class_sizes[2] * task_class_sizes[3])
            + hum_val * task_class_sizes[3]
            + dmg_val
        )

    plt.style.use("default")
    sns.set_theme()

    # Ensure directory exists if saving
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Convert each task's label array (true/pred) into at-least-1D arrays.
    dt_true = np.atleast_1d(true_labels["disaster_types"])
    inf_true = np.atleast_1d(true_labels["informative"])
    hum_true = np.atleast_1d(true_labels["humanitarian"])
    dmg_true = np.atleast_1d(true_labels["damage_severity"])

    dt_pred = np.atleast_1d(predictions["disaster_types"])
    inf_pred = np.atleast_1d(predictions["informative"])
    hum_pred = np.atleast_1d(predictions["humanitarian"])
    dmg_pred = np.atleast_1d(predictions["damage_severity"])

    # Encode each sample's combined true/pred into [0..167]
    combined_true = []
    combined_pred = []

    # Loop over however many samples we have (could be zero!)
    num_samples = min(
        len(dt_true),
        len(inf_true),
        len(hum_true),
        len(dmg_true),
        len(dt_pred),
        len(inf_pred),
        len(hum_pred),
        len(dmg_pred),
    )
    for i in range(num_samples):
        ctrue = encode_label(dt_true[i], inf_true[i], hum_true[i], dmg_true[i])
        cpred = encode_label(dt_pred[i], inf_pred[i], hum_pred[i], dmg_pred[i])
        combined_true.append(ctrue)
        combined_pred.append(cpred)

    # Confusion matrix over 168 possible label combos
    cm = confusion_matrix(combined_true, combined_pred, labels=range(n_classes))

    # Normalize by row
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=(row_sums != 0),
    )

    # Diagonal > 0, off-diagonal < 0 for the diverging palette
    custom_norm = np.zeros_like(cm_normalized)
    for r in range(n_classes):
        for c in range(n_classes):
            if r == c:
                custom_norm[r, c] = cm_normalized[r, c]
            else:
                custom_norm[r, c] = -cm_normalized[r, c]

    # Prepare annotations
    annotations = []
    for row in cm_normalized:
        row_annot = []
        for val in row:
            if val == 1.0:
                row_annot.append("1.00")
            elif val == 0.0:
                row_annot.append("0")
            else:
                row_annot.append(f"{val:.2f}")
        annotations.append(row_annot)

    # Make a large figure so the matrix is more readable
    fig, ax = plt.subplots(figsize=(60, 60))

    sns.heatmap(
        custom_norm,
        annot=annotations,
        fmt="",
        cmap=sns.diverging_palette(10, 240, s=100, l=40, n=9),
        xticklabels=big_label_list,
        yticklabels=big_label_list,
        ax=ax,
        cbar=False,
        annot_kws={"fontsize": 8},
        vmin=-0.5,
        center=0,
        vmax=1.0,
        linewidths=0.1,
        linecolor="gray",
    )

    ax.set_title("Full Confusion Matrix (All 4 Tasks Combined)", fontsize=30, pad=30)
    ax.set_xlabel("Predicted", fontsize=20, labelpad=20)
    ax.set_ylabel("True", fontsize=20, labelpad=20)

    # Rotate the x-tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)

    if save_dir:
        outfile = os.path.join(save_dir, "full_confusion_matrix.png")
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close(fig)

    global _LATEST_FULL_CM, _LATEST_BIG_LABEL_LIST
    _LATEST_FULL_CM = cm
    _LATEST_BIG_LABEL_LIST = big_label_list


def analyze_full_confusion_matrix(top_n=10):
    """
    Perform multi-faceted analysis on the 168x168 confusion matrix computed
    by plot_full_confusion_matrices, without needing to pass cm, labels, etc.

    Parameters
    ----------
    top_n : int
        How many “top confusions” to display in certain summaries.

    Returns
    -------
    results : dict
        A dictionary containing DataFrames with various insights:
         {
           "row_summaries": ...,
           "column_summaries": ...,
           "top_off_diagonal": ...,
           "taskwise_summary": ...,
           "pairwise_summary": ...
         }
    """
    import numpy as np
    import pandas as pd

    pd.set_option("display.max_colwidth", None)

    # ------------------ Access the global matrix & labels ------------------
    global _LATEST_FULL_CM, _LATEST_BIG_LABEL_LIST
    cm = _LATEST_FULL_CM
    big_label_list = _LATEST_BIG_LABEL_LIST

    if cm is None or big_label_list is None:
        raise ValueError(
            "No full confusion matrix is available. "
            "Make sure you call plot_full_confusion_matrices first."
        )

    # -----------------------------------------------------------------------
    # Bake in your known task info. (No need to pass them as arguments.)
    # -----------------------------------------------------------------------
    task_names = {
        "disaster_types": ["quake", "fire", "flood", "hurr.", "land.", "none", "other"],
        "informative": ["not inf", "inf"],
        "humanitarian": ["injured", "infra", "not hum", "rescue"],
        "damage_severity": ["none", "mild", "severe"],
    }
    tasks_order = ["disaster_types", "informative", "humanitarian", "damage_severity"]

    # Optionally, if you have a real-world frequency or importance weighting for each row,
    # you could hardcode that here or load from a file. We'll default to None for simplicity:
    row_frequencies = None

    # Now we can reuse the analysis logic from the previous snippet
    # (shown here in condensed form).
    n = cm.shape[0]
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_norm = np.divide(
            cm, row_sums, out=np.zeros_like(cm, dtype=float), where=(row_sums != 0)
        )

    # 1) Row-wise summary: accuracy + top confusions
    row_info = []
    for i in range(n):
        diag_acc = row_norm[i, i]
        support = cm[i, :].sum()
        row_copy = row_norm[i, :].copy()
        row_copy[i] = -1  # ignore diagonal
        top_conf_inds = row_copy.argsort()[::-1][:top_n]
        top_conf_list = []
        for idx in top_conf_inds:
            if row_copy[idx] <= 0:
                break
            top_conf_list.append(f"{big_label_list[idx]}({row_copy[idx]:.2f})")
        row_info.append(
            {
                "Label": big_label_list[i],
                "Support": int(support),
                "Accuracy": float(diag_acc),
                "TopConfusions": "; ".join(top_conf_list),
            }
        )
    df_row_summaries = pd.DataFrame(row_info)
    df_row_summaries.sort_values(
        by=["Accuracy", "Support"], ascending=[True, False], inplace=True
    )
    df_row_summaries.reset_index(drop=True, inplace=True)

    # 2) Column-wise summary: top true sources for each predicted label
    col_sums = cm.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        col_norm = np.divide(
            cm, col_sums, out=np.zeros_like(cm, dtype=float), where=(col_sums != 0)
        )
    col_info = []
    for j in range(n):
        total_pred = cm[:, j].sum()
        col_copy = col_norm[:, j].copy()
        col_copy[j] = -1
        top_src_inds = col_copy.argsort()[::-1][:top_n]
        top_src_list = []
        for idx in top_src_inds:
            if col_copy[idx] <= 0:
                break
            top_src_list.append(f"{big_label_list[idx]}({col_copy[idx]:.2f})")
        col_info.append(
            {
                "Label": big_label_list[j],
                "TotalPred": int(total_pred),
                "TopSources": "; ".join(top_src_list),
            }
        )
    df_col_summaries = pd.DataFrame(col_info)
    df_col_summaries.sort_values("TotalPred", ascending=False, inplace=True)
    df_col_summaries.reset_index(drop=True, inplace=True)

    # 3) Global top-N off-diagonal confusions
    off_list = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = row_norm[i, j]
            if val > 0:
                off_list.append((i, j, val, cm[i, j]))
    off_list.sort(key=lambda x: x[2], reverse=True)
    off_data = []
    for i, j, frac, abs_count in off_list[:top_n]:
        off_data.append(
            {
                "TrueLabel": big_label_list[i],
                "PredLabel": big_label_list[j],
                "RowFrac": frac,
                "AbsCount": abs_count,
            }
        )
    df_top_off_diag = pd.DataFrame(off_data)

    # 4) Summaries at the single-task level (aggregating the other 3 tasks)
    decoded_labels = [lab.split("|") for lab in big_label_list]
    results_taskwise = []
    for t in tasks_order:
        possible_class_names = task_names[t]
        t_idx = tasks_order.index(t)
        for c_name in possible_class_names:
            row_inds = [r for r in range(n) if decoded_labels[r][t_idx] == c_name]
            total_count = 0
            correct_count = 0
            for r_i in row_inds:
                row_sum_i = cm[r_i, :].sum()
                total_count += row_sum_i
                # columns j that also have c_name in dimension t_idx
                col_matches = [
                    c for c in range(n) if decoded_labels[c][t_idx] == c_name
                ]
                correct_count += cm[r_i, col_matches].sum()
            acc = (correct_count / total_count) if total_count > 0 else 0.0
            results_taskwise.append(
                {
                    "Task": t,
                    "Label": c_name,
                    "Support": int(total_count),
                    "Accuracy": acc,
                }
            )
    df_taskwise = pd.DataFrame(results_taskwise)

    # 5) Summaries at the pair-of-tasks level
    from itertools import combinations

    pairwise_data = []
    for tA, tB in combinations(tasks_order, 2):
        idxA = tasks_order.index(tA)
        idxB = tasks_order.index(tB)
        for lblA in task_names[tA]:
            for lblB in task_names[tB]:
                row_inds = [
                    r
                    for r in range(n)
                    if decoded_labels[r][idxA] == lblA
                    and decoded_labels[r][idxB] == lblB
                ]
                total_count = 0
                correct_count = 0
                for r_i in row_inds:
                    row_sum_i = cm[r_i, :].sum()
                    total_count += row_sum_i
                    col_matches = [
                        c
                        for c in range(n)
                        if decoded_labels[c][idxA] == lblA
                        and decoded_labels[c][idxB] == lblB
                    ]
                    correct_count += cm[r_i, col_matches].sum()
                acc = (correct_count / total_count) if total_count > 0 else 0.0
                pairwise_data.append(
                    {
                        "Task Pair": f"{tA}+{tB}",
                        "Label Pair": f"{lblA}|{lblB}",
                        "Support": int(total_count),
                        "Accuracy": acc,
                    }
                )
    df_pairwise = pd.DataFrame(pairwise_data)

    # Return a dictionary of dataframes
    return {
        "row_summaries": df_row_summaries,
        "column_summaries": df_col_summaries,
        "top_off_diagonal": df_top_off_diag,
        "taskwise_summary": df_taskwise,
        "pairwise_summary": df_pairwise,
    }


def top_label_combos_by_error_contribution(top_n=20, min_support=1, display_html=True):
    """
    Identify the label-combination rows (among the 168 possible) that contribute
    the most total misclassifications, and return a sorted table.

    The function uses the global _LATEST_FULL_CM and _LATEST_BIG_LABEL_LIST
    generated by plot_full_confusion_matrices(...). It computes, for each row i:

        support_i = sum of row i in the CM
        correct_i = cm[i, i]
        errors_i  = support_i - correct_i
        error_fraction = errors_i / sum_of_all_errors

    Then sorts rows by errors_i descending. You can also specify a minimum
    support threshold to ignore extremely tiny subsets.

    Parameters
    ----------
    top_n : int
        How many of the top error-contributing rows to display.
    min_support : int
        Minimum row support to consider. Rows with fewer than this many samples
        are omitted from the final table.
    display_html : bool
        If True, returns a Pandas Styler for pretty HTML display in Jupyter.
        Otherwise, returns the raw DataFrame.

    Returns
    -------
    table : pd.DataFrame or pd.io.formats.style.Styler
        A DataFrame (or Styler) sorted by descending error contribution.
        Columns include "ComboLabel", "Support", "Correct", "Errors",
        "ErrorFraction", "Accuracy", plus a breakdown of each sub-task label
        (disaster_types, informative, humanitarian, damage_severity).
    """
    import numpy as np
    import pandas as pd
    from IPython.display import display, HTML

    global _LATEST_FULL_CM, _LATEST_BIG_LABEL_LIST
    cm = _LATEST_FULL_CM
    big_label_list = _LATEST_BIG_LABEL_LIST

    if cm is None or big_label_list is None:
        raise ValueError(
            "No full confusion matrix is available. Call plot_full_confusion_matrices first."
        )

    # Basic info
    n = cm.shape[0]  # should be 168
    row_sums = cm.sum(axis=1)
    correct_diagonal = np.diag(cm)
    errors_all = row_sums - correct_diagonal
    total_errors = errors_all.sum()

    # We can parse each combined label into (disaster_types, informative, humanitarian, damage_severity)
    # to show them in separate columns. The order is consistent with tasks_order in your code.
    tasks_order = ["disaster_types", "informative", "humanitarian", "damage_severity"]

    # Helper to decode a label like "quake|inf|rescue|mild" => (quake, inf, rescue, mild)
    def decode_label_parts(full_str):
        return full_str.split("|")  # e.g. ["quake", "inf", "rescue", "mild"]

    rows_data = []
    for i in range(n):
        support_i = row_sums[i]
        if support_i < min_support:
            continue
        correct_i = correct_diagonal[i]
        errors_i = errors_all[i]
        acc_i = correct_i / support_i if support_i > 0 else 0.0
        err_frac = errors_i / total_errors if total_errors > 0 else 0.0

        # parse the label combo string
        parts = decode_label_parts(
            big_label_list[i]
        )  # e.g. ["fire", "not inf", "rescue", "none"]
        row_dict = {
            "ComboLabel": big_label_list[i],
            "Support": int(support_i),
            "Correct": int(correct_i),
            "Errors": int(errors_i),
            "ErrorFraction": err_frac,
            "Accuracy": acc_i,
        }
        # attach each sub-task label in separate columns
        row_dict["DisasterType"] = parts[0]
        row_dict["Informativeness"] = parts[1]
        row_dict["Humanitarian"] = parts[2]
        row_dict["DamageSeverity"] = parts[3]

        rows_data.append(row_dict)

    df = pd.DataFrame(rows_data)
    # Sort by the largest absolute Errors first
    df.sort_values("Errors", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # If desired, only keep top_n rows
    df = df.head(top_n)

    # Format numeric columns
    df["Accuracy"] = df["Accuracy"].apply(lambda x: f"{100*x:.1f}%")
    df["ErrorFraction"] = df["ErrorFraction"].apply(lambda x: f"{100*x:.2f}%")

    # Optionally return a styled HTML table
    if display_html:
        styled_df = df.style.set_properties(**{"text-align": "center"})
        styled_df = styled_df.set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("text-align", "center"), ("font-weight", "bold")],
                },
                {"selector": "td", "props": [("text-align", "center")]},
                {"selector": "td:nth-child(1)", "props": [("text-align", "left")]},
                {"selector": ".index_name", "props": "display: none;"},
                {"selector": ".index_col", "props": "display: none;"},
            ]
        )

        # Make sure subset columns that represent numbers are right-aligned or highlighted, etc.
        # E.g., highlight the largest "Errors".
        def highlight_max(s):
            if s.name != "Errors":
                return ["" for _ in s]
            max_val = s.max()
            return ["color: red; font-weight: bold;" if v == max_val else "" for v in s]

        styled_df = styled_df.apply(highlight_max, subset=["Errors"], axis=0)

        display(styled_df)
        return styled_df
    else:
        return df


def pair_label_combos_by_error_contribution(
    top_n=20, min_support=10, display_html=True
):
    """
    Compute and return a table of pairwise label combos (across all pairs of tasks),
    ranked by how many partial misclassifications they contribute.

    'Partial correctness' means that for a pair of tasks (TaskA, TaskB), we only
    care whether the model got TaskA and TaskB right, ignoring Tasks C, D. For each
    possible label pair (e.g., 'quake|inf'), we determine:

        - Support: number of images in which GT(TaskA) = 'quake' and GT(TaskB) = 'inf'
        - Correct: among those images, how many times does predicted(TaskA) = 'quake'
          AND predicted(TaskB) = 'inf' (irrespective of the other tasks).
        - Errors = Support - Correct
        - ErrorFraction = fraction of total errors among *all* pair combos
        - Accuracy = Correct / Support

    We combine the results from all 6 possible pairs of tasks into one DataFrame,
    sort by Errors descending, and display the top_n. Also skip any combos with
    < min_support in the ground truth.

    Parameters
    ----------
    top_n : int
        Number of top combos to display.
    min_support : int
        Minimum number of samples required for a pairwise label combo to appear.
    display_html : bool
        If True, return a styled DataFrame for IPython display; else return the raw DataFrame.

    Returns
    -------
    results : pd.DataFrame or pd.io.formats.style.Styler
        A DataFrame (or Styler if display_html=True) with columns:
          ['PairTask', 'PairLabel', 'Support', 'Correct', 'Errors',
           'ErrorFraction', 'Accuracy']
        sorted by Errors desc. Only the top_n rows are shown.
    """
    import numpy as np
    import pandas as pd
    from itertools import combinations
    from IPython.display import display

    # We rely on the globally stored 168x168 confusion matrix + label list
    # from plot_full_confusion_matrices.
    global _LATEST_FULL_CM, _LATEST_BIG_LABEL_LIST
    cm = _LATEST_FULL_CM
    big_label_list = _LATEST_BIG_LABEL_LIST

    if cm is None or big_label_list is None:
        raise ValueError(
            "No full confusion matrix found. Please call plot_full_confusion_matrices(...) first."
        )

    # The 4 tasks in the known order: (disaster_types, informative, humanitarian, damage_severity)
    tasks_order = ["disaster_types", "informative", "humanitarian", "damage_severity"]
    # For convenience, define how many classes each task has:
    task_class_sizes = [7, 2, 4, 3]  # consistent with your code

    # 1) We need a quick decode function: given an index 0..167, which
    #    labels does it represent for each of the 4 tasks?
    def decode_label_index(combo_idx):
        # e.g. dt_val = combo_idx // (2*4*3)
        rema = combo_idx
        out = []
        for i, size in enumerate(task_class_sizes[:-1]):
            block_size = np.prod(task_class_sizes[i + 1 :])
            val = rema // block_size
            out.append(int(val))
            rema = rema % block_size
        out.append(int(rema))
        return tuple(out)  # (dt_idx, inf_idx, hum_idx, dmg_idx)

    # Also build a dictionary of label strings (like "quake", "inf", etc.) for each task index:
    task_label_strings = {
        "disaster_types": ["quake", "fire", "flood", "hurr.", "land.", "none", "other"],
        "informative": ["not inf", "inf"],
        "humanitarian": ["injured", "infra", "not hum", "rescue"],
        "damage_severity": ["none", "mild", "severe"],
    }

    # Decode the big_label_list into a list of (dt_idx, inf_idx, hum_idx, dmg_idx)
    # or we can do it on the fly. But let's store it for speed:
    n = cm.shape[0]
    decoded = []
    for i in range(n):
        # e.g. big_label_list[i] = 'quake|inf|infra|none'
        # or we can decode by the numeric approach above.
        # We'll do numeric approach to be safe:
        tuple_indices = decode_label_index(i)
        decoded.append(tuple_indices)

    # 2) We'll accumulate partial correctness data for *all* pairs of tasks:
    #    For each pair (taskA, taskB), for each label combo in that pair,
    #    compute the total Support, Correct, and so on.
    all_rows = []
    # We'll define total_errors as the sum of row_sums - diag for the entire 168x168 CM,
    # so we can define an "ErrorFraction" that is consistent with your prior approach:
    row_sums = cm.sum(axis=1)
    diag_vals = np.diag(cm)
    total_errors = (row_sums - diag_vals).sum()

    # A small helper to find all row indices that match (taskA=valA, taskB=valB)
    # ignoring tasks C, D.
    def row_indices_for_pair_value(taskA_idx, valA, taskB_idx, valB):
        # We want all i in [0..167] for which decoded[i][taskA_idx] == valA
        # and decoded[i][taskB_idx] == valB
        # We'll just do a list comprehension:
        return [
            i
            for i in range(n)
            if (decoded[i][taskA_idx] == valA and decoded[i][taskB_idx] == valB)
        ]

    # Similarly, to find column indices that match predicted (taskA=valA, taskB=valB).
    # The predicted combos are the same dimension, so it's the same decode logic:
    def col_indices_for_pair_value(taskA_idx, valA, taskB_idx, valB):
        return row_indices_for_pair_value(taskA_idx, valA, taskB_idx, valB)

    # We'll loop over all pairs of tasks:
    for tA, tB in combinations(range(4), 2):
        # e.g. tA=0, tB=1 => (disaster_types, informative)
        # gather the label strings
        sizeA = task_class_sizes[tA]
        sizeB = task_class_sizes[tB]
        taskA_name = tasks_order[tA]
        taskB_name = tasks_order[tB]

        for valA in range(sizeA):
            for valB in range(sizeB):
                # 2.1) Determine the row subset
                row_inds = row_indices_for_pair_value(tA, valA, tB, valB)
                if not row_inds:
                    continue
                # The 'support' is the sum of row sums across those rows
                support = row_sums[row_inds].sum()
                if support < min_support:
                    continue

                # 2.2) Determine how many were correct for (taskA=valA, taskB=valB)
                # This is the sum of cm[row_inds, col_inds] for col_inds that also have
                # predicted (tA=valA, tB=valB).
                col_inds = col_indices_for_pair_value(tA, valA, tB, valB)
                correct = 0
                for r in row_inds:
                    correct += cm[r, col_inds].sum()

                errors = support - correct
                if support > 0:
                    acc = correct / support
                else:
                    acc = 0.0
                if total_errors > 0:
                    err_frac = errors / total_errors
                else:
                    err_frac = 0.0

                # 2.3) Build row dict
                # We'll produce a label like "quake|inf" (depending on the tasks).
                # We'll get the actual strings from task_label_strings[taskA_name][valA], etc.
                labelA_str = task_label_strings[taskA_name][valA]
                labelB_str = task_label_strings[taskB_name][valB]
                pair_label = f"{labelA_str}|{labelB_str}"
                pair_task_str = f"{taskA_name}|{taskB_name}"

                row_dict = {
                    "PairTask": pair_task_str,
                    "PairLabel": pair_label,
                    "Support": int(support),
                    "Correct": int(correct),
                    "Errors": int(errors),
                    "ErrorFraction": err_frac,
                    "Accuracy": acc,
                }
                all_rows.append(row_dict)

    # 3) Convert to DataFrame and sort by errors descending
    df_all = pd.DataFrame(all_rows)
    if df_all.empty:
        # If we skip everything due to min_support, may be empty
        print("No pairwise combos found above the min_support threshold.")
        return df_all

    df_all.sort_values("Errors", ascending=False, inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    # 4) Keep top_n
    df_top = df_all.head(top_n).copy()

    # Format numeric columns nicely
    df_top["Accuracy"] = df_top["Accuracy"].apply(lambda x: f"{100*x:.1f}%")
    df_top["ErrorFraction"] = df_top["ErrorFraction"].apply(lambda x: f"{100*x:.2f}%")

    # 5) Optionally return a styled or raw DataFrame
    if display_html:
        styled = df_top.style.set_properties(**{"text-align": "center"})
        styled = styled.set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("text-align", "center"), ("font-weight", "bold")],
                },
                {"selector": "td", "props": [("text-align", "center")]},
                {"selector": "td:nth-child(1)", "props": [("text-align", "left")]},
                {"selector": ".index_name", "props": "display: none;"},
                {"selector": ".index_col", "props": "display: none;"},
            ]
        )

        # Example highlight for max errors in the top_n:
        def highlight_max_errors(s):
            if s.name != "Errors":
                return ["" for _ in s]
            max_val = s.max()
            return ["color: red; font-weight: bold;" if v == max_val else "" for v in s]

        styled = styled.apply(highlight_max_errors, subset=["Errors"], axis=0)

        display(styled)
        return styled
    else:
        return df_top


def plot_confusion_subsets(
    true_labels,
    predictions,
    subsets_of_interest,
    save_dir=None,
    min_support=10,
    min_threshold=50,
    top_x=15,
):
    """
    Plot confusion matrices for the given subsets, but only display
    up to top_x row/column combos that have either:
      - row_support >= min_threshold OR row_errors >= min_threshold

    Also ensures the final matrix is square by intersecting
    the kept row labels and col labels. Highlights diagonal in bold.

    Parameters
    ----------
    true_labels, predictions : dict
        As before.
    subsets_of_interest : list of dict
        As before.
    save_dir : str or None
        If provided, store each figure. Otherwise just show inline.
    min_support : int
        If the entire subset has fewer than this many examples, skip it altogether.
    min_threshold : int
        The minimum row support or error count needed to keep a row.
    top_x : int
        Maximum number of row/col labels to keep after filtering by threshold.
    """

    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ----------------------------------------------------------------------
    # Same preliminary steps: define tasks, encode the 168 combos, etc.
    # ----------------------------------------------------------------------
    task_names = {
        "disaster_types": ["quake", "fire", "flood", "hurr.", "land.", "none", "other"],
        "informative": ["not inf", "inf"],
        "humanitarian": ["injured", "infra", "not hum", "rescue"],
        "damage_severity": ["none", "mild", "severe"],
    }
    tasks_order = ["disaster_types", "informative", "humanitarian", "damage_severity"]
    label2idx = {}
    for t in tasks_order:
        label2idx[t] = {}
        for i, lab in enumerate(task_names[t]):
            label2idx[t][lab] = i

    class_sizes = [len(task_names[t]) for t in tasks_order]  # e.g. [7,2,4,3]

    def encode_label(dt_val, inf_val, hum_val, dmg_val):
        return (
            dt_val * (class_sizes[1] * class_sizes[2] * class_sizes[3])
            + inf_val * (class_sizes[2] * class_sizes[3])
            + hum_val * class_sizes[3]
            + dmg_val
        )

    # Build the big label list
    big_label_list = []
    for dt_lbl in task_names["disaster_types"]:
        for inf_lbl in task_names["informative"]:
            for hum_lbl in task_names["humanitarian"]:
                for dmg_lbl in task_names["damage_severity"]:
                    big_label_list.append(f"{dt_lbl}|{inf_lbl}|{hum_lbl}|{dmg_lbl}")

    # Flatten arrays
    dt_true = np.ravel(true_labels["disaster_types"])
    inf_true = np.ravel(true_labels["informative"])
    hum_true = np.ravel(true_labels["humanitarian"])
    dmg_true = np.ravel(true_labels["damage_severity"])
    dt_pred = np.ravel(predictions["disaster_types"])
    inf_pred = np.ravel(predictions["informative"])
    hum_pred = np.ravel(predictions["humanitarian"])
    dmg_pred = np.ravel(predictions["damage_severity"])

    num_samples = len(dt_true)
    combined_true_all = np.empty(num_samples, dtype=int)
    combined_pred_all = np.empty(num_samples, dtype=int)
    for i in range(num_samples):
        combined_true_all[i] = encode_label(
            dt_true[i], inf_true[i], hum_true[i], dmg_true[i]
        )
        combined_pred_all[i] = encode_label(
            dt_pred[i], inf_pred[i], hum_pred[i], dmg_pred[i]
        )

    def wrap_label(label_str):
        parts = label_str.split("|")
        return f"{parts[0]}|{parts[1]}\n{parts[2]}|{parts[3]}"

    # ----------------------------------------------------------------------
    # Now the main logic for each subset
    # ----------------------------------------------------------------------
    for subset_info in subsets_of_interest:
        subset_name = subset_info["name"]
        conditions = subset_info["conditions"]

        # 1) Filter samples that match the ground-truth conditions
        mask = np.ones(num_samples, dtype=bool)
        for tsk, allowed_labels in conditions.items():
            allowed_idx = [label2idx[tsk][lbl] for lbl in allowed_labels]
            if tsk == "disaster_types":
                mask &= np.isin(dt_true, allowed_idx)
            elif tsk == "informative":
                mask &= np.isin(inf_true, allowed_idx)
            elif tsk == "humanitarian":
                mask &= np.isin(hum_true, allowed_idx)
            elif tsk == "damage_severity":
                mask &= np.isin(dmg_true, allowed_idx)
        subset_inds = np.where(mask)[0]
        if len(subset_inds) < min_support:
            continue  # skip entirely

        # 2) Build confusion matrix over the selected subset
        subset_true = combined_true_all[subset_inds]
        subset_pred = combined_pred_all[subset_inds]

        # Collect unique row labels & col labels
        row_labels = np.unique(subset_true)
        col_labels = np.unique(subset_pred)
        # If we want a strictly "square" matrix aligned on the diagonal, let's
        # intersect row_labels & col_labels so the same set is used for both.
        # That way, label i for row is the same index in col => diagonal aligns.
        common_labels = np.intersect1d(row_labels, col_labels)
        # If you prefer to keep the union, you'd do np.union1d, but then
        # the diagonal won't necessarily line up. We'll assume intersect.
        row_labels = list(common_labels)
        col_labels = list(common_labels)

        if not row_labels or not col_labels:
            continue

        row_indexer = {val: i for i, val in enumerate(row_labels)}
        col_indexer = {val: j for j, val in enumerate(col_labels)}

        # Fill matrix
        small_cm = np.zeros((len(row_labels), len(col_labels)), dtype=int)
        for i in range(len(subset_true)):
            if subset_true[i] in row_indexer and subset_pred[i] in col_indexer:
                r = row_indexer[subset_true[i]]
                c = col_indexer[subset_pred[i]]
                small_cm[r, c] += 1

        # 3) Filter rows & cols by threshold
        row_sums = small_cm.sum(axis=1)
        diag_vals = np.diag(small_cm)
        row_errors = row_sums - diag_vals
        # We'll keep rows if row_sums[i]>=min_threshold or row_errors[i]>=min_threshold
        row_keep_indices = [
            i
            for i in range(len(row_labels))
            if (row_sums[i] >= min_threshold or row_errors[i] >= min_threshold)
        ]
        # Then sort them by row_errors descending (or row_sums, your call)
        row_keep_indices.sort(key=lambda i: row_errors[i], reverse=True)
        row_keep_indices = row_keep_indices[:top_x]  # keep top X

        # We'll do the same for columns
        col_sums = small_cm.sum(axis=0)
        col_diag = np.diag(small_cm)
        col_errors = col_sums - col_diag
        col_keep_indices = [
            j
            for j in range(len(col_labels))
            if (col_sums[j] >= min_threshold or col_errors[j] >= min_threshold)
        ]
        col_keep_indices.sort(key=lambda j: col_errors[j], reverse=True)
        col_keep_indices = col_keep_indices[:top_x]

        # Finally intersect the row_keep_indices with col_keep_indices
        # so we get a square matrix with the same combos in rows & cols:
        final_keep = np.intersect1d(row_keep_indices, col_keep_indices)
        if len(final_keep) == 0:
            continue

        # Sort final_keep by row_errors desc again or by label index
        final_keep = sorted(final_keep, key=lambda i: row_errors[i], reverse=True)

        # Build the trimmed matrix
        small_cm = small_cm[final_keep, :][:, final_keep]
        new_row_labels = [row_labels[i] for i in final_keep]
        new_col_labels = [col_labels[i] for i in final_keep]

        # Normalise row-wise
        row_sums_trimmed = small_cm.sum(axis=1, keepdims=True)
        small_cm_norm = np.divide(
            small_cm.astype(float),
            row_sums_trimmed,
            out=np.zeros_like(small_cm, dtype=float),
            where=(row_sums_trimmed != 0),
        )

        # Create "custom_norm" for red/blue palette (diagonal positive, off-diag negative)
        custom_norm = np.zeros_like(small_cm_norm)
        for rr in range(len(final_keep)):
            for cc in range(len(final_keep)):
                val = small_cm_norm[rr, cc]
                if rr == cc:
                    # diagonal => positive
                    custom_norm[rr, cc] = val
                else:
                    # off diag => negative
                    custom_norm[rr, cc] = -val

        # Format annotations, with diagonal in bold
        annotations = []
        for rr in range(len(final_keep)):
            row_annot = []
            for cc in range(len(final_keep)):
                frac_val = small_cm_norm[rr, cc]
                abs_count = small_cm[rr, cc]
                frac_str = (
                    f"{frac_val:.2f}" if (0 < frac_val < 1) else f"{int(frac_val)}"
                )
                # bold on diagonal
                if rr == cc:
                    row_annot.append(f"**{frac_str}\n({abs_count})**")
                else:
                    row_annot.append(f"{frac_str}\n({abs_count})")
            annotations.append(row_annot)

        # Wrap label
        row_label_texts = [wrap_label(big_label_list[idx]) for idx in new_row_labels]
        col_label_texts = [wrap_label(big_label_list[idx]) for idx in new_col_labels]

        total_subset = small_cm.sum()

        # Plot
        fig_width = max(6, 0.8 * len(final_keep) + 1)
        fig_height = max(4, 0.6 * len(final_keep) + 1)
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(
            custom_norm,
            annot=annotations,
            fmt="",
            cmap=sns.diverging_palette(10, 240, s=100, l=40, n=9),
            xticklabels=col_label_texts,
            yticklabels=row_label_texts,
            cbar=False,
            annot_kws={"fontsize": 9},
            vmin=-1,
            center=0,
            vmax=1,
            linewidths=0.1,
            linecolor="gray",
        )
        plt.title(
            f"{subset_name} - Filtered to top combos (N={total_subset})", fontsize=12
        )
        plt.xlabel("Predicted Label", fontsize=10)
        plt.ylabel("True Label", fontsize=10)
        plt.xticks(rotation=0, fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            outpath = os.path.join(save_dir, f"subset_{subset_name}_filtered.png")
            plt.savefig(outpath, dpi=150, bbox_inches="tight")

        plt.show()
        plt.close()
