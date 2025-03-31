"""
class_imbalance.py

This module contains functions for analysing class imbalance effects and generating
aggregated performance tables across multiple models. Configuration parameters such as
label mappings, tasks, batch size, and data directory are now imported from config.py.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple
from collections import defaultdict
from IPython.display import display, HTML

from ..core.model_tester import load_model, test_model
from .model_analyser import (
    calculate_metrics,
)  # assumes calculate_metrics still works with provided mapping

# Import configuration variables from config.py
from config import DATA_DIR, BATCH_SIZE, NUM_THREADS, LABEL_MAPPINGS, TASKS, MODEL_PATHS

# Remove the local definition of label_mappings (now in config.py)
# Also, tasks will be imported as TASKS from config.py


def format_label(label: str) -> str:
    """
    Format task/class labels for display.

    Parameters
    ----------
    label : str
        The snake_case label to format.

    Returns
    -------
    str
        The label in Title Case.
    """
    return " ".join(word.capitalize() for word in label.replace("_", " ").split())


def analyze_class_imbalance_effects(
    data_dir: str = None, model_paths: Dict[str, str] = None
) -> None:
    """
    Analyze class imbalance effects using scatter plots to show clusters.

    Parameters
    ----------
    data_dir : str
        Path to the MEDIC dataset.
    model_paths : Dict[str, str]
        Dictionary mapping model names to checkpoint paths.

    Returns
    -------
    None
        Displays scatter plots for performance metrics vs. class frequency.
    """
    # --- Seaborn style ---
    sns.set_theme(style="whitegrid")

    # Use TASKS imported from config.py instead of hard-coded list.
    tasks = TASKS
    data_dir = DATA_DIR
    model_paths = MODEL_PATHS
    metrics = ["f1_score", "precision", "recall"]

    # Color scheme for tasks (still hard-coded here; you may later choose to centralize these too)
    task_colors = {
        "damage_severity": "royalblue",
        "informative": "forestgreen",
        "humanitarian": "darkorange",
        "disaster_types": "purple",
    }

    # Markers for models
    model_markers = {"resnet50": "o", "efficientnet_b1": "s", "mobilenet_v2": "^"}

    # Store data for plotting
    plot_data = []

    # Process each model
    for model_name, model_path in model_paths.items():
        # Load the model using the tester function.
        model = load_model(model_name, model_path)
        # Use defaults from config for batch_size and num_threads.
        true_labels, predictions, files = test_model(
            model, data_dir, batch_size=BATCH_SIZE, num_threads=NUM_THREADS
        )
        # Pass the global LABEL_MAPPINGS (from config) to calculate_metrics.
        metrics_dict = calculate_metrics(true_labels, predictions, LABEL_MAPPINGS)

        # Calculate frequencies and metrics for each task.
        test_tsv = os.path.join(data_dir, "MEDIC_test.tsv")
        data_df = pd.read_csv(test_tsv, sep="\t")
        total_samples = len(data_df)

        for task in tasks:
            class_counts = data_df[task].value_counts()
            class_frequencies = class_counts / total_samples

            # Iterate through each class for this task using LABEL_MAPPINGS.
            for class_id, class_name in LABEL_MAPPINGS[task].items():
                if class_name in metrics_dict[task]["classes"]:
                    perf_metrics = metrics_dict[task]["classes"][class_name]
                    plot_data.append(
                        {
                            "task": task,
                            "model": model_name,
                            "class": class_name,
                            "frequency": class_frequencies.get(class_name, 0),
                            "f1_score": perf_metrics["f1_score"],
                            "precision": perf_metrics["precision"],
                            "recall": perf_metrics["recall"],
                            "support": class_counts.get(class_name, 0),
                        }
                    )

    # Convert to DataFrame for easier plotting.
    plot_df = pd.DataFrame(plot_data)

    # Create figure with 3 subplots (one per metric)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(right=0.85)  # Make room for legend

    # Plot each metric.
    for idx, metric in enumerate(metrics):
        sns.scatterplot(
            data=plot_df,
            x="frequency",
            y=metric,
            hue="task",
            style="model",
            s=100,
            alpha=0.7,
            ax=axes[idx],
        )
        axes[idx].set_title(f"{metric.replace('_', ' ').title()}")
        axes[idx].set_xlabel("Class Frequency")
        axes[idx].set_ylabel(f"{metric.replace('_', ' ').title()} Score (%)")
        axes[idx].set_ylim(0, 100)
        if idx < 2:
            axes[idx].get_legend().remove()

    # Customize the legend on the last plot.
    axes[2].legend(
        bbox_to_anchor=(1.05, 0.5), loc="center left", title="Tasks & Models"
    )

    plt.tight_layout()
    plt.show()


def analyze_class_imbalance_table(
    data_dir: str = None, model_paths: Dict[str, str] = None
) -> None:
    """
    Generates and displays an aggregated table (one row per Task and Class) that aggregates results
    from multiple models. Performance metrics (F1, Precision, Recall) are shown as min–max ranges,
    and F1 rank is presented as a list of ranks from each model.

    Parameters
    ----------
    data_dir : str, optional
        Directory with 'MEDIC_test.tsv' for the test set. Defaults to DATA_DIR from config.
    model_paths : Dict[str, str], optional
        Mapping from model names to checkpoint paths. Defaults to MODEL_PATHS from config.

    Returns
    -------
    None
        Displays an aggregated table (in IPython/Jupyter) or prints it if no display is available.
    """
    # Use centralized configuration values if parameters are not provided.
    from config import DATA_DIR, MODEL_PATHS  # Ensure these are imported if not already

    if data_dir is None:
        data_dir = DATA_DIR
    if model_paths is None:
        model_paths = MODEL_PATHS

    # ----- Helper Functions -------------------------------------------------------- #
    def title_case(label: str) -> str:
        return " ".join(word.capitalize() for word in label.split("_"))

    def format_range(values: List[float], decimals: int = 1) -> str:
        clean_vals = [round(v, decimals) for v in values if not np.isnan(v)]
        if not clean_vals:
            return "—"
        min_val, max_val = min(clean_vals), max(clean_vals)
        if abs(min_val - max_val) < 1e-8:
            return f"{min_val:.{decimals}f}"
        return f"{min_val:.{decimals}f}–{max_val:.{decimals}f}"

    def format_list(values: List[int]) -> str:
        return f"[{','.join(str(v) for v in values)}]"

    # ----- Data Accumulation ------------------------------------------------------ #
    rows = []
    from config import (
        TASKS,
        LABEL_MAPPINGS,
    )  # Importing TASKS and LABEL_MAPPINGS from config

    tasks = TASKS

    test_tsv = os.path.join(data_dir, "MEDIC_test.tsv")
    data_df = pd.read_csv(test_tsv, sep="\t")
    total_samples = len(data_df)

    for model_name, model_path in model_paths.items():
        model = load_model(model_name, model_path)
        true_labels, predictions, files = test_model(
            model, data_dir, batch_size=BATCH_SIZE, num_threads=NUM_THREADS
        )
        metrics_dict = calculate_metrics(true_labels, predictions, LABEL_MAPPINGS)

        for task in tasks:
            class_counts = data_df[task].value_counts()
            for class_id, class_name in LABEL_MAPPINGS[task].items():
                class_df_val = 0
                if class_counts.index.dtype == object:
                    class_df_val = class_counts.get(class_name, 0)
                freq = class_df_val / total_samples if total_samples > 0 else 0

                if class_name in metrics_dict[task]["classes"]:
                    f1 = metrics_dict[task]["classes"][class_name]["f1_score"]
                    prec = metrics_dict[task]["classes"][class_name]["precision"]
                    rec = metrics_dict[task]["classes"][class_name]["recall"]
                    f1_rank = metrics_dict[task]["classes"][class_name].get(
                        "f1_rank", np.nan
                    )
                else:
                    f1, prec, rec, f1_rank = np.nan, np.nan, np.nan, np.nan

                rows.append(
                    {
                        "Model": model_name,
                        "Task": task,
                        "Class": class_name,
                        "Support": class_df_val,
                        "Frequency": freq,
                        "F1": f1,
                        "Precision": prec,
                        "Recall": rec,
                        "F1_Rank": f1_rank,
                    }
                )

    df_raw = pd.DataFrame(rows)
    if df_raw.empty:
        print("No data found for aggregated class imbalance table.")
        return

    if df_raw["F1_Rank"].isna().all():
        df_raw["F1_Rank"] = (
            df_raw.groupby(["Model", "Task"])["F1"].rank(
                method="dense", ascending=False
            )
        ).astype(int)

    agg_rows = []
    for (task, cls), subdf in df_raw.groupby(["Task", "Class"]):
        subdf = subdf.sort_values("Model")
        freqs = list(subdf["Frequency"])
        supports = list(subdf["Support"])
        f1s = list(subdf["F1"])
        precs = list(subdf["Precision"])
        recs = list(subdf["Recall"])
        ranks = list(subdf["F1_Rank"].astype(int))

        support_str = format_range(supports, decimals=0)
        freq_str = format_range(freqs, decimals=4)
        f1_str = format_range(f1s, decimals=1)
        prec_str = format_range(precs, decimals=1)
        rec_str = format_range(recs, decimals=1)
        rank_str = format_list(ranks)

        task_title = title_case(task)
        class_title = title_case(cls)
        agg_rows.append(
            {
                "Task": task_title,
                "Class": class_title,
                "Support (#)": support_str,
                "Frequency": freq_str,
                "F1": f1_str,
                "Precision": prec_str,
                "Recall": rec_str,
                "F1 Rank": rank_str,
            }
        )

    df_agg = pd.DataFrame(agg_rows)
    df_agg["Frequency_Sort"] = df_agg["Frequency"].apply(
        lambda x: float(x.strip("%")) / 100
    )
    df_agg = df_agg.sort_values(
        ["Task", "Frequency_Sort"], ascending=[True, False]
    ).drop("Frequency_Sort", axis=1)
    df_agg.rename(
        columns={"Support (#)": "Support (#)", "F1": "F1", "F1 Rank": "F1 Rank"},
        inplace=True,
    )
    df_agg["Frequency"] = df_agg["Frequency"].apply(
        lambda x: (
            f"{float(x.split('–')[0])*100:.0f}%" if "–" in x else f"{float(x)*100:.0f}%"
        )
    )
    styled = df_agg.style
    styled.hide(axis="index")
    styled.set_properties(**{"text-align": "center"})
    styled.set_properties(subset=["Task", "Class"], **{"text-align": "left"})
    styled.set_table_styles(
        [{"selector": "thead th", "props": [("text-align", "center")]}]
    )

    def highlight_task_groups(x):
        df_styles = pd.DataFrame("", index=x.index, columns=x.columns)
        mask = x["Task"].duplicated()
        df_styles.loc[mask, "Task"] = "opacity: 0"
        return df_styles

    styled = styled.apply(highlight_task_groups, axis=None)
    styled.set_caption("Aggregated Class Frequency & Performance Metrics Across Models")
    try:
        display(styled)
    except NameError:
        print(df_agg.to_string(index=False))
