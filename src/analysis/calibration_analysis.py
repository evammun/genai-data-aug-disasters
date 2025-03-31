import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from typing import Dict
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from IPython.display import display

from ..data.load_dali import get_dali_loaders
import config  # Import config as a whole


def analyze_calibration(
    data_dir: str = None,
    model_paths: Dict[str, str] = None,
    batch_size: int = None,
    num_threads: int = None,
    device: str = None,
    phase: str = "val",  # phase remains local because it changes with context
) -> None:
    """
    Displays calibration curves (reliability diagrams) inline for each model in model_paths.

    Parameters
    ----------
    data_dir : str, optional
        Path to the MEDIC dataset. Defaults to config.DATA_DIR if not provided.
    model_paths : Dict[str, str], optional
        Dictionary mapping model names to checkpoint paths. Defaults to config.get_model_paths().
    batch_size : int, optional
        Batch size for data loading. Defaults to config.BATCH_SIZE if not provided.
    num_threads : int, optional
        Number of worker threads for the DALI loader. Defaults to config.NUM_THREADS if not provided.
    device : str, optional
        Device for inference ("cuda" or "cpu"). Defaults to config.DEVICE if not provided.
    phase : str, optional
        Which split to evaluate ("val" or "test"). Defaults to "val".

    Returns
    -------
    None
    """
    from config import PROJECT_ROOT

    # Use centralized config values if parameters are not provided.
    if data_dir is None:
        data_dir = config.DATA_DIR
    if model_paths is None:
        # Instead of old config.MODEL_PATHS, use the versioned get_model_paths()
        model_paths = config.get_model_paths()
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_threads is None:
        num_threads = config.NUM_THREADS
    if device is None:
        device = config.DEVICE

    # --- Seaborn styling: use centralized theme settings ---
    sns.set_theme(style=config.SNS_STYLE)
    sns.set_context(config.SNS_CONTEXT, font_scale=config.SNS_FONT_SCALE)
    palette_name = config.SNS_PALETTE

    # --- Build the DALI loader for the chosen phase ---
    # We'll pick the GPU device ID from the current device
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
    loaders = get_dali_loaders(data_dir, batch_size, num_threads, device_id, phase)
    if phase == "val":
        loader, total_batches, _ = loaders[1]
    elif phase == "test":
        loader, total_batches, _ = loaders[2]
    else:
        raise ValueError(f"Invalid phase '{phase}'. Must be 'val' or 'test'.")

    tasks = config.TASKS
    n_models = len(model_paths)
    n_tasks = len(tasks)

    fig, axes = plt.subplots(
        nrows=n_models, ncols=n_tasks, figsize=(20, 12), sharey=False
    )
    fig.suptitle(f"Calibration Curves ({phase.capitalize()}) - All Models", fontsize=16)

    model_items = list(model_paths.items())
    brier_data = []

    for row_idx, (model_name, ckpt_path) in enumerate(model_items):
        # Ensure path is absolute
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(PROJECT_ROOT, "WSLcode", ckpt_path)

        # --- Load model ---
        print(f"[analyze_calibration] Loading {model_name} from: {ckpt_path}")
        model = torch.load(ckpt_path, map_location=device)
        model.to(device)
        model.eval()

        # Prepare storage for true labels & predicted probs
        prob_storage = {t: {} for t in tasks}
        true_storage = {t: {} for t in tasks}
        for task in tasks:
            for class_id, class_name in config.LABEL_MAPPINGS[task].items():
                prob_storage[task][class_name] = []
                true_storage[task][class_name] = []

        # --- Inference ---
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                inputs = batch[0]["data"].to(device)
                combined_label = batch[0]["labels"].to(device)

                # Rebuild each label by decoding the combined integer
                labels = {
                    "damage_severity": combined_label // 1000,
                    "humanitarian": (combined_label % 1000) // 100,
                    "informative": (combined_label % 100) // 10,
                    "disaster_types": combined_label % 10,
                }
                for t in labels:
                    labels[t] = labels[t].view(-1).long()

                outputs = model(inputs)

                # For each task, store predicted probabilities & true labels
                for task in tasks:
                    logits = outputs[task]
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    gt = labels[task].cpu().numpy()
                    for class_id, class_name in config.LABEL_MAPPINGS[task].items():
                        y_true_bin = (gt == class_id).astype(int)
                        y_prob = probs[:, class_id]
                        true_storage[task][class_name].extend(y_true_bin.tolist())
                        prob_storage[task][class_name].extend(y_prob.tolist())

        # --- Plotting for each task ---
        for col_idx, task in enumerate(tasks):
            ax = axes[row_idx, col_idx]
            ax.plot([0, 1], [0, 1], "k:", label="Perfect")

            n_classes = len(config.LABEL_MAPPINGS[task])
            colors = sns.color_palette(palette_name, n_colors=n_classes)

            for i, (class_id, class_name) in enumerate(
                config.LABEL_MAPPINGS[task].items()
            ):
                cls_label = class_name.replace("_", " ").title()
                y_true = np.array(true_storage[task][class_name])
                y_prob = np.array(prob_storage[task][class_name])
                if y_true.size == 0:
                    continue
                frac_of_pos, mean_pred = calibration_curve(
                    y_true, y_prob, n_bins=10, strategy="uniform"
                )
                brier = brier_score_loss(y_true, y_prob)
                ax.plot(
                    mean_pred,
                    frac_of_pos,
                    label=cls_label,
                    linestyle="-",
                    linewidth=2,
                    color=colors[i],
                )
                brier_data.append(
                    {
                        "Model": model_name,
                        "Task": task.replace("_", " ").title(),
                        "Class": cls_label,
                        "Brier Score": brier,
                    }
                )

            if row_idx == 0:
                ax.set_title(task.replace("_", " ").title())
            if col_idx == 0:
                ax.set_ylabel(f"{model_name}\nFraction of Positives", fontsize=12)
            else:
                ax.set_ylabel("")
            if row_idx == n_models - 1:
                ax.set_xlabel("Predicted Probability")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.show()

    # --- Summaries of Brier Scores ---
    df_brier = pd.DataFrame(brier_data)
    grouped = (
        df_brier.groupby(["Task", "Class"])["Brier Score"]
        .agg(["min", "max"])
        .reset_index()
    )

    def format_range(row):
        b_min = round(row["min"], 3)
        b_max = round(row["max"], 3)
        return f"{b_min:.3f}" if b_min == b_max else f"{b_min:.3f}–{b_max:.3f}"

    grouped["Brier Range"] = grouped.apply(format_range, axis=1)
    grouped.drop(columns=["min", "max"], inplace=True)
    grouped.sort_values(["Task", "Class"], inplace=True)

    # Collapse repeated task names for a neat table
    prev_task = None
    for idx, row in grouped.iterrows():
        current_task = row["Task"]
        if current_task == prev_task:
            grouped.at[idx, "Task"] = ""
        prev_task = current_task

    text_cols = ["Task", "Class"]
    num_cols = ["Brier Range"]
    df_style = (
        grouped.style.hide(axis="index")
        .set_caption("Brier Score Ranges per (Task, Class) – Lower = Better")
        .set_properties(subset=text_cols, **{"text-align": "left"})
        .set_properties(subset=num_cols, **{"text-align": "center"})
    )
    display(df_style)
