import os
import inspect
import random
import pandas as pd
import numpy as np
from IPython.display import display, Image
from typing import Dict, List
import base64

from src.core.model_tester import load_model, test_model
from src.data.load_dali import get_dali_loaders

# Instead of importing label_mappings locally, import from config.
from config import LABEL_MAPPINGS, DATA_DIR, MODEL_PATHS, BATCH_SIZE, NUM_THREADS, TASKS


def create_prediction_dataframe(
    data_dir: str = None,
    model_paths: Dict[str, str] = None,
    batch_size: int = None,
    num_threads: int = None,
) -> pd.DataFrame:
    """
    1) For each model in model_paths, run test_model, which returns
       (true_labels_dict, predictions_dict, file_paths).
    2) Build a "matched" DataFrame that merges the original TSV rows with
       the correct predictions/labels by file path.
    3) New columns are created (e.g. "damage_severity_true_matched" and
       "damage_severity_{model_name}_pred_matched") to avoid indexing mismatches.

    Returns
    -------
    A DataFrame with one row per test image, containing:
       - 'full_img_path'
       - Matched true labels for each task (e.g. "damage_severity_true_matched")
       - Matched predictions for each model (e.g. "damage_severity_{model_name}_pred_matched")
    """
    # Use config defaults if not provided.
    if data_dir is None:
        data_dir = DATA_DIR
    if model_paths is None:
        model_paths = MODEL_PATHS
    if batch_size is None:
        batch_size = BATCH_SIZE
    if num_threads is None:
        num_threads = NUM_THREADS

    # Load the test TSV
    test_tsv = os.path.join(data_dir, "MEDIC_test.tsv")
    df_test = pd.read_csv(test_tsv, sep="\t")

    # Create "full_img_path" column in the same style as used by get_dali_loaders.
    df_test["full_img_path"] = df_test["image_path"].apply(
        lambda x: os.path.join(data_dir, x)
    )

    # Initialize storage for ground truth and predictions.
    all_true = None  # Will store the first model's true_labels_dict
    all_preds = {}  # Predictions per model
    # Use TASKS from config; if desired, you can change the order here.
    tasks = TASKS
    pipeline_files = None

    # Process each model
    for model_name, model_path in model_paths.items():
        # test_model now returns three items.
        true_labels_dict, predictions_dict, file_paths = test_model(
            load_model(model_name, model_path),
            data_dir,
            batch_size=batch_size,
            num_threads=num_threads,
        )
        if pipeline_files is None:
            pipeline_files = file_paths
        all_preds[model_name] = predictions_dict
        if all_true is None:
            all_true = true_labels_dict

    # Build a mapping from full image path to pipeline index.
    path_to_index = {path: i for i, path in enumerate(pipeline_files)}

    # Create new columns for each task and model.
    for task in tasks:
        df_test[f"{task}_true_matched"] = None
        for model_name in model_paths.keys():
            df_test[f"{task}_{model_name}_pred_matched"] = None

    # Helper: decode an integer label into text using centralized LABEL_MAPPINGS.
    def decode_label(task: str, val) -> str:
        val = int(val)
        return LABEL_MAPPINGS[task].get(val, "UNKNOWN")

    # For each row in the test DataFrame, look up the matching pipeline sample.
    for idx in df_test.index:
        fpath = df_test.at[idx, "full_img_path"]
        if fpath not in path_to_index:
            continue
        sample_i = path_to_index[fpath]
        # Fill in the true label for each task.
        for task in tasks:
            raw_val = all_true[task][sample_i]
            df_test.at[idx, f"{task}_true_matched"] = decode_label(task, raw_val)
        # Fill in predictions for each model and task.
        for model_name in model_paths.keys():
            for task in tasks:
                raw_pred = all_preds[model_name][task][sample_i]
                col_pred = f"{task}_{model_name}_pred_matched"
                df_test.at[idx, col_pred] = decode_label(task, raw_pred)

    return df_test


def run_qualitative_audit(
    data_dir: str = None,
    model_paths: Dict[str, str] = None,
    classes_to_inspect: Dict[str, List[str]] = None,
    misclassified_by: int = None,
    sample_size: int = None,
    images_to_show: int = None,
    batch_size: int = None,
    num_threads: int = None,
) -> None:
    """
    1. Builds a DataFrame of test predictions and ground truth, aligned by file path.
    2. Filters rows where, for each task in classes_to_inspect, the true label matches one of
       the specified classes.
    3. Keeps images misclassified by at least 'misclassified_by' models for each task.
    4. Displays up to 'images_to_show' images side-by-side with a table of matched tasks.

    Relies on columns such as "damage_severity_true_matched" and "damage_severity_{model}_pred_matched"
    for alignment.
    """
    # Use config defaults where applicable.
    from config import DATA_DIR, MODEL_PATHS, BATCH_SIZE, NUM_THREADS, TASKS

    if data_dir is None:
        data_dir = DATA_DIR
    if model_paths is None:
        model_paths = MODEL_PATHS
    if batch_size is None:
        batch_size = BATCH_SIZE
    if num_threads is None:
        num_threads = NUM_THREADS
    if misclassified_by is None:
        misclassified_by = 1  # Default threshold; adjust as needed.
    if sample_size is None:
        sample_size = 20
    if images_to_show is None:
        images_to_show = 10
    if classes_to_inspect is None:
        # Default: inspect all classes for all tasks (or adjust as desired)
        classes_to_inspect = {
            task: list(LABEL_MAPPINGS[task].values()) for task in TASKS
        }

    # Build the matched predictions DataFrame.
    df = create_prediction_dataframe(
        data_dir, model_paths, batch_size=batch_size, num_threads=num_threads
    )
    model_list = list(model_paths.keys())
    # The user-specified tasks to check.
    relevant_tasks = list(classes_to_inspect.keys())

    # 1. Filter rows based on true labels.
    mask = pd.Series(True, index=df.index)
    for task, class_list in classes_to_inspect.items():
        col_true_matched = f"{task}_true_matched"
        mask = mask & df[col_true_matched].isin(class_list)
    filtered = df[mask].copy()
    if filtered.empty:
        print("No rows match the specified 'true' labels across all tasks.")
        return

    # 2. Require that at least 'misclassified_by' models are wrong for each relevant task.
    def meets_misclf(row):
        for t in relevant_tasks:
            true_val = row[f"{t}_true_matched"]
            wrong_count = sum(
                row[f"{t}_{m}_pred_matched"] != true_val for m in model_list
            )
            if wrong_count < misclassified_by:
                return False
        return True

    filtered["meets_misclf_requirement"] = filtered.apply(meets_misclf, axis=1)
    final_subset = filtered[filtered["meets_misclf_requirement"]]
    if final_subset.empty:
        print("No images found with the required misclassification count across tasks.")
        return

    # 3. Randomly sample up to sample_size rows.
    if len(final_subset) > sample_size:
        final_subset = final_subset.sample(sample_size)

    print(
        f"Found {len(final_subset)} candidate images; showing up to {images_to_show} below:\n"
    )
    # 4. Randomly choose up to images_to_show images and display them.
    chosen = final_subset.sample(min(images_to_show, len(final_subset)))
    for _, row in chosen.iterrows():
        display_image_row_side_by_side(row, model_list, relevant_tasks)
        print("-" * 70)
    print()


def display_image_row_side_by_side(
    row: pd.Series, model_list: List[str], relevant_tasks: List[str]
):
    """
    Displays an image and a mini-table side by side showing the '..._true_matched'
    and '..._{model}_pred_matched' columns for each relevant task.
    """
    import os
    from IPython.display import HTML, display

    final_path = row.get("full_img_path", None)
    if not final_path or not os.path.exists(final_path):
        img_html = f'<p style="color:red;">Image not found:<br>{final_path}</p>'
    else:
        try:
            with open(final_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(final_path)[1].lower()
            mime = "image/png"
            if ext in [".jpg", ".jpeg"]:
                mime = "image/jpeg"
            img_html = f'<img src="data:{mime};base64,{encoded}" width="400" />'
        except Exception as e:
            img_html = f'<p style="color:red;">Error reading file:<br>{e}</p>'

    rows_labels = ["true"] + model_list
    col_names = [t.replace("_", " ").title() for t in relevant_tasks]
    table_data = []
    for row_label in rows_labels:
        row_values = []
        for t in relevant_tasks:
            if row_label == "true":
                val = row.get(f"{t}_true_matched", "???")
            else:
                val = row.get(f"{t}_{row_label}_pred_matched", "???")
            row_values.append(val)
        table_data.append(row_values)

    df_table = pd.DataFrame(table_data, index=rows_labels, columns=col_names)
    table_html = df_table.to_html(escape=False)
    combined_html = f"""
    <div style="display: flex; align-items: flex-start;">
      <div style="margin-right: 20px;">
        {img_html}
      </div>
      <div>
        {table_html}
      </div>
    </div>
    """
    display(HTML(combined_html))
