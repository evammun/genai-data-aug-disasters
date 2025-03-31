"""
label_consensus.py

This module provides two main functions:

1) run_label_consensus_inference(...):
   - Merges train/dev/test TSVs into one DataFrame.
   - Normalises ground-truth textual labels so they match your label mappings.
   - Runs inference with multiple models to get predicted labels.
   - Saves the resulting DataFrame (with columns 'pred_*_<model>') to a TSV.

2) label_consensus_analysis(...):
   - Loads (or takes in-memory) the TSV from step 1.
   - Computes misclassification pivot tables for each task.
   - Displays a 3-model (or N-model) consensus confusion matrix, 
     showing how often the ensemble disagrees with the official label.
   - Returns the final DataFrame (optionally you can do more analysis).

By separating inference from analysis, you avoid re-running inference 
(which can take ~10 min) every time you tweak your analysis code.
"""

import os
import warnings
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

warnings.simplefilter("ignore")  # Suppress common PIL & future warnings


# --------------------------------------------------------------------------
# A) Utility: Merging TSVs and Normalising the Ground-Truth Labels
# --------------------------------------------------------------------------
def get_consolidated_tsv(data_dir):
    """
    Reads the original train/dev/test TSVs, appends a 'split' column,
    and concatenates them into a single DataFrame with columns:
       [image_id, event_name, image_path, damage_severity, informative,
        humanitarian, disaster_types, split]
    """
    train_tsv = os.path.join(data_dir, "MEDIC_train.tsv")
    dev_tsv = os.path.join(data_dir, "MEDIC_dev.tsv")
    test_tsv = os.path.join(data_dir, "MEDIC_test.tsv")

    df_train = pd.read_csv(train_tsv, sep="\t")
    df_dev = pd.read_csv(dev_tsv, sep="\t")
    df_test = pd.read_csv(test_tsv, sep="\t")

    df_train["split"] = "train"
    df_dev["split"] = "val"
    df_test["split"] = "test"

    return pd.concat([df_train, df_dev, df_test], ignore_index=True)


def unify_label_text(lbl, mapping_dict):
    """
    Ensures the textual label 'lbl' matches one of the values in 'mapping_dict'.
    'mapping_dict' is something like {0: "little_or_none", 1: "mild", 2: "severe"}.
    If 'lbl' can be matched ignoring case/underscore differences, we unify it.
    Otherwise, we return it as-lowercase-with-underscores if nothing matches.

    This helps prevent pivot-table mismatches between ground-truth label strings
    and the predicted label strings produced by the model.
    """
    if pd.isna(lbl):
        return None

    # We'll convert to lower, replace spaces with underscores, etc.
    # Then see if that matches any mapping exactly
    norm = lbl.strip().lower().replace(" ", "_")

    # Some data might have "none" vs "little_or_none", or synonyms. You can add logic.
    # We'll do a quick best-effort check:
    for k, v in mapping_dict.items():
        # v is the canonical string like "little_or_none"
        if v == norm:
            return v  # matched exactly
        # Or if it differs only slightly, you could do partial matching

    # If no exact match, we fallback to the normalised string we generated
    return norm


# --------------------------------------------------------------------------
# B) Dataset & Inference (No Debug)
# --------------------------------------------------------------------------
class MEDICDataset(Dataset):
    """
    Simple dataset that loads each row's image_path,
    returns "df_index" and the 4 textual labels (for reference).
    """

    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.data_dir = data_dir
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["image_path"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        sample = {
            "df_index": row["df_index"],
            "damage_severity": row["damage_severity"],
            "informative": row["informative"],
            "humanitarian": row["humanitarian"],
            "disaster_types": row["disaster_types"],
            "image": image,
        }
        return sample


def run_inference_one_model(model, df, data_dir, label_mappings, batch_size=32):
    """
    Runs a single model on all images (df) in small batches.
    Returns a small DF with columns:
      df_index,
      pred_damage_severity,
      pred_informative,
      pred_humanitarian,
      pred_disaster_types
    """
    device = next(model.parameters()).device

    infer_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = MEDICDataset(df, data_dir, transform=infer_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # We'll accumulate results
    all_idx = []
    all_ds = []
    all_inf = []
    all_hum = []
    all_dis = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            b_idx = batch["df_index"].cpu().numpy()
            imgs = batch["image"].to(device)

            outputs = model(imgs)  # dict with keys: damage_severity, etc.

            # Argmax => textual label
            _, ds_pred = torch.max(outputs["damage_severity"], 1)
            _, inf_pred = torch.max(outputs["informative"], 1)
            _, hum_pred = torch.max(outputs["humanitarian"], 1)
            _, dis_pred = torch.max(outputs["disaster_types"], 1)

            ds_pred = ds_pred.cpu().numpy()
            inf_pred = inf_pred.cpu().numpy()
            hum_pred = hum_pred.cpu().numpy()
            dis_pred = dis_pred.cpu().numpy()

            # Convert ID -> text
            for i in range(len(b_idx)):
                idxi = b_idx[i]
                # The label mappings are something like {0:"little_or_none",1:"mild",2:"severe"}
                ds_txt = label_mappings["damage_severity"][ds_pred[i]]
                inf_txt = label_mappings["informative"][inf_pred[i]]
                hum_txt = label_mappings["humanitarian"][hum_pred[i]]
                dis_txt = label_mappings["disaster_types"][dis_pred[i]]

                all_idx.append(idxi)
                all_ds.append(ds_txt)
                all_inf.append(inf_txt)
                all_hum.append(hum_txt)
                all_dis.append(dis_txt)

    return pd.DataFrame(
        {
            "df_index": all_idx,
            "pred_damage_severity": all_ds,
            "pred_informative": all_inf,
            "pred_humanitarian": all_hum,
            "pred_disaster_types": all_dis,
        }
    )


# --------------------------------------------------------------------------
# C) Main Public Function #1: run_label_consensus_inference
# --------------------------------------------------------------------------
def run_label_consensus_inference(
    data_dir,
    model_paths_dict,
    batch_size=32,
    output_tsv="src/labelling/consolidated_with_predictions.tsv",
):
    """
    1) Merges train/val/test TSV => single df_merged
    2) Normalises ground-truth textual labels to match your label mappings
    3) For each model, runs inference => predicted columns
    4) Saves the final df (with columns [pred_*_<modelname>] ) to TSV

    NOTE: This function does NOT produce pivot tables.
    That analysis is done by label_consensus_analysis(...).
    """
    from config import DEVICE, LABEL_MAPPINGS
    from ..core.model_tester import load_model

    # Merge data
    df_merged = get_consolidated_tsv(data_dir)
    df_merged.reset_index(drop=True, inplace=True)

    # Add a df_index for stable referencing
    df_merged["df_index"] = df_merged.index
    # Reorder columns to put df_index up front
    cols = ["df_index"] + [c for c in df_merged.columns if c != "df_index"]
    df_merged = df_merged[cols]

    # Normalise ground-truth labels so pivot table won't mismatch
    tasks = ["damage_severity", "informative", "humanitarian", "disaster_types"]
    for t in tasks:
        mapping_dict = LABEL_MAPPINGS[
            t
        ]  # e.g. {0:"little_or_none",1:"mild",2:"severe",...}
        df_merged[t] = df_merged[t].apply(lambda x: unify_label_text(x, mapping_dict))

    df_final = df_merged.copy()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # Inference for each model
    for model_name, ckpt_path in model_paths_dict.items():
        model = load_model(model_name, ckpt_path)
        model.to(device)

        df_preds = run_inference_one_model(
            model=model,
            df=df_merged,
            data_dir=data_dir,
            label_mappings=LABEL_MAPPINGS,
            batch_size=batch_size,
        )
        # rename predicted columns
        df_preds.rename(
            columns={
                "pred_damage_severity": f"pred_damage_severity_{model_name}",
                "pred_informative": f"pred_informative_{model_name}",
                "pred_humanitarian": f"pred_humanitarian_{model_name}",
                "pred_disaster_types": f"pred_disaster_types_{model_name}",
            },
            inplace=True,
        )

        df_final = df_final.merge(df_preds, how="left", on="df_index")

    # Save
    df_final.to_csv(output_tsv, sep="\t", index=False)
    return df_final


# --------------------------------------------------------------------------
# D) Main Public Function #2: label_consensus_analysis
# --------------------------------------------------------------------------
import pandas as pd
from IPython.display import display, HTML
from collections import Counter


import pandas as pd
from IPython.display import display, HTML
from collections import Counter


import pandas as pd
from IPython.display import display, HTML
from collections import Counter


import pandas as pd
from IPython.display import display, HTML
from collections import Counter


import pandas as pd
from IPython.display import display
from collections import Counter


import pandas as pd
from IPython.display import display
from collections import Counter


import pandas as pd
from IPython.display import display
from collections import Counter


import pandas as pd
from IPython.display import display
from collections import Counter


def label_consensus_analysis(df, model_names=None):
    """
    Produces exactly two styled tables in a Jupyter notebook with no extraneous prints.

    TABLE 1: #Models Misclassified (0..3,All)
      - Single 'Label' column with an indented Class row or a bold Task row.
      - Columns = [0,1,2,3,"All"].
      - The 0..3 columns => 'count (xx%)' with row-based percentages.
      - The "All" column => count-only (no '100%').
      - At the end, a final row for 'All Tasks'.

    TABLE 2: Largest Alt-Label Agreement among misclassifying models
      - Same row structure, columns=[1,2,3,"All"].
      - 'All' column => count-only, columns 1..3 => 'count (xx%)' row-based.
      - Also a final 'All Tasks' row.

    No big DF or extraneous text is printed. Just two styled tables.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ground truth columns (damage_severity, informative,
        humanitarian, disaster_types) plus predicted columns for each model
        (pred_damage_severity_<model>, etc.).
    model_names : list of str, optional
        E.g. ["resnet50","efficientnet_b1","mobilenet_v2"]. Default if None.
    """
    if model_names is None:
        model_names = ["resnet50", "efficientnet_b1", "mobilenet_v2"]

    tasks = ["damage_severity", "informative", "humanitarian", "disaster_types"]

    # Helper to format
    def bold(s: str) -> str:
        return f"<strong>{s}</strong>"

    def proper_case(s: str) -> str:
        return s.replace("_", " ").title()

    def indent(cls_name: str) -> str:
        return "&nbsp;&nbsp;" + cls_name

    def get_preds(row, t):
        preds = []
        for m in model_names:
            col = f"pred_{t}_{m}"
            val = row.get(col, None)
            preds.append(val if pd.notna(val) else None)
        return preds

    # ----------------------------------------------------------------
    # TABLE 1: #Models Misclassified (0..3,All)
    # ----------------------------------------------------------------
    records1 = []
    for _, row in df.iterrows():
        for t in tasks:
            if t not in row or pd.isna(row[t]):
                continue
            gt = row[t]
            preds = get_preds(row, t)
            miscount = sum((p != gt) for p in preds if p is not None)
            records1.append((t, gt, miscount))

    df_mis = pd.DataFrame(records1, columns=["task", "class", "miscount"])
    tab1 = df_mis.pivot_table(
        index=["task", "class"], columns="miscount", aggfunc="size", fill_value=0
    )
    for c in [0, 1, 2, 3]:
        if c not in tab1.columns:
            tab1[c] = 0
    tab1 = tab1[[0, 1, 2, 3]]  # reorder
    tab1["All"] = tab1.sum(axis=1)
    tab1 = tab1.sort_index()

    # We'll accumulate "label + col0..3 + colAll"
    rows1 = []
    overall1 = tab1.sum(numeric_only=True)
    # group by first level => tasks, but we'll just iterate entire index (task,class)
    # in a single pass, checking if the "task" changed
    current_task = None
    task_sum_series = None
    # we'll store partial sums for each task. But let's do simpler approach:
    # after building the entire pivot, we do subdf = tab1.xs(task_name, level="task", drop_level=False).
    # Then we produce class rows, then a bold row for the task.
    # Finally a grand total row.

    table1_rows = []
    # gather tasks from the pivot:
    unique_tasks = tab1.index.levels[0]
    # We'll do the sums for each task after the subdf
    for tk in unique_tasks:
        if tk not in tab1.index:
            continue
        subdf = tab1.xs(tk, level="task", drop_level=False)
        # create class rows
        for (task_val, cls_val), rowvals in subdf.iterrows():
            # rowvals => columns 0..3 + All
            row_sum = int(rowvals["All"])
            row_cols = []
            for c in [0, 1, 2, 3]:
                v = int(rowvals[c])
                pct = int(round(100 * v / row_sum)) if row_sum > 0 else 0
                row_cols.append(f"{v:,} ({pct}%)")
            row_cols.append(f"{row_sum:,}")
            row_label = indent(proper_case(cls_val))
            table1_rows.append((tk, row_label, row_cols))

        # now the bold row for the task
        t_sum = subdf.sum()
        rsum_t = int(t_sum["All"])
        t_cols = []
        for c in [0, 1, 2, 3]:
            val = int(t_sum[c])
            pct = int(round(100 * val / rsum_t)) if rsum_t > 0 else 0
            t_cols.append(f"{val:,} ({pct}%)")
        t_cols.append(f"{rsum_t:,}")
        table1_rows.append((tk, bold(proper_case(tk)), t_cols))

    # a final "ALL TASKS" row
    sum_all_1 = int(overall1["All"])
    final_cols_1 = []
    for c in [0, 1, 2, 3]:
        v = int(overall1[c])
        pct = int(round(100 * v / sum_all_1)) if sum_all_1 > 0 else 0
        final_cols_1.append(f"{v:,} ({pct}%)")
    final_cols_1.append(f"{sum_all_1:,}")
    table1_rows.append(("ALL TASKS", bold("All Tasks"), final_cols_1))

    # flatten
    table1_list = []
    for task_val, row_lab, colvals in table1_rows:
        table1_list.append([row_lab] + colvals)
    df_tab1 = pd.DataFrame(table1_list, columns=["Label", "0", "1", "2", "3", "All"])

    # ----------------------------------------------------------------
    # TABLE 2: Largest Alt-Label cluster (1..3,All)
    # ----------------------------------------------------------------
    records2 = []
    for _, row in df.iterrows():
        for t in tasks:
            if t not in row or pd.isna(row[t]):
                continue
            gt = row[t]
            preds = get_preds(row, t)
            alt = [p for p in preds if (p is not None and p != gt)]
            if not alt:
                continue
            ccount = Counter(alt)
            largest = min(ccount.most_common(1)[0][1], 3)
            records2.append((t, gt, largest))

    df_alt = pd.DataFrame(records2, columns=["task", "class", "largest_alt"])
    tab2 = df_alt.pivot_table(
        index=["task", "class"], columns="largest_alt", aggfunc="size", fill_value=0
    )
    for c in [1, 2, 3]:
        if c not in tab2.columns:
            tab2[c] = 0
    tab2 = tab2[[1, 2, 3]]
    tab2["All"] = tab2.sum(axis=1)
    tab2 = tab2.sort_index()

    table2_rows = []
    overall2 = tab2.sum(numeric_only=True)
    # group by tasks
    unique_tasks_2 = tab2.index.levels[0]
    for tk in unique_tasks_2:
        if tk not in tab2.index:
            continue
        subdf = tab2.xs(tk, level="task", drop_level=False)
        for (task_val, cls_val), rowvals in subdf.iterrows():
            rsum = int(rowvals["All"])
            rowc = []
            for c in [1, 2, 3]:
                v = int(rowvals[c])
                pct = int(round(100 * v / rsum)) if rsum > 0 else 0
                rowc.append(f"{v:,} ({pct}%)")
            rowc.append(f"{rsum:,}")
            row_label = indent(proper_case(cls_val))
            table2_rows.append((tk, row_label, rowc))

        # bold row for the task
        t_sum = subdf.sum()
        rsum_t2 = int(t_sum["All"])
        tcols2 = []
        for c in [1, 2, 3]:
            vv = int(t_sum[c])
            pct = int(round(100 * vv / rsum_t2)) if rsum_t2 > 0 else 0
            tcols2.append(f"{vv:,} ({pct}%)")
        tcols2.append(f"{rsum_t2:,}")
        table2_rows.append((tk, bold(proper_case(tk)), tcols2))

    # final ALL TASKS row
    sum_all_2 = int(overall2["All"])
    final2 = []
    for c in [1, 2, 3]:
        v2 = int(overall2[c])
        pct2 = int(round(100 * v2 / sum_all_2)) if sum_all_2 > 0 else 0
        final2.append(f"{v2:,} ({pct2}%)")
    final2.append(f"{sum_all_2:,}")
    table2_rows.append(("ALL TASKS", bold("All Tasks"), final2))

    table2_list = []
    for task_val, row_lab, colvals in table2_rows:
        table2_list.append([row_lab] + colvals)
    df_tab2 = pd.DataFrame(table2_list, columns=["Label", "1", "2", "3", "All"])

    # ----------------------------------------------------------------
    # STYLING
    # ----------------------------------------------------------------
    def is_bold_html(val):
        return "<strong>" in str(val)

    def align_label(val):
        if is_bold_html(val):
            return "text-align:center;"
        else:
            return "text-align:left;"

    def style_table(df_, caption):
        data_cols = df_.columns[1:]  # numeric columns
        sty = df_.style.set_caption(caption)
        sty = sty.applymap(align_label, subset=["Label"])
        sty = sty.set_properties(
            subset=data_cols, **{"text-align": "right", "padding": "4px"}
        )
        sty = sty.set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("text-align", "center"),
                        ("background-color", "#2F2F2F"),
                        ("color", "white"),
                        ("padding", "4px"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("background-color", "#1E1E1E"),
                        ("color", "white"),
                        ("border", "1px solid #444"),
                    ],
                },
            ]
        )
        return sty

    sty1 = style_table(df_tab1, "TABLE 1: #Models Misclassified (0..3,All)")
    sty2 = style_table(df_tab2, "TABLE 2: Largest Alt-Label Agreement (1,2,3,All)")

    display(sty1, sty2)
    return
