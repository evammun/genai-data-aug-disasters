import os
import pandas as pd
from collections import Counter
from IPython.display import display
from tqdm import tqdm  # <-- for progress bar

from .llm_calls import call_claude_api, call_gpt_api

VALID_DAMAGE_SEVERITY = {"little_or_none", "mild", "severe"}
VALID_INFORMATIVE = {"informative", "not_informative"}
VALID_HUMANITARIAN = {
    "affected_injured_or_dead_people",
    "infrastructure_and_utility_damage",
    "not_humanitarian",
    "rescue_volunteering_or_donation_effort",
}
VALID_DISASTER_TYPES = {
    "earthquake",
    "fire",
    "flood",
    "hurricane",
    "landslide",
    "not_disaster",
    "other_disaster",
}


def validate_llm_labels(label_dict):
    """
    Checks that label_dict has 4 keys and each value is in the correct set.
    Returns True if valid, False otherwise.
    """
    if not isinstance(label_dict, dict):
        return False

    required_keys = ["damage_severity", "informative", "humanitarian", "disaster_types"]
    for k in required_keys:
        if k not in label_dict:
            return False

    ds = label_dict["damage_severity"]
    inf = label_dict["informative"]
    hum = label_dict["humanitarian"]
    dis = label_dict["disaster_types"]

    if ds not in VALID_DAMAGE_SEVERITY:
        return False
    if inf not in VALID_INFORMATIVE:
        return False
    if hum not in VALID_HUMANITARIAN:
        return False
    if dis not in VALID_DISASTER_TYPES:
        return False

    return True


def relabel_medic_dataset(
    input_tsv,
    data_dir,
    output_dir,
    min_num_models_misclassified=1,
    reclassify_limit=None,
):
    """
    The main pipeline:
      1) Load the dataset and store 'orig_{task}' for each row to preserve original ground truth.
      2) Flag images (those with >= min_num_models_misclassified), call Claude/GPT,
         produce final_{task}, etc.
      3) Overwrite df[t] with final_{t}.
      4) Save train/val/test_relabelled, flagged_intermediate, llm_failures.
      5) Generate the "Largest Alt-Label Agreement (1,2,3,All) + Relabelled" table across *all* rows.

    Note: If reclassify_limit is larger than the number of flagged rows,
    it has no effect (so no error occurs).
    """

    from config import PROJECT_ROOT

    # 1) Fix up paths if needed
    if not os.path.isabs(input_tsv):
        input_tsv = os.path.join(PROJECT_ROOT, "WSLcode", input_tsv)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, "WSLcode", output_dir)

    tasks = ["damage_severity", "informative", "humanitarian", "disaster_types"]
    models = ["resnet50", "efficientnet_b1", "mobilenet_v2"]

    # 2) Read data
    df = pd.read_csv(input_tsv, sep="\t")

    # Store original ground-truth for each row
    for t in tasks:
        df[f"orig_{t}"] = df[t]

    # Count misclassifications, filter flagged
    mismatch_counts = []
    for _, row in df.iterrows():
        mismatch = 0
        for t in tasks:
            gt_label = row[t]
            for m in models:
                pred_col = f"pred_{t}_{m}"
                if pred_col not in df.columns:
                    continue
                if row[pred_col] != gt_label:
                    mismatch += 1
        mismatch_counts.append(mismatch)
    df["total_misclassifications"] = mismatch_counts

    flagged_df = df[
        df["total_misclassifications"] >= min_num_models_misclassified
    ].copy()

    print("Total images qualifying for relabelling: ", str(len(flagged_df)))
    if reclassify_limit is not None and reclassify_limit < len(flagged_df):
        flagged_df = flagged_df.iloc[:reclassify_limit].copy()

    print(
        f"[INFO] Found {len(flagged_df)} images to re-classify (threshold={min_num_models_misclassified})."
    )

    # Create final_{t} columns
    for t in tasks:
        df[f"final_{t}"] = df[t]
        flagged_df[f"final_{t}"] = flagged_df[t]

    # Also placeholders for LLM outputs in flagged_df
    llm_cols = [
        "claude_damage_severity",
        "claude_informative",
        "claude_humanitarian",
        "claude_disaster_types",
        "gpt_damage_severity",
        "gpt_informative",
        "gpt_humanitarian",
        "gpt_disaster_types",
    ]
    for c in llm_cols:
        flagged_df[c] = None

    # For logging LLM fails
    failures = []
    claude_fail_count = 0
    gpt_fail_count = 0

    # 3) Re-labelling steps for flagged
    for idx in tqdm(flagged_df.index, desc="Relabelling"):
        row = flagged_df.loc[idx]
        image_rel_path = row["image_path"]
        fullpath = os.path.join(data_dir, image_rel_path)

        # Call Claude
        claude_labels, claude_fail = call_claude_api(fullpath)
        if claude_fail:
            claude_fail["image_path"] = image_rel_path
            failures.append(claude_fail)
            claude_fail_count += 1
            claude_labels = {}

        # Call GPT
        gpt_labels, gpt_fail = call_gpt_api(fullpath)
        if gpt_fail:
            gpt_fail["image_path"] = image_rel_path
            failures.append(gpt_fail)
            gpt_fail_count += 1
            gpt_labels = {}

        # Validate each
        if not validate_llm_labels(claude_labels):
            claude_labels = {}
        if not validate_llm_labels(gpt_labels):
            gpt_labels = {}

        # If both valid & match => adopt c_val else keep orig
        for t in tasks:
            c_val = claude_labels.get(t)
            g_val = gpt_labels.get(t)
            orig_val = row[f"orig_{t}"]
            if c_val and g_val and c_val == g_val:
                final_val = c_val
            else:
                final_val = orig_val
            flagged_df.at[idx, f"final_{t}"] = final_val
            df.at[idx, f"final_{t}"] = final_val

        # Store whichever are valid in separate columns
        if len(claude_labels) == 4:
            flagged_df.at[idx, "claude_damage_severity"] = claude_labels[
                "damage_severity"
            ]
            flagged_df.at[idx, "claude_informative"] = claude_labels["informative"]
            flagged_df.at[idx, "claude_humanitarian"] = claude_labels["humanitarian"]
            flagged_df.at[idx, "claude_disaster_types"] = claude_labels[
                "disaster_types"
            ]

        if len(gpt_labels) == 4:
            flagged_df.at[idx, "gpt_damage_severity"] = gpt_labels["damage_severity"]
            flagged_df.at[idx, "gpt_informative"] = gpt_labels["informative"]
            flagged_df.at[idx, "gpt_humanitarian"] = gpt_labels["humanitarian"]
            flagged_df.at[idx, "gpt_disaster_types"] = gpt_labels["disaster_types"]

    # Overwrite in df
    for t in tasks:
        df[t] = df[f"final_{t}"]

    # 4) Save flagged_relabelling_intermediate
    col_order = ["image_id", "event_name", "image_path"]

    def grouped_task_cols(task_name):
        out = []
        for m in models:
            pc = f"pred_{task_name}_{m}"
            if pc in flagged_df.columns:
                out.append(pc)
        out += [
            f"orig_{task_name}",
            f"claude_{task_name}",
            f"gpt_{task_name}",
            f"final_{task_name}",
        ]
        return out

    for t in tasks:
        col_order.extend(grouped_task_cols(t))
    col_order += ["total_misclassifications", "split"]

    subcols = [c for c in col_order if c in flagged_df.columns]
    flagged_tsv = os.path.join(output_dir, "flagged_relabelling_intermediate.tsv")
    os.makedirs(output_dir, exist_ok=True)
    flagged_df[subcols].to_csv(flagged_tsv, sep="\t", index=False)
    print(f"[INFO] Wrote {len(flagged_df)} rows to {flagged_tsv}.")

    # 5) Save train/val/test
    keep_cols = [
        "image_id",
        "event_name",
        "image_path",
        "damage_severity",
        "informative",
        "humanitarian",
        "disaster_types",
        "split",
    ]
    for t in tasks:
        for m in models:
            pred_col = f"pred_{t}_{m}"
            if pred_col in df.columns:
                keep_cols.append(pred_col)
    final_df = df[keep_cols].copy()
    for subset in ["train", "val", "test"]:
        subdf = final_df[final_df["split"] == subset].copy()
        out_file = os.path.join(output_dir, f"MEDIC_{subset}.tsv")
        subdf[keep_cols].to_csv(out_file, sep="\t", index=False)
        print(f"[INFO] Wrote {len(subdf)} rows to {out_file}.")

    # 6) llm_failures
    if failures:
        fail_tsv = os.path.join(output_dir, "llm_failures.tsv")
        pd.DataFrame(
            failures, columns=["model", "image_path", "reason", "raw_content"]
        ).to_csv(fail_tsv, sep="\t", index=False)
    print(f"[INFO] Claude: {claude_fail_count} images failed")
    print(f"[INFO] GPT: {gpt_fail_count} images failed")

    # Call stats function at the end
    relabelling_stats(input_tsv, output_dir)


def relabelling_stats(input_tsv, output_dir):
    """
    Generates the "Largest Alt-Label Agreement (1,2,3,All) + Relabelled" table by:
    1. Getting CNN disagreement stats from the input file
    2. Comparing original vs relabeled files for the "Relabelled" column
    """
    from config import PROJECT_ROOT

    # Convert input_tsv to absolute path if needed
    if not os.path.isabs(input_tsv):
        input_tsv = os.path.join(PROJECT_ROOT, "WSLcode", input_tsv)

    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, "WSLcode", output_dir)

    tasks = ["damage_severity", "informative", "humanitarian", "disaster_types"]
    models = ["resnet50", "efficientnet_b1", "mobilenet_v2"]

    flagged_path = os.path.join(output_dir, "flagged_relabelling_intermediate.tsv")
    print(f"\nDEBUG: Loading flagged data from {flagged_path}")
    flagged_df = pd.read_csv(flagged_path, sep="\t")
    print(f"DEBUG: Found {len(flagged_df)} flagged images")

    # Count actual changes from flagged data
    changes = 0
    for t in tasks:
        orig_col = f"orig_{t}"
        final_col = f"final_{t}"
        if orig_col in flagged_df.columns and final_col in flagged_df.columns:
            task_changes = (flagged_df[orig_col] != flagged_df[final_col]).sum()
            print(f"DEBUG: Task {t} had {task_changes} changes")
            changes += task_changes
    print(f"DEBUG: Total actual changes across all tasks: {changes}\n")

    print("DEBUG: Loading and comparing original vs relabeled data...")
    df_orig = pd.read_csv(input_tsv, sep="\t")

    # Load all relabeled data
    df_new = pd.DataFrame()
    for subset in ["train", "val", "test"]:
        path = os.path.join(output_dir, f"MEDIC_{subset}.tsv")
        if os.path.exists(path):
            temp = pd.read_csv(path, sep="\t")
            df_new = pd.concat([df_new, temp], ignore_index=True)

    # Create map from image_path to new labels
    new_labels = {}
    for _, row in df_new.iterrows():
        new_labels[row["image_path"]] = {
            "damage_severity": row["damage_severity"],
            "informative": row["informative"],
            "humanitarian": row["humanitarian"],
            "disaster_types": row["disaster_types"],
        }

    # Add flag for what changed
    for t in tasks:
        df_orig[f"changed_{t}"] = df_orig.apply(
            lambda r: (
                new_labels[r["image_path"]][t] != r[t]
                if r["image_path"] in new_labels
                else False
            ),
            axis=1,
        )

    _render_table2_largest_alt_with_relabelled(df_orig, tasks, models)


def _render_table2_largest_alt_with_relabelled(df, tasks, model_names):
    """
    1) For each row & task, get predictions = pred_{t}_{model}.
       If no misclassifications, skip. Else find 'largest_alt' = number of models that pick the same alt label ( != original ).
    2) 'Relabelled' = True if changed_{t} is True
    3) We pivot & format exactly like the original Table2.
    """
    import pandas as pd
    from collections import Counter

    def get_preds(row, t):
        preds = []
        for m in model_names:
            col = f"pred_{t}_{m}"
            if col in row:
                preds.append(row[col])
            else:
                preds.append(None)
        return preds

    records = []
    for _, row in df.iterrows():
        for t in tasks:
            gt_label = row[t]
            preds = get_preds(row, t)

            alt_list = [p for p in preds if (p is not None and p != gt_label)]
            if not alt_list:
                continue

            cnt = Counter(alt_list)
            largest = min(cnt.most_common(1)[0][1], 3)
            relab = int(row[f"changed_{t}"])
            records.append((t, gt_label, largest, relab))

    if not records:
        print("No misclassifications => no Table2 to show.")
        return

    alt_df = pd.DataFrame(records, columns=["task", "class", "largest_alt", "relab"])
    tab2 = alt_df.pivot_table(
        index=["task", "class"], columns="largest_alt", aggfunc="size", fill_value=0
    )
    for c in [1, 2, 3]:
        if c not in tab2.columns:
            tab2[c] = 0
    tab2 = tab2[[1, 2, 3]]
    tab2["All"] = tab2.sum(axis=1)

    tab2_rel = alt_df.pivot_table(
        index=["task", "class"],
        columns="largest_alt",
        values="relab",
        aggfunc="sum",
        fill_value=0,
    )
    for c in [1, 2, 3]:
        if c not in tab2_rel.columns:
            tab2_rel[c] = 0
    tab2_rel = tab2_rel[[1, 2, 3]]
    tab2_rel["All"] = tab2_rel.sum(axis=1)

    table2_rows = []
    overall = tab2.sum(numeric_only=True)
    overall_rel = tab2_rel.sum(numeric_only=True)

    unique_tasks = tab2.index.levels[0]
    for tk in unique_tasks:
        if tk not in tab2.index:
            continue
        subdf = tab2.xs(tk, level="task", drop_level=False)
        subrel = tab2_rel.xs(tk, level="task", drop_level=False)

        for (task_val, cls_val), rowvals in subdf.iterrows():
            total_ = int(rowvals["All"])
            rowcols = []
            for c in [1, 2, 3]:
                v = int(rowvals[c])
                pct = int(round(100 * v / total_)) if total_ > 0 else 0
                rowcols.append(f"{v:,} ({pct}%)")
            rowcols.append(f"{total_:,}")
            rel_ = int(subrel.loc[(task_val, cls_val), "All"])
            rel_pct = int(round(100 * rel_ / total_)) if total_ > 0 else 0
            rowcols.append(f"{rel_:,} ({rel_pct}%)")
            table2_rows.append((tk, f"  {cls_val}", rowcols))

        t_sum = subdf.sum()
        tot_t = int(t_sum["All"])
        tcols = []
        for c in [1, 2, 3]:
            vv = int(t_sum[c])
            pct = int(round(100 * vv / tot_t)) if tot_t > 0 else 0
            tcols.append(f"{vv:,} ({pct}%)")
        tcols.append(f"{tot_t:,}")
        rel_sum_t = int(subrel.sum()["All"])
        rel_sum_pct = int(round(100 * rel_sum_t / tot_t)) if tot_t > 0 else 0
        tcols.append(f"{rel_sum_t:,} ({rel_sum_pct}%)")
        table2_rows.append((tk, f"<strong>{tk}</strong>", tcols))

    all_total = int(overall["All"])
    rowfinal = []
    for c in [1, 2, 3]:
        val = int(overall[c])
        pct = int(round(100 * val / all_total)) if all_total > 0 else 0
        rowfinal.append(f"{val:,} ({pct}%)")
    rowfinal.append(f"{all_total:,}")
    rel_total = int(overall_rel["All"])
    rel_pct_total = int(round(100 * rel_total / all_total)) if all_total > 0 else 0
    rowfinal.append(f"{rel_total:,} ({rel_pct_total}%)")
    table2_rows.append(("ALL TASKS", "<strong>All Tasks</strong>", rowfinal))

    final_data = []
    for taskval, labelval, rowcols in table2_rows:
        final_data.append([labelval] + rowcols)

    df_tab2 = pd.DataFrame(
        final_data, columns=["Label", "1", "2", "3", "All", "Relabelled"]
    )

    def align_label(val):
        return "text-align:center;" if "<strong>" in str(val) else "text-align:left;"

    sty = df_tab2.style.set_caption(
        "TABLE 2: Largest Alt-Label Agreement (1,2,3,All) + Relabelled"
    )
    sty = sty.map(align_label, subset="Label")
    sty = sty.set_properties(
        subset=df_tab2.columns[1:], **{"text-align": "right", "padding": "4px"}
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
    display(sty)
