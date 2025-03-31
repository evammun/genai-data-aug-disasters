"""
test_subset_generator.py

Generate balanced test subsets from the MEDIC dataset for zero-shot classification experiments.
"""

import os
import sys
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the parent directory to the path to import config.py
sys.path.append(str(Path(__file__).parents[3]))  # WSLcode directory
import config


def load_medic_dev_data():
    """
    Load the MEDIC development dataset using paths from config.py

    Returns:
        pd.DataFrame: The loaded dataset
    """
    # Get the data paths from config
    data_paths = config.get_data_paths()

    # Load the development set
    dev_path = data_paths["val"]

    # Read the TSV file
    df = pd.read_csv(dev_path, sep="\t")

    return df


def create_small_subset(df, target_size=30, seed=42):
    """
    Create a small balanced subset (40-50 images) ensuring coverage of key categories
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    selected_indices = []

    # Get all possible values for each task
    disaster_types = df["disaster_types"].unique()
    damage_severity = df["damage_severity"].unique()
    informative = df["informative"].unique()
    humanitarian = df["humanitarian"].unique()

    # 1. First, ensure at least 1 image from each disaster type
    for disaster in disaster_types:
        indices = df[df["disaster_types"] == disaster].index.tolist()
        if indices:
            selected_index = np.random.choice(indices)
            selected_indices.append(selected_index)

    # 2. Ensure at least 1 image from each damage severity level
    for severity in damage_severity:
        # Skip if we already have this combination
        if any(df.loc[idx, "damage_severity"] == severity for idx in selected_indices):
            continue

        indices = df[df["damage_severity"] == severity].index.tolist()
        if indices:
            selected_index = np.random.choice(indices)
            selected_indices.append(selected_index)

    # 3. Ensure at least 1 image from each informative status
    for info in informative:
        # Skip if we already have this combination
        if any(df.loc[idx, "informative"] == info for idx in selected_indices):
            continue

        indices = df[df["informative"] == info].index.tolist()
        if indices:
            selected_index = np.random.choice(indices)
            selected_indices.append(selected_index)

    # 4. Ensure at least 1 image from each humanitarian category
    for hum in humanitarian:
        # Skip if we already have this combination
        if any(df.loc[idx, "humanitarian"] == hum for idx in selected_indices):
            continue

        indices = df[df["humanitarian"] == hum].index.tolist()
        if indices:
            selected_index = np.random.choice(indices)
            selected_indices.append(selected_index)

    # 5. Fill remaining slots with diverse examples
    remaining_count = target_size - len(selected_indices)
    if remaining_count > 0:
        # Get indices not already selected
        remaining_indices = [idx for idx in df.index if idx not in selected_indices]

        # Select the remaining indices
        if remaining_indices:
            num_to_select = min(remaining_count, len(remaining_indices))
            additional_indices = np.random.choice(
                remaining_indices, num_to_select, replace=False
            )
            selected_indices.extend(additional_indices)

    return df.loc[selected_indices].copy()


def create_medium_subset(df, target_size=500, seed=42):
    """
    Create a medium-sized subset (150-500 images) with comprehensive coverage
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Start with a small subset to ensure basic coverage
    small_subset = create_small_subset(
        df, target_size=min(50, target_size // 4), seed=seed
    )
    selected_indices = small_subset.index.tolist()

    # Get all unique combinations of labels in the dataset
    df_copy = df.copy()
    df_copy["label_combo"] = df_copy.apply(
        lambda row: f"{row['disaster_types']}_{row['damage_severity']}_{row['informative']}_{row['humanitarian']}",
        axis=1,
    )

    # Count occurrences of each label combination
    combo_counts = df_copy["label_combo"].value_counts()

    # First, add rare combinations (occurring 3 times or less)
    rare_combos = combo_counts[combo_counts <= 3].index.tolist()
    for combo in rare_combos:
        # Get indices for this combination that aren't already selected
        combo_indices = df_copy[df_copy["label_combo"] == combo].index.tolist()
        combo_indices = [idx for idx in combo_indices if idx not in selected_indices]

        if combo_indices:
            # Add one example of each rare combination
            selected_index = np.random.choice(combo_indices)
            selected_indices.append(selected_index)

    # Calculate how many more samples we need
    remaining_count = target_size - len(selected_indices)

    # Add more examples of common combinations
    if remaining_count > 0:
        # Get indices not already selected
        remaining_indices = [idx for idx in df.index if idx not in selected_indices]

        if remaining_indices:
            # Select the remaining samples randomly
            num_to_select = min(remaining_count, len(remaining_indices))
            additional_indices = np.random.choice(
                remaining_indices, num_to_select, replace=False
            )
            selected_indices.extend(additional_indices)

    # Return the final subset
    return df.loc[selected_indices].copy()


def save_subset(df, output_path):
    """
    Save the selected subset as a CSV file, keeping only essential columns
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a copy to avoid modifying the original
    output_df = df.copy().reset_index(drop=True)

    # Keep only the required columns
    required_columns = [
        "image_id",
        "event_name",
        "image_path",
        "damage_severity",
        "informative",
        "humanitarian",
        "disaster_types",
    ]

    # Filter columns, keeping only those that exist in the DataFrame
    existing_columns = [col for col in required_columns if col in output_df.columns]
    output_df = output_df[existing_columns]

    # Save to CSV
    output_df.to_csv(output_path, index=False)


def visualize_subsets(subsets):
    """
    Create visualizations for the generated subsets
    """
    # Set up the seaborn style using config settings
    sns.set(
        style=config.SNS_STYLE,
        context=config.SNS_CONTEXT,
        font_scale=config.SNS_FONT_SCALE,
        palette=config.SNS_PALETTE,
    )

    # Get the tasks from config
    tasks = config.TASKS

    # Plot the distribution of labels in each subset
    for subset_name, subset_df in subsets.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))  # Smaller size
        fig.suptitle(
            f"Label Distribution in {subset_name.capitalize()} Subset ({len(subset_df)} images)",
            fontsize=16,
            y=1.02,
        )

        for i, task in enumerate(tasks):
            ax = axes[i // 2, i % 2]

            # Get sorted categories
            categories = sorted(subset_df[task].unique())

            # Create countplot with explicitly set order
            sns.countplot(data=subset_df, x=task, ax=ax, order=categories)

            # Add counts above bars
            for p in ax.patches:
                ax.annotate(
                    f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            # Format the plot - properly set ticks first, then labels
            ax.set_title(f'{task.replace("_", " ").title()}', fontsize=14)

            # Fix the UserWarning by setting ticks first then labels
            ticks = ax.get_xticks()
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels=categories, rotation=45, ha="right")

            ax.set_xlabel("")
            ax.set_ylabel("Count")

        plt.tight_layout()
        plt.show()

    # Calculate overlap between small and medium subsets
    small_image_paths = set(subsets["small"]["image_path"])
    medium_image_paths = set(subsets["medium"]["image_path"])
    overlap = small_image_paths.intersection(medium_image_paths)

    # Create a Venn-style diagram showing overlap
    try:
        plt.figure(figsize=(8, 5))  # Smaller size
        from matplotlib_venn import venn2

        v = venn2(
            subsets=(
                len(small_image_paths - overlap),
                len(medium_image_paths - overlap),
                len(overlap),
            ),
            set_labels=("Small Subset", "Medium Subset"),
        )

        # Custom styling
        for text in v.set_labels:
            text.set_fontsize(12)
        for text in v.subset_labels:
            if text is not None:
                text.set_fontsize(10)

        plt.title("Overlap Between Small and Medium Subsets", fontsize=14)
        plt.show()
    except ImportError:
        # Skip venn diagram if matplotlib-venn not available
        pass

    # Create a heatmap of task co-occurrences for medium subset
    plt.figure(figsize=(10, 6))  # Smaller size
    crosstab = pd.crosstab(
        subsets["medium"]["disaster_types"], subsets["medium"]["damage_severity"]
    )

    sns.heatmap(crosstab, annot=True, cmap="YlGnBu", fmt="d", linewidths=0.5)

    plt.title("Disaster Types vs. Damage Severity (Medium Subset)", fontsize=14)
    plt.xlabel("Damage Severity")
    plt.ylabel("Disaster Type")
    plt.tight_layout()
    plt.show()


def generate_test_subsets(output_dir, seed=42):
    """
    Generate and save small and medium test subsets

    Parameters:
        output_dir (str): Directory to save the output files
        seed (int): Random seed for reproducibility

    Returns:
        dict: Dictionary containing the generated subsets
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the MEDIC dev data
    df = load_medic_dev_data()
    print(f"Loaded {len(df)} images from MEDIC dev dataset")

    # Create the small subset (40-50 images)
    small_subset = create_small_subset(df, target_size=100, seed=seed)
    small_output_path = os.path.join(output_dir, "small_test_subset.csv")
    save_subset(small_subset, small_output_path)
    print(f"Saved {len(small_subset)} images to {small_output_path}")

    # Create the medium subset (150-200 images)
    medium_subset = create_medium_subset(df, target_size=500, seed=seed)
    medium_output_path = os.path.join(output_dir, "medium_test_subset.csv")
    save_subset(medium_subset, medium_output_path)
    print(f"Saved {len(medium_subset)} images to {medium_output_path}")

    # Create a dictionary with the subsets
    subsets = {"small": small_subset, "medium": medium_subset, "dev": df}

    # Create and display the formatted table
    create_distribution_table(subsets)

    return subsets


def create_distribution_table(subsets):
    """
    Create and display a formatted table of label distributions across datasets

    Parameters:
        subsets (dict): Dictionary containing 'dev', 'small', and 'medium' dataframes
    """
    from IPython.display import display, HTML
    import pandas as pd

    # Get the dataframes
    dev_df = subsets["dev"]
    small_df = subsets["small"]
    medium_df = subsets["medium"]

    # Define the tasks and their classes
    task_classes = {
        "damage_severity": ["little_or_none", "mild", "severe"],
        "informative": ["not_informative", "informative"],
        "humanitarian": [
            "affected_injured_or_dead_people",
            "infrastructure_and_utility_damage",
            "not_humanitarian",
            "rescue_volunteering_or_donation_effort",
        ],
        "disaster_types": [
            "earthquake",
            "fire",
            "flood",
            "hurricane",
            "landslide",
            "not_disaster",
            "other_disaster",
        ],
    }

    # Initialize lists to store table data
    rows = []
    row_types = []  # To track which rows are task headers

    # Process each task and its classes
    for task, classes in task_classes.items():
        # Calculate frequencies for the dev set
        dev_counts = dev_df[task].value_counts(normalize=True) * 100

        # Calculate frequencies for the small subset
        small_counts = small_df[task].value_counts()
        small_percentages = small_df[task].value_counts(normalize=True) * 100

        # Calculate frequencies for the medium subset
        medium_counts = medium_df[task].value_counts()
        medium_percentages = medium_df[task].value_counts(normalize=True) * 100

        # Add task row (will be formatted differently)
        formatted_task = task.replace("_", " ").title()
        rows.append([formatted_task, "", "", ""])
        row_types.append("task")

        # Add class rows
        for cls in classes:
            # Format the class name
            formatted_class = cls.replace("_", " ").title()

            # Calculate values - using .get() to handle missing values safely
            dev_pct = dev_counts.get(cls, 0)
            small_count = small_counts.get(cls, 0)
            small_pct = small_percentages.get(cls, 0)
            medium_count = medium_counts.get(cls, 0)
            medium_pct = medium_percentages.get(cls, 0)

            rows.append(
                [
                    f"&nbsp;&nbsp;{formatted_class}",
                    f"{dev_pct:.1f}%",
                    f"{small_count} ({small_pct:.1f}%)",
                    f"{medium_count} ({medium_pct:.1f}%)",
                ]
            )
            row_types.append("class")

    # Generate HTML with custom row formatting
    html_parts = ['<table border="1" class="dataframe">']

    # Add header row
    html_parts.append("<thead>")
    html_parts.append('<tr style="text-align: center; background-color: 040404;">')
    for col in (
        table_df.columns
        if "table_df" in locals()
        else ["Task/Class", "% in Dev Set", "Small Subset", "Medium Subset"]
    ):
        html_parts.append(f"<th>{col}</th>")
    html_parts.append("</tr>")
    html_parts.append("</thead>")

    # Add body rows with custom formatting
    html_parts.append("<tbody>")
    for i in range(len(rows)):
        row = rows[i]
        if row_types[i] == "task":
            # Task header row (dark background, white text, no background for cells)
            html_parts.append('<tr style="color: white;">')
            html_parts.append(
                f'<td style="text-align: left; background-color: #333333;"><b>{row[0]}</b></td>'
            )
            for j in range(1, len(row)):
                html_parts.append(
                    f'<td style="text-align: center; background-color: #333333;">{row[j]}</td>'
                )
            html_parts.append("</tr>")
        else:
            # Class row (transparent background)
            html_parts.append("<tr>")
            html_parts.append(f'<td style="text-align: left;">{row[0]}</td>')
            for j in range(1, len(row)):
                html_parts.append(f'<td style="text-align: center;">{row[j]}</td>')
            html_parts.append("</tr>")

    html_parts.append("</tbody>")
    html_parts.append("</table>")

    # Join all parts
    html = "".join(html_parts)

    # Display the table
    display(HTML(html))
