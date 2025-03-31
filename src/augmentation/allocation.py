"""
allocation.py

This module computes how many synthetic images should be generated for each multi-task
label combination when augmenting the MEDIC dataset with diffusion-based methods.
It uses model-based performance metrics (F1 scores) to prioritise underperforming labels
and applies additional correlation- and impact-based bonuses.

Class and function responsibilities:
- WeaknessThresholds & AllocationConfig: Store configuration parameters for thresholds and bonuses.
- SyntheticAllocationCalculator: Orchestrates the calculation of raw and adjusted (reallocated) synthetic
  image allocations for every possible label combination across four tasks:
    [damage_severity, informative, humanitarian, disaster_types].
- display_allocation_summary: Visualises the allocation breakdown in a tabular format.
"""

import os
from collections import defaultdict
import numpy as np
import pandas as pd
import itertools
from dataclasses import dataclass
from typing import Dict, Tuple, List
import logging
from IPython.display import display, HTML

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importing modules related to data loading and metrics
from src.analysis.class_imbalance import (
    calculate_metrics,
    load_model,
    test_model,
    DATA_DIR,
    BATCH_SIZE,
    NUM_THREADS,
)
from config import LABEL_MAPPINGS, TASKS, get_model_paths, DATA_VERSIONS


@dataclass
class WeaknessThresholds:
    """
    Defines numeric thresholds for categorising a label as 'critical', 'moderate',
    'below_average', or 'normal'. These thresholds are interpreted as F1-score
    cutoffs in percentage form.

    Attributes:
        CRITICAL (float): Upper bound for 'critical' (lowest-performing) labels.
        MODERATE (float): Upper bound for 'moderate' labels.
        BELOW_AVERAGE (float): Upper bound for 'below_average' labels.
    """

    CRITICAL: float = 40.0
    MODERATE: float = 60.0
    BELOW_AVERAGE: float = 75.0


@dataclass
class AllocationConfig:
    """
    Houses numeric bonus values and base allocation counts used during synthetic
    image allocation. Each label combination begins with a base number of images,
    and additional bonuses are applied depending on label weaknesses, correlations,
    and humanitarian/impact factors.

    Attributes:
        BASE_IMAGES (int): Baseline number of synthetic images allocated to all combinations.
        CRITICAL_BONUS (int): Additional images if the min F1 for a label is below the 'critical' threshold.
        MODERATE_BONUS (int): Additional images if the min F1 for a label is below the 'moderate' threshold.
        BELOW_AVG_BONUS (int): Additional images if the min F1 for a label is below the 'below_average' threshold.
        MILD_DISASTER_BONUS (int): Bonus for 'mild' damage combined with certain disaster types.
        OTHER_INFORMATIVE_BONUS (int): Bonus for the 'other_disaster' + 'informative' pairing.
        HUMANITARIAN_DAMAGE_BONUS (int): Bonus for certain humanitarian labels combined with 'mild' or 'severe' damage.
        LIFE_CRITICAL_BONUS (int): Extra for labels that involve 'affected_injured_or_dead_people'.
        RESPONSE_CRITICAL_BONUS (int): Extra for 'severe' damage + major disasters like hurricanes/floods.
        RESCUE_CRITICAL_BONUS (int): Extra for 'rescue_volunteering_or_donation_effort'.
    """

    # Base allocation for all combinations
    BASE_IMAGES: int = 10

    # Performance-based bonuses
    CRITICAL_BONUS: int = 80
    MODERATE_BONUS: int = 40
    BELOW_AVG_BONUS: int = 20

    # Correlation-based bonuses
    MILD_DISASTER_BONUS: int = 32
    OTHER_INFORMATIVE_BONUS: int = 32
    HUMANITARIAN_DAMAGE_BONUS: int = 20

    # Impact-based bonuses
    LIFE_CRITICAL_BONUS: int = 28
    RESPONSE_CRITICAL_BONUS: int = 24
    RESCUE_CRITICAL_BONUS: int = 20


def get_min_f1_scores() -> Dict[Tuple[str, str], float]:
    """
    Computes the minimum F1 score for each (task, class_name) pair across all
    currently saved model checkpoints. It first loads each model, computes per-class F1,
    and then aggregates the lowest F1 value for each class.

    Returns:
        dict: A dictionary keyed by (task, class_name) to the min F1 score (float).
    """
    model_paths = get_model_paths()
    f1_scores = defaultdict(list)

    # Loop over each registered model checkpoint and compute per-class F1
    for model_name, model_path in model_paths.items():
        model = load_model(model_name, model_path)
        true_labels, predictions, _ = test_model(
            model, DATA_DIR, batch_size=BATCH_SIZE, num_threads=NUM_THREADS
        )
        metrics_dict = calculate_metrics(true_labels, predictions, LABEL_MAPPINGS)

        # Extract class-level F1 for each task; accumulate in a dictionary for min() calculation
        for task in TASKS:
            for class_name, class_metrics in metrics_dict[task]["classes"].items():
                f1_scores[(task, class_name)].append(class_metrics["f1_score"])

    # For each (task, label) pair, identify the minimum F1 across all models
    min_scores = {}
    for (task, class_name), scores in f1_scores.items():
        min_score = min(scores)
        min_scores[(task, class_name)] = min_score

    return min_scores


class SyntheticAllocationCalculator:
    """
    Calculates and redistributes synthetic image counts for each combination of the four tasks:
    (damage_severity, informative, humanitarian, disaster_types). The process:

    1. Determines how weak each label is (critical, moderate, below_average, or normal)
       based on the min F1 score across multiple models.
    2. Computes a 'raw' allocation for each 4-label combo by adding base + weakness + correlation + impact bonuses.
    3. Separates combos that actually appear in the training data from those that do not.
       Sums the allocations for 'invalid' combos and redistributes that sum to valid combos.
    4. Scales the total to a target (e.g., ~10,000 synthetic images).
    5. Returns a DataFrame with final synthetic image counts.

    Usage:
        - Instantiate with optional custom thresholds and config.
        - Call generate_all_allocations() to produce the final CSV and summary table.
    """

    def __init__(
        self,
        thresholds: WeaknessThresholds = WeaknessThresholds(),
        config: AllocationConfig = AllocationConfig(),
    ):
        """
        Initialise the allocation calculator with threshold cutoffs and bonus configurations.

        Args:
            thresholds (WeaknessThresholds): Defines numeric cutoffs for critical, moderate, below-average label performance.
            config (AllocationConfig): Specifies base images, performance-based bonuses, correlation-based bonuses, etc.
        """
        self.thresholds = thresholds
        self.config = config

        # Retrieve min F1 scores from the loaded models
        self.f1_scores = get_min_f1_scores()

        # For each (task, class_name) pair, store the weakness level
        self.label_weakness = {}
        self._initialize_weakness_levels()

    def _initialize_weakness_levels(self) -> None:
        """
        Categorises each (task, class_name) into 'critical', 'moderate', 'below_average', or 'normal'
        by comparing its min F1 score to the threshold values in self.thresholds.
        """
        for (task, class_name), f1 in self.f1_scores.items():
            if f1 < self.thresholds.CRITICAL:
                w = "critical"
            elif f1 < self.thresholds.MODERATE:
                w = "moderate"
            elif f1 < self.thresholds.BELOW_AVERAGE:
                w = "below_average"
            else:
                w = "normal"
            self.label_weakness[(task, class_name)] = w

    def _calculate_weakness_bonuses(self, combo: Tuple[str, str, str, str]) -> int:
        """
        Computes the total bonus for performance-based weaknesses. This checks
        each label in the 4-label combo to see if it is 'critical', 'moderate', or 'below_average'.

        Args:
            combo (tuple): A 4-element tuple (damage_severity, informative, humanitarian, disaster_type).

        Returns:
            int: Sum of performance-based bonuses for the 4-label combo.
        """
        total_bonus = 0
        for task, label in zip(TASKS, combo):
            weakness = self.label_weakness.get((task, label), "normal")
            if weakness == "critical":
                total_bonus += self.config.CRITICAL_BONUS
            elif weakness == "moderate":
                total_bonus += self.config.MODERATE_BONUS
            elif weakness == "below_average":
                total_bonus += self.config.BELOW_AVG_BONUS
        return total_bonus

    def _calculate_correlation_bonuses(self, combo: Tuple[str, str, str, str]) -> int:
        """
        Identifies specific label correlations that are known to be misclassified or especially important.
        Adds additional synthetic images for these correlations.

        Args:
            combo (tuple): (damage_severity, informative, humanitarian, disaster_type).

        Returns:
            int: Total correlation-based bonus for the given 4-label combination.
        """
        damage, informative, humanitarian, disaster = combo
        total_bonus = 0

        # 'mild' combined with certain disasters
        if damage == "mild" and disaster in [
            "hurricane",
            "flood",
            "other_disaster",
            "landslide",
        ]:
            total_bonus += self.config.MILD_DISASTER_BONUS

        # 'other_disaster' + 'informative'
        if disaster == "other_disaster" and informative == "informative":
            total_bonus += self.config.OTHER_INFORMATIVE_BONUS

        # synergy between certain humanitarian labels and damage levels
        if humanitarian in [
            "infrastructure_and_utility_damage",
            "rescue_volunteering_or_donation_effort",
        ] and damage in ["mild", "severe"]:
            total_bonus += self.config.HUMANITARIAN_DAMAGE_BONUS

        return total_bonus

    def _calculate_impact_bonuses(self, combo: Tuple[str, str, str, str]) -> int:
        """
        Allocates additional images for labels with high humanitarian impact (e.g., 'affected_injured_or_dead_people')
        or for severe disaster scenarios that are highly critical for real-world outcomes.

        Args:
            combo (tuple): (damage_severity, informative, humanitarian, disaster_type).

        Returns:
            int: Sum of impact-based bonuses added to this combination.
        """
        damage, informative, humanitarian, disaster = combo
        total_bonus = 0

        # Life-critical scenario if the humanitarian label indicates human casualties or injuries
        if humanitarian == "affected_injured_or_dead_people":
            total_bonus += self.config.LIFE_CRITICAL_BONUS

        # Major disaster synergy if damage is severe and disaster is among typical high-impact events
        if disaster in ["earthquake", "hurricane", "flood"] and damage == "severe":
            total_bonus += self.config.RESPONSE_CRITICAL_BONUS

        # Rescue operations synergy
        if humanitarian == "rescue_volunteering_or_donation_effort":
            total_bonus += self.config.RESCUE_CRITICAL_BONUS

        return total_bonus

    def calculate_allocation(self, combo: Tuple[str, str, str, str]) -> int:
        """
        Computes the total initial (raw) synthetic image allocation for a single 4-label combination,
        by summing up base allocation, weakness-based bonuses, correlation bonuses, and impact bonuses.

        Args:
            combo (tuple): (damage_severity, informative, humanitarian, disaster_type).

        Returns:
            int: The unadjusted allocation for this combination.
        """
        base = self.config.BASE_IMAGES
        w = self._calculate_weakness_bonuses(combo)
        c = self._calculate_correlation_bonuses(combo)
        i = self._calculate_impact_bonuses(combo)
        return base + w + c + i

    def generate_all_allocations(self) -> pd.DataFrame:
        """
        Orchestrates the full synthetic allocation process:

        1. Loads the relabelled training data to identify which (damage, informative, humanitarian, disaster) combos are valid.
        2. Constructs a list of all possible combos from LABEL_MAPPINGS.
        3. Calculates a raw allocation for each combo.
        4. Splits combos into valid vs. invalid, summing allocations for each subset.
        5. Redistributes the allocation from invalid combos to valid combos proportionally.
        6. Scales the grand total to ~10,000, preserving relative ratios.
        7. Saves the final allocations as 'synthetic_allocations.csv' and displays a summary.

        Returns:
            pd.DataFrame: A sorted DataFrame containing the final synthetic image allocations
                          and additional information (weakness levels, F1 scores, etc.).
        """
        # (1) Identify valid combinations from the training dataset
        train_path = DATA_VERSIONS["relabelled"]["train"]
        train_df = pd.read_csv(train_path, sep="\t")[TASKS]

        valid_combo_set = set(tuple(row) for _, row in train_df.iterrows())

        # (2) Build all possible combinations in the order: damage, informative, humanitarian, disaster
        task_values = {
            task: list(mapping.values()) for task, mapping in LABEL_MAPPINGS.items()
        }
        all_combos = list(
            itertools.product(
                task_values["damage_severity"],
                task_values["informative"],
                task_values["humanitarian"],
                task_values["disaster_types"],
            )
        )

        valid_rows = []
        invalid_rows = []
        sum_valid = 0
        sum_invalid = 0

        # (3) Calculate raw allocations, separate valid vs invalid combos
        for combo in all_combos:
            raw_alloc = self.calculate_allocation(combo)
            row_dict = {
                "DamageSeverity": combo[0],
                "Informativeness": combo[1],
                "Humanitarian": combo[2],
                "DisasterType": combo[3],
                "RawAllocation": raw_alloc,
            }

            if combo in valid_combo_set:
                valid_rows.append(row_dict)
                sum_valid += raw_alloc
            else:
                invalid_rows.append(row_dict)
                sum_invalid += raw_alloc

        logger.info(
            f"[Initial allocation] valid combos sum={sum_valid}, "
            f"invalid combos sum={sum_invalid}, total={sum_valid + sum_invalid}"
        )

        # (4) Redistribute invalid allocations among valid combos
        if sum_valid == 0:
            # If no valid combos exist, return an empty DataFrame
            logger.warning("No valid combos found in the training set!")
            final_df = pd.DataFrame()
        else:
            final_rows = []
            for row_dict in valid_rows:
                old_alloc = row_dict["RawAllocation"]
                fraction = old_alloc / float(sum_valid)
                reallocated = fraction * sum_invalid
                new_alloc = old_alloc + reallocated
                row_dict["ReallocatedAlloc"] = new_alloc
                final_rows.append(row_dict)

            # (5) Scale from the original total to the target of 10,000 synthetic images
            original_total = sum_valid + sum_invalid
            desired_total = 10000.0
            ratio = desired_total / original_total if original_total > 0 else 1.0

            logger.info(
                f"Scaling from {original_total:.1f} -> {desired_total:.1f} (ratio={ratio:.4f})"
            )

            # Construct DataFrame after scaling
            df_list = []
            for row_dict in final_rows:
                scaled = round(row_dict["ReallocatedAlloc"] * ratio)
                df_list.append(
                    {
                        "DamageSeverity": row_dict["DamageSeverity"],
                        "Informativeness": row_dict["Informativeness"],
                        "Humanitarian": row_dict["Humanitarian"],
                        "DisasterType": row_dict["DisasterType"],
                        "SyntheticImages": scaled,
                        "RawAllocation": row_dict["RawAllocation"],
                        "ReallocatedAlloc": row_dict["ReallocatedAlloc"],
                    }
                )
            final_df = pd.DataFrame(df_list)

        # (6) Attach weakness and F1 details for reference
        if not final_df.empty:
            weaknesses_list = []
            f1_list = []
            for _, row in final_df.iterrows():
                combo = (
                    row["DamageSeverity"],
                    row["Informativeness"],
                    row["Humanitarian"],
                    row["DisasterType"],
                )
                w_levels = []
                f1_vals = []
                for task, lbl in zip(TASKS, combo):
                    w_levels.append(self.label_weakness.get((task, lbl), "normal"))
                    f1_vals.append(self.f1_scores.get((task, lbl), 0.0))

                weaknesses_list.append("|".join(w_levels))
                f1_list.append("|".join(f"{v:.1f}%" for v in f1_vals))

            final_df["WeaknessLevels"] = weaknesses_list
            final_df["F1Scores"] = f1_list

            total_final = final_df["SyntheticImages"].sum()
            logger.info(
                f"[After reallocation + scaling] final synthetic sum={total_final}"
            )

        # (7) Sort and save final table
        df_sorted = final_df.sort_values(
            by=["SyntheticImages", "RawAllocation"], ascending=False
        ).reset_index(drop=True)

        # Changed to a relative path
        output_path = os.path.join(
            os.path.dirname(__file__), "synthetic_allocations.csv"
        )
        # df_sorted.to_csv(output_path, index=False)

        # Display summary of allocations
        display_allocation_summary(df_sorted, self.f1_scores)
        return df_sorted


def display_allocation_summary(df: pd.DataFrame, f1_scores: Dict) -> None:
    """
    Generates a formatted HTML summary showing how many synthetic images
    are allocated to each label in each task, along with F1 scores and real
    training set counts. Also displays a row-level total for each task.

    Args:
        df (pd.DataFrame): DataFrame of final allocations, containing columns
                           ['DamageSeverity', 'Informativeness', 'Humanitarian',
                            'DisasterType', 'SyntheticImages', ...].
        f1_scores (dict): A dictionary mapping (task, class_name) to the min F1
                          score across models for that label.
    """
    if df.empty:
        display(
            HTML(
                "<h3>No valid combos found - no synthetic allocations to display.</h3>"
            )
        )
        return

    # Load the relabelled training set for reference
    train_path = DATA_VERSIONS["relabelled"]["train"]
    train_df = pd.read_csv(train_path, sep="\t")[TASKS]

    # Calculate how many real samples each task/class has
    support_counts = {task: train_df[task].value_counts() for task in TASKS}

    # Tally synthetic allocations per task/class
    allocations = {task: defaultdict(int) for task in TASKS}
    combos_map = {task: defaultdict(set) for task in TASKS}
    total_synth = df["SyntheticImages"].sum()

    for _, row in df.iterrows():
        dmg = row["DamageSeverity"]
        inf = row["Informativeness"]
        hum = row["Humanitarian"]
        dis = row["DisasterType"]
        alloc = row["SyntheticImages"]

        allocations["damage_severity"][dmg] += alloc
        allocations["informative"][inf] += alloc
        allocations["humanitarian"][hum] += alloc
        allocations["disaster_types"][dis] += alloc

        combo_tuple = (dmg, inf, hum, dis)
        combos_map["damage_severity"][dmg].add(combo_tuple)
        combos_map["informative"][inf].add(combo_tuple)
        combos_map["humanitarian"][hum].add(combo_tuple)
        combos_map["disaster_types"][dis].add(combo_tuple)

    # Build a summary table containing the final allocation details
    summary_rows = []
    for task in TASKS:
        task_display = task.replace("_", " ").title()
        for class_name in LABEL_MAPPINGS[task].values():
            real_count = int(support_counts[task].get(class_name, 0))
            synth_count = allocations[task][class_name]
            ratio = (synth_count / real_count * 100) if real_count > 0 else 0.0

            summary_rows.append(
                {
                    "Task": task_display,
                    "Class": class_name,
                    "F1 Score (%)": f"{f1_scores.get((task, class_name), 0.0):.1f}%",
                    "Train Count": f"{real_count:,d}",
                    "Synthetic": f"{synth_count:,d}",
                    "Boost Ratio (%)": f"{ratio:.1f}%",
                    "Label Combos": f"{len(combos_map[task][class_name])}",
                    "Share of Total (%)": f"{(synth_count / total_synth)*100:.1f}%",
                }
            )

        # Provide a total row for each task
        total_real = support_counts[task].sum()
        total_synth_task = sum(allocations[task].values())
        ratio_task = (total_synth_task / total_real * 100) if total_real > 0 else 0.0
        summary_rows.append(
            {
                "Task": task_display,
                "Class": "Total",
                "F1 Score (%)": "",
                "Train Count": f"{total_real:,d}",
                "Synthetic": f"{total_synth_task:,d}",
                "Boost Ratio (%)": f"{ratio_task:.1f}%",
                "Label Combos": "All",
                "Share of Total (%)": "100.0%",
            }
        )

    df_summary = pd.DataFrame(summary_rows)

    # Sort to place the total row last for each task
    df_summary["SortKey"] = df_summary["Class"].map(
        lambda x: "z" if x == "Total" else "a"
    )
    df_summary = df_summary.sort_values(["Task", "SortKey"]).drop("SortKey", axis=1)

    # Generate a styled HTML table for readability
    styled = df_summary.style.set_properties(
        subset=["Task", "Class"], **{"text-align": "left"}
    ).set_properties(
        subset=[
            "F1 Score (%)",
            "Train Count",
            "Synthetic",
            "Boost Ratio (%)",
            "Label Combos",
            "Share of Total (%)",
        ],
        **{"text-align": "center"},
    )

    display(HTML("<h3>Synthetic Image Allocation Summary</h3>"))
    display(styled)
    display(HTML(f"<b>Total Synthetic Images: {int(total_synth):,d}</b>"))
