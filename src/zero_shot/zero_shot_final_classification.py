"""
zero_shot_final_classification.py

A utility module providing a function to run zero-shot inference using
GPT-4o on a TSV file containing test data. Unlike the general zero_shot_tester.py,
this script is specifically designed for the final test dataset in TSV format
and uses only GPT-4o as the vision LLM.

Usage:
    from src.zero_shot.zero_shot_final_classification import zero_shot_classification

    zero_shot_classification(
        test_tsv_filename="final_test.tsv",
        prompt_name="my_prompt",
        num_workers=4  # Use 4 parallel processes
    )
"""

import os
import json
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing
import time
import traceback
from typing import Union, List

from src.zero_shot.zero_shot_tester import (
    extract_analysis,
    extract_labels,
    is_valid_response,
    is_refusal,
    process_image,
    LABEL_MAPPINGS,
)


# Import the vision LLM factory and config
from src.zero_shot.testing.models import get_vision_llm
import config


def zero_shot_classification(
    test_tsv_filename: str,
    prompt_name: str,
    num_workers: int = 1,
    output_dir: str = "results",
    max_images: int = None,
    test_mode: bool = False,
):
    """
    Runs zero-shot inference on a TSV file of test images using GPT-4o only,
    with multiprocessing support.

    Args:
        test_tsv_filename (str):
            Path to the TSV file containing test data

        prompt_name (str):
            Identifier for the prompt to use from prompts.json

        num_workers (int, optional):
            Number of worker processes to use for parallel inference.
            Default is 1 (sequential processing).

        output_dir (str, optional):
            Directory to save results. Default is "results".

        max_images (int, optional):
            If provided, only process this many images from the dataset.
            Default is None (process all images).

        test_mode (bool, optional):
            If True, only process the first 5 images as a quick test.
            Default is False.

    Returns:
        pandas.DataFrame: The results DataFrame with predictions added
    """
    start_time = time.time()

    # Fixed model name for GPT-4o
    model_name = "gpt4v"

    # -------------------------------------------------------------------
    # 1) Read the test TSV using pandas
    # -------------------------------------------------------------------
    df = pd.read_csv(test_tsv_filename, sep="\t")
    if df.empty:
        print(f"Warning: The TSV '{test_tsv_filename}' is empty or invalid.")
        return

    # Limit the number of images if test_mode or max_images is specified
    if test_mode:
        df = df.head(5)
        print(f"TEST MODE: Limited to first 5 images")
    elif max_images is not None:
        df = df.head(max_images)
        print(f"Limited to first {max_images} images")

    print(f"Loaded {len(df)} rows from {test_tsv_filename}")

    # -------------------------------------------------------------------
    # 2) Load prompt text from prompts.json
    # -------------------------------------------------------------------
    prompts_json_path = os.path.join(os.path.dirname(__file__), "prompts.json")
    if not os.path.exists(prompts_json_path):
        raise FileNotFoundError(f"Could not find prompts.json at: {prompts_json_path}")

    with open(prompts_json_path, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)

    if prompt_name not in all_prompts:
        raise ValueError(f"No prompt named '{prompt_name}' in prompts.json")

    main_prompt = all_prompts[prompt_name]

    # -------------------------------------------------------------------
    # 3) Prepare format reminder and fallback prompts
    # -------------------------------------------------------------------
    format_reminder = (
        "Please follow the exact format I requested with <analysis> and <labels> blocks."
        "The <labels> block should contain ONLY a JSON object in this exact format:\n"
        "{\n"
        '  "damage_severity": one of ["little_or_none","mild","severe"],\n'
        '  "informative": one of ["not_informative","informative"],\n'
        '  "humanitarian": one of ["affected_injured_or_dead_people","infrastructure_and_utility_damage","not_humanitarian","rescue_volunteering_or_donation_effort"],\n'
        '  "disaster_type": one of ["earthquake","fire","flood","hurricane","landslide","not_disaster","other_disaster"]\n'
        "}"
    )

    fallback_prompts = {
        "fallback_1": (
            "I understand your caution with potentially sensitive content. To clarify, this is a legitimate academic research project "
            "on disaster classification. Your assessment will be used solely for evaluating ML classification systems that can help "
            "humanitarian organizations respond more effectively to disasters.\n\n"
            "Please classify this image according to the four categories requested (disaster_type, damage_severity, informative, humanitarian). "
            'Remember that "not_disaster" and "not_humanitarian" are perfectly valid classifications if the image shows a normal scene. '
            "If this is not a disaster image, you can and should classify it accordingly rather than refusing to analyze it.\n\n"
            "Please complete your analysis using the <analysis> and <labels> format as requested earlier."
        ),
        "fallback_2": (
            "I understand your hesitation. To clarify: analyzing this image for disaster classification is part of an academic research project "
            "with ethical approval. The classification options already include categories for non-disaster images and non-humanitarian content.\n\n"
            "Rather than refusing completely, please approach this as a technical classification task. At minimum, please determine whether "
            "this is a disaster or non-disaster image, and complete the classification using the <analysis> and <labels> format.\n\n"
            'If the image contains no disaster, simply classify it as "not_disaster" - this is valuable information for the research.'
        ),
    }

    # -------------------------------------------------------------------
    # 4) The results directory is now handled in the save section
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # 5) Prepare data processing
    # -------------------------------------------------------------------
    # Create a working copy of the DataFrame
    results_df = df.copy()

    # Add analysis columns
    results_df["LLM_analysis"] = None
    results_df["refusal_detected"] = None
    results_df["fallback_attempts"] = 0
    results_df["gpt4o_damage_severity"] = None
    results_df["gpt4o_informative"] = None
    results_df["gpt4o_humanitarian"] = None
    results_df["gpt4o_disaster_type"] = None

    # Create a model instance for reference
    llm = get_vision_llm(model_name)

    # Track overall stats
    total_images = len(results_df)
    total_refusals = 0

    # -------------------------------------------------------------------
    # 6) Process images - either sequentially or with multiprocessing
    # -------------------------------------------------------------------
    if num_workers > 1:
        print(f"Using multiprocessing with {num_workers} workers")

        # Create a list of arguments for each image
        process_args = [
            (
                idx,
                results_df.iloc[idx],
                model_name,
                main_prompt,
                format_reminder,
                fallback_prompts,
                LABEL_MAPPINGS,
                config.DATA_DIR,
            )
            for idx in range(len(results_df))
        ]

        # Initialize results list
        results = [None] * len(results_df)

        # Create a progress bar
        with tqdm(
            total=len(results_df),
            desc=f"Processing with {model_name}",
            dynamic_ncols=True,
            position=0,
            leave=True,
        ) as progress_bar:

            # Function to process and update progress
            def update_progress(result):
                idx = result["idx"]
                is_refused = result["refusal"] == "Yes"
                if is_refused:
                    nonlocal total_refusals
                    total_refusals += 1

                # Store result at correct position in results list
                results[idx] = result

                # Update progress bar
                relative_path = results_df.iloc[idx]["image_path"]
                refusal_marker = " 🚫" if is_refused else ""
                progress_bar.set_postfix_str(
                    f"#{idx}: {'✓' if not is_refused else '✗'} {os.path.basename(relative_path)}{refusal_marker}"
                )
                progress_bar.update(1)

            # Create pool and apply async processing with callback
            with multiprocessing.Pool(processes=num_workers) as pool:
                for i, args in enumerate(process_args):
                    pool.apply_async(
                        process_image, args=(args,), callback=update_progress
                    )

                # Wait for all workers to complete
                pool.close()
                pool.join()

        # Process all results and update DataFrame
        for result in results:
            if result is None:
                continue

            idx = result["idx"]

            # Update DataFrame with results
            results_df.at[idx, "LLM_analysis"] = result["analysis"]
            results_df.at[idx, "refusal_detected"] = result["refusal"]
            results_df.at[idx, "fallback_attempts"] = result["fallback_attempts"]

            parsed_resp = result["parsed_resp"]
            results_df.at[idx, "gpt4o_damage_severity"] = parsed_resp["damage_severity"]
            results_df.at[idx, "gpt4o_informative"] = parsed_resp["informative"]
            results_df.at[idx, "gpt4o_humanitarian"] = parsed_resp["humanitarian"]

            # Handle the field name difference between prompt and code
            disaster_type_key = (
                "disaster_type" if "disaster_type" in parsed_resp else "disaster_types"
            )
            results_df.at[idx, "gpt4o_disaster_type"] = parsed_resp[disaster_type_key]

    else:
        # Sequential processing with progress bar
        progress_bar = tqdm(
            total=len(results_df),
            desc=f"Processing with {model_name}",
        )

        for idx in range(len(results_df)):
            row = results_df.iloc[idx]
            relative_path = row["image_path"]
            full_path = os.path.join(config.DATA_DIR, relative_path)

            # Track refusal handling
            refusal_attempts = []
            all_responses = []

            # Make sure the image exists
            if not os.path.exists(full_path):
                error_msg = f"Error: Image file not found at {full_path}"
                results_df.at[idx, "LLM_analysis"] = error_msg
                results_df.at[idx, "refusal_detected"] = "Error"
                results_df.at[idx, "fallback_attempts"] = 0
                results_df.at[idx, "gpt4o_damage_severity"] = "N/A"
                results_df.at[idx, "gpt4o_informative"] = "N/A"
                results_df.at[idx, "gpt4o_humanitarian"] = "N/A"
                results_df.at[idx, "gpt4o_disaster_type"] = "N/A"

                progress_bar.update(1)
                continue

            # Call the LLM for the first time
            try:
                result = llm.classify_image(image_path=full_path, prompt=main_prompt)

                # We expect a text response that contains <analysis> and <labels> blocks
                raw_response = result.get("summary", "")
                all_responses.append(f"INITIAL RESPONSE:\n{raw_response}")

                # Extract the analysis and labels from the response
                analysis_text = extract_analysis(raw_response)
                parsed_resp = extract_labels(raw_response)

                # Check if this is a refusal
                is_refused = is_refusal(raw_response)
                if is_refused:
                    total_refusals += 1

                # Check if the response has the EXACT valid format according to label_mappings
                has_valid_format = is_valid_response(parsed_resp, LABEL_MAPPINGS)

                # Try fallback prompts if the LLM refuses
                current_prompt_text = main_prompt
                if is_refused:
                    refusal_attempts.append(
                        "Initial prompt: LLM refused to analyze the image"
                    )

                    # Try each fallback prompt
                    for fallback_name, fallback_text in fallback_prompts.items():
                        fallback_prompt = f"{current_prompt_text}\n\n{fallback_text}"
                        fallback_result = llm.classify_image(
                            image_path=full_path, prompt=fallback_prompt
                        )
                        fallback_response = fallback_result.get("summary", "")
                        all_responses.append(
                            f"\n\n{fallback_name.upper()}:\n{fallback_response}"
                        )

                        # Check if fallback worked
                        if not is_refusal(fallback_response):
                            refusal_attempts.append(
                                f"Fallback {fallback_name}: Successful"
                            )
                            # Extract the analysis and labels from the fallback response
                            fallback_analysis = extract_analysis(fallback_response)
                            fallback_parsed = extract_labels(fallback_response)

                            # If we got valid analysis and labels, use them
                            if fallback_analysis:
                                analysis_text = fallback_analysis
                            if is_valid_response(fallback_parsed, LABEL_MAPPINGS):
                                parsed_resp = fallback_parsed
                                has_valid_format = True
                                break
                        else:
                            refusal_attempts.append(
                                f"Fallback {fallback_name}: LLM still refused"
                            )

                # If no refusal or we've handled it but format is still not valid,
                # try format correction
                if not has_valid_format:
                    # Format correction attempt
                    corrected_prompt = f"{current_prompt_text}\n\n{format_reminder}"

                    second_result = llm.classify_image(
                        image_path=full_path, prompt=corrected_prompt
                    )
                    second_raw_response = second_result.get("summary", "")
                    all_responses.append(
                        f"\n\nFORMAT CORRECTION ATTEMPT:\n{second_raw_response}"
                    )

                    # Try to extract again from the second response
                    second_analysis = extract_analysis(second_raw_response)
                    if second_analysis:
                        analysis_text = second_analysis
                    second_parsed_resp = extract_labels(second_raw_response)

                    if is_valid_response(second_parsed_resp, LABEL_MAPPINGS):
                        parsed_resp = second_parsed_resp
                        has_valid_format = True

                # If it's still invalid after all attempts, store placeholders
                if not has_valid_format:
                    parsed_resp = {
                        "damage_severity": "N/A",
                        "informative": "N/A",
                        "humanitarian": "N/A",
                        "disaster_type": "N/A",
                    }

                # Store the analysis and predictions
                results_df.at[idx, "LLM_analysis"] = "\n".join(all_responses)
                results_df.at[idx, "refusal_detected"] = "Yes" if is_refused else "No"
                results_df.at[idx, "fallback_attempts"] = len(refusal_attempts)
                results_df.at[idx, "gpt4o_damage_severity"] = parsed_resp[
                    "damage_severity"
                ]
                results_df.at[idx, "gpt4o_informative"] = parsed_resp["informative"]
                results_df.at[idx, "gpt4o_humanitarian"] = parsed_resp["humanitarian"]

                # Handle the field name difference between prompt and code
                disaster_type_key = (
                    "disaster_type"
                    if "disaster_type" in parsed_resp
                    else "disaster_types"
                )
                results_df.at[idx, "gpt4o_disaster_type"] = parsed_resp[
                    disaster_type_key
                ]

            except Exception as e:
                # Handle errors
                error_message = f"Error processing {relative_path}: {str(e)}\n{traceback.format_exc()}"
                results_df.at[idx, "LLM_analysis"] = error_message
                results_df.at[idx, "refusal_detected"] = "Error"
                results_df.at[idx, "fallback_attempts"] = 0
                results_df.at[idx, "gpt4o_damage_severity"] = "N/A"
                results_df.at[idx, "gpt4o_informative"] = "N/A"
                results_df.at[idx, "gpt4o_humanitarian"] = "N/A"
                results_df.at[idx, "gpt4o_disaster_type"] = "N/A"

            # Update progress bar
            refusal_marker = " 🚫" if is_refused else ""
            progress_bar.set_postfix_str(
                f"{'✓' if not is_refused else '✗'} {os.path.basename(relative_path)}{refusal_marker}"
            )
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()

    # -------------------------------------------------------------------
    # 7) Save the results
    # -------------------------------------------------------------------
    # Use fixed output filename with test indicator if needed
    if test_mode:
        output_filename = "test_zero_shot_classification.csv"
    else:
        output_filename = "final_zero_shot_classification.csv"

    # Create results directory path - ensure it's relative
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory at: {results_dir}")

    output_path = os.path.join(results_dir, output_filename)

    # Save results to CSV
    results_df.to_csv(output_path, index=False)

    # Calculate and display metrics
    end_time = time.time()
    elapsed_time = end_time - start_time
    images_per_second = total_images / elapsed_time if elapsed_time > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"SUMMARY")
    print(f"{'=' * 50}")
    print(f"Processed {total_images} images in {elapsed_time:.1f} seconds")
    print(f"Speed: {images_per_second:.2f} images/sec")
    print(f"Total refusals: {total_refusals} ({total_refusals/total_images:.1%})")
    print(f"Results saved to: {output_path}")

    return results_df


"""
Simple functions to analyze GPT-4o performance and display confusion matrices 
directly in the notebook, without saving files.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


"""
Simple functions to analyze GPT-4o performance and display confusion matrices 
directly in the notebook, with proper data type handling.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


"""
Direct replica of the original functions, adapted only for the different file format.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


def analyze_final_performance(results_csv_filename: str):
    """
    Analyzes the performance of the GPT-4o model on the final test dataset.
    Displays a styled comparison table with both accuracy and F1 scores.

    Args:
        results_csv_filename (str):
            Name of the CSV file with results, located in the src/zero_shot/results directory.
    """
    import os
    import pandas as pd
    import numpy as np
    from sklearn.metrics import f1_score
    from IPython.display import display, HTML

    # Construct the full path to the results CSV
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    results_path = os.path.join(results_dir, results_csv_filename)

    # Read the results CSV
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found at: {results_path}")

    df = pd.read_csv(results_path)

    # Check if necessary columns exist
    required_cols = [
        "damage_severity",
        "informative",
        "humanitarian",
        "disaster_types",  # Ground truth
        "gpt4o_damage_severity",
        "gpt4o_informative",
        "gpt4o_humanitarian",
        "gpt4o_disaster_type",  # Predictions
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Define the categories and their classes
    categories = {
        "Damage Severity": {
            "truth_column": "damage_severity",
            "pred_column": "gpt4o_damage_severity",
            "classes": ["little_or_none", "mild", "severe"],
        },
        "Informative": {
            "truth_column": "informative",
            "pred_column": "gpt4o_informative",
            "classes": ["not_informative", "informative"],
        },
        "Humanitarian": {
            "truth_column": "humanitarian",
            "pred_column": "gpt4o_humanitarian",
            "classes": [
                "affected_injured_or_dead_people",
                "infrastructure_and_utility_damage",
                "not_humanitarian",
                "rescue_volunteering_or_donation_effort",
            ],
        },
        "Disaster Types": {
            "truth_column": "disaster_types",
            "pred_column": "gpt4o_disaster_type",
            "classes": [
                "earthquake",
                "fire",
                "flood",
                "hurricane",
                "landslide",
                "not_disaster",
                "other_disaster",
            ],
        },
    }

    # Create dictionaries to store the results
    accuracy_results = {"gpt4o": {}}
    f1_results = {"gpt4o": {}}
    all_accuracies = {"gpt4o": []}
    all_f1s = {"gpt4o": []}

    # Calculate metrics for each category
    for category_name, category_info in categories.items():
        truth_col = category_info["truth_column"]
        pred_col = category_info["pred_column"]

        if category_name not in accuracy_results["gpt4o"]:
            accuracy_results["gpt4o"][category_name] = {}
            f1_results["gpt4o"][category_name] = {}

        # Skip N/A predictions
        valid_preds = df[pred_col] != "N/A"
        valid_df = df[valid_preds]

        if len(valid_df) > 0:
            # Convert data to consistent string type to avoid comparison issues
            y_true = valid_df[truth_col].astype(str).values
            y_pred = valid_df[pred_col].astype(str).values

            # Calculate accuracy (string comparison works fine for this)
            accuracy = (y_true == y_pred).mean() * 100
            accuracy_results["gpt4o"][category_name]["Overall"] = accuracy
            all_accuracies["gpt4o"].append(accuracy)

            # Calculate per-class metrics
            class_f1_values = []
            for class_name in category_info["classes"]:
                # Get rows where the true label is this class
                class_mask = y_true == class_name

                if np.any(class_mask):
                    # Calculate class accuracy
                    class_accuracy = (y_pred[class_mask] == class_name).mean() * 100
                    accuracy_results["gpt4o"][category_name][
                        class_name
                    ] = class_accuracy

                    # Calculate class F1 score
                    try:
                        # Convert to binary classification for this class
                        binary_true = (y_true == class_name).astype(int)
                        binary_pred = (y_pred == class_name).astype(int)

                        # Calculate F1 for this class
                        class_f1 = (
                            f1_score(binary_true, binary_pred, zero_division=0) * 100
                        )
                        f1_results["gpt4o"][category_name][class_name] = class_f1
                        class_f1_values.append(class_f1)
                    except Exception as e:
                        print(f"Error calculating F1 for class {class_name}: {str(e)}")
                        f1_results["gpt4o"][category_name][class_name] = 0.0
                else:
                    # No examples of this class in the dataset
                    accuracy_results["gpt4o"][category_name][class_name] = float("nan")
                    f1_results["gpt4o"][category_name][class_name] = float("nan")

            # Calculate task F1 score as the macro average of class F1 scores
            if class_f1_values:
                macro_f1 = np.mean(class_f1_values)
                f1_results["gpt4o"][category_name]["Overall"] = macro_f1
                all_f1s["gpt4o"].append(macro_f1)
            else:
                f1_results["gpt4o"][category_name]["Overall"] = 0.0
                all_f1s["gpt4o"].append(0.0)
        else:
            # No valid predictions
            accuracy_results["gpt4o"][category_name]["Overall"] = 0.0
            f1_results["gpt4o"][category_name]["Overall"] = 0.0
            all_accuracies["gpt4o"].append(0.0)
            all_f1s["gpt4o"].append(0.0)

            for class_name in category_info["classes"]:
                accuracy_results["gpt4o"][category_name][class_name] = float("nan")
                f1_results["gpt4o"][category_name][class_name] = float("nan")

    # Calculate overall metrics
    overall_accuracy = {"gpt4o": np.nanmean(all_accuracies["gpt4o"])}
    overall_f1 = {"gpt4o": np.nanmean(all_f1s["gpt4o"])}

    # Generate HTML table
    html = '<table class="styled-table" style="border-collapse: collapse; width: 100%; border: 1px solid #ddd;">'

    # Add title row
    html += f'<tr><th colspan="3" style="padding: 12px; background-color: #222; color: white; text-align: center; font-size: 1.2em;">Final GPT-4o Model Performance</th></tr>'

    # Table header
    html += '<tr><th style="padding: 8px; background-color: #333; color: white; text-align: left; width: 30%;">Category / Class</th>'
    html += '<th style="padding: 8px; background-color: #444; color: white; text-align: center;">Acc (%)</th>'
    html += '<th style="padding: 8px; background-color: #444; color: white; text-align: center;">F1 (%)</th></tr>'

    # Format function for metric values
    def format_metric(value):
        if pd.isna(value):
            return "-"
        elif value == 100.0:
            return "100"
        else:
            return f"{value:.1f}"

    # Table rows with results
    row_count = 0
    for category_name, category_info in categories.items():
        row_color = "#3a3a3a" if row_count % 2 == 0 else "#333"

        # Category header row (bold)
        html += f'<tr style="background-color: {row_color};">'
        html += f'<td style="padding: 8px; font-weight: bold; color: white; text-align: left;">{category_name}</td>'

        # Accuracy value
        acc = accuracy_results["gpt4o"][category_name].get("Overall", 0.0)
        html += f'<td style="padding: 8px; text-align: center; color: white;">{format_metric(acc)}</td>'

        # F1 value
        f1 = f1_results["gpt4o"][category_name].get("Overall", 0.0)
        html += f'<td style="padding: 8px; text-align: center; color: white;">{format_metric(f1)}</td>'

        html += "</tr>"

        # Class rows
        for class_name in category_info["classes"]:
            row_count += 1
            row_color = "#3a3a3a" if row_count % 2 == 0 else "#333"

            # Format class name with spaces and capitalize
            formatted_class_name = " ".join(
                word.capitalize() for word in class_name.split("_")
            )

            html += f'<tr style="background-color: {row_color};">'
            html += f'<td style="padding: 8px; padding-left: 20px; color: white; text-align: left;">{formatted_class_name}</td>'

            # Class accuracy
            class_acc = accuracy_results["gpt4o"][category_name].get(
                class_name, float("nan")
            )
            html += f'<td style="padding: 8px; text-align: center; color: white;">{format_metric(class_acc)}</td>'

            # Class F1
            class_f1 = f1_results["gpt4o"][category_name].get(class_name, float("nan"))
            html += f'<td style="padding: 8px; text-align: center; color: white;">{format_metric(class_f1)}</td>'

            html += "</tr>"

        row_count += 1

    # Add overall total row
    html += f'<tr style="background-color: #222; font-weight: bold;">'
    html += f'<td style="padding: 8px; color: white; text-align: left;">OVERALL</td>'

    # Overall accuracy
    acc_value = overall_accuracy["gpt4o"]
    html += f'<td style="padding: 8px; text-align: center; color: white;">{format_metric(acc_value)}</td>'

    # Overall F1
    f1_value = overall_f1["gpt4o"]
    html += f'<td style="padding: 8px; text-align: center; color: white;">{format_metric(f1_value)}</td>'

    html += "</tr>"
    html += "</table>"

    # Display the table
    display(HTML(html))


def display_final_conf_matrices(results_csv_filename: str, save_dir: str = None):
    """
    Generates and displays confusion matrices for GPT-4o predictions across all tasks.
    Creates a single row of four confusion matrices (one for each task).

    Args:
        results_csv_filename (str):
            Name of the CSV file with results, located in the src/zero_shot/results directory.
        save_dir (str, optional):
            Directory to save the generated confusion matrix plots. If None, plots are only displayed.
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # Construct the full path to the results CSV
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    results_path = os.path.join(results_dir, results_csv_filename)

    # Read the results CSV
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found at: {results_path}")

    df = pd.read_csv(results_path)

    # Check if necessary columns exist
    required_cols = [
        "damage_severity",
        "informative",
        "humanitarian",
        "disaster_types",  # Ground truth
        "gpt4o_damage_severity",
        "gpt4o_informative",
        "gpt4o_humanitarian",
        "gpt4o_disaster_type",  # Predictions
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Task mapping configs
    tasks = [
        ("damage_severity", "gpt4o_damage_severity"),
        ("informative", "gpt4o_informative"),
        ("humanitarian", "gpt4o_humanitarian"),
        ("disaster_types", "gpt4o_disaster_type"),
    ]

    task_names = {
        "damage_severity": ["none", "mild", "severe"],
        "informative": ["not inf", "inf"],
        "humanitarian": ["injured", "infra", "not hum", "rescue"],
        "disaster_types": ["quake", "fire", "flood", "hurr.", "land.", "none", "other"],
    }

    # Dictionary to convert task names to proper case titles
    task_titles = {
        "damage_severity": "Damage Severity",
        "informative": "Informative",
        "humanitarian": "Humanitarian",
        "disaster_types": "Disaster Types",
    }

    # Create dictionaries to store true labels and predictions for each task
    true_labels_dict = {}
    predictions_dict = {}

    # Extract true labels and predictions for each task
    for truth_col, pred_col in tasks:
        task_name = truth_col  # Use the truth column name as the task name

        # Skip N/A predictions and ensure we have both true and prediction values
        valid_preds = (
            (df[pred_col] != "N/A") & (~df[truth_col].isna()) & (~df[pred_col].isna())
        )
        valid_df = df[valid_preds]

        if len(valid_df) > 0:
            # Get true labels and predictions - ensure both are strings
            true_labels_dict[task_name] = valid_df[truth_col].astype(str).values
            predictions_dict[task_name] = valid_df[pred_col].astype(str).values
        else:
            print(f"Warning: No valid predictions for task {task_name}")
            true_labels_dict[task_name] = []
            predictions_dict[task_name] = []

    # Skip if any task has no valid predictions
    if any(len(true_labels_dict[task_name]) == 0 for task_name, _ in tasks):
        print("Skipping due to missing valid predictions")
        return

    # Set up the plotting style
    plt.style.use("default")
    sns.set_theme()

    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Create figure with four subplots in a row
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"GPT-4o Confusion Matrices", fontsize=16, y=1.05)

    for idx, ((task_name, _), ax) in enumerate(zip(tasks, axes)):
        if len(true_labels_dict[task_name]) > 0:
            try:
                # Compute confusion matrix
                cm = confusion_matrix(
                    true_labels_dict[task_name], predictions_dict[task_name]
                )
            except Exception as e:
                print(f"Error creating confusion matrix for task {task_name}: {str(e)}")
                # Create empty placeholder
                ax.text(
                    0.5,
                    0.5,
                    f"Error: {str(e)}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="red",
                )
                ax.set_title(task_titles[task_name])
                continue

            # Normalize by row (true labels)
            cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            # Create custom normalization for visualization
            # Positive values on the diagonal, negative values elsewhere
            custom_norm = np.zeros_like(cm_normalized)
            for i in range(len(custom_norm)):
                for j in range(len(custom_norm[i])):
                    if i == j:
                        custom_norm[i, j] = cm_normalized[i, j]
                    else:
                        custom_norm[i, j] = -cm_normalized[i, j]

            # Format annotations (showing the actual normalized values)
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

            # Create heatmap
            sns.heatmap(
                custom_norm,
                annot=annotations,
                fmt="",
                cmap=sns.diverging_palette(10, 240, s=100, l=40, n=9),
                xticklabels=task_names[task_name],
                yticklabels=task_names[task_name],
                ax=ax,
                cbar=False,
                annot_kws={"fontsize": 12},
                vmin=-0.5,
                center=0,
                vmax=1.0,
            )

            # Set titles and labels
            ax.set_title(task_titles[task_name])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=30, ha="right", fontsize=12
            )
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        else:
            # No data for this task
            ax.text(
                0.5,
                0.5,
                "No valid data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title(task_titles[task_name])

    # Adjust layout
    plt.tight_layout(w_pad=1.5, h_pad=1.5)

    # Save if save_dir is specified
    if save_dir:
        filename = "gpt4o_confusion_matrices.png"
        plt.savefig(
            os.path.join(save_dir, filename),
            bbox_inches="tight",
            dpi=300,
        )
        print(f"Saved to {os.path.join(save_dir, filename)}")

    # Display the plot
    plt.show()
    plt.close()
