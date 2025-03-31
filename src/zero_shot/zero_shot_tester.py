"""
zero_shot_tester.py

A utility module providing a function to run zero-shot inference using
a selected vision LLM on a given test CSV file, with support for multiprocessing.
It reads each row, constructs a full image path, calls the LLM with a specified prompt,
extracts the <analysis> and <labels> blocks, validates the returned JSON,
and writes a new CSV with appended columns for the analysis and predictions.

Usage (from Jupyter notebook or script):
    from src.zero_shot.testing.zero_shot_tester import run_zero_shot_inference

    run_zero_shot_inference(
        test_csv_filename="small_test.csv",
        model_name="gpt4v",
        prompt_name="my_prompt",
        num_workers=4  # Use 4 parallel processes
    )
"""

import os
import json
import pandas as pd
import re
from tqdm.auto import tqdm
import multiprocessing
from functools import partial
import time
import traceback

# Relative import for your model factory, config, etc.
from src.zero_shot.testing.models import get_vision_llm
import config

# Example of how label mappings might appear in code.
LABEL_MAPPINGS = {
    "damage_severity": {0: "little_or_none", 1: "mild", 2: "severe"},
    "informative": {0: "not_informative", 1: "informative"},
    "humanitarian": {
        0: "affected_injured_or_dead_people",
        1: "infrastructure_and_utility_damage",
        2: "not_humanitarian",
        3: "rescue_volunteering_or_donation_effort",
    },
    "disaster_types": {
        0: "earthquake",
        1: "fire",
        2: "flood",
        3: "hurricane",
        4: "landslide",
        5: "not_disaster",
        6: "other_disaster",
    },
}


# Helper functions to extract analysis and labels sections from response
def extract_analysis(text):
    # Try standard format first
    analysis_match = re.search(
        r"<analysis>(.*?)</analysis>", text, re.DOTALL | re.IGNORECASE
    )
    if not analysis_match:
        # Try alternative formats with escaped slashes
        analysis_match = re.search(
            r"<\\analysis>(.*?)</\\analysis>", text, re.DOTALL | re.IGNORECASE
        )
    if not analysis_match:
        # Try with backslashed closing tags
        analysis_match = re.search(
            r"<analysis>(.*?)<\/analysis>", text, re.DOTALL | re.IGNORECASE
        )

    if analysis_match:
        return analysis_match.group(1).strip()
    return ""


def extract_labels(text):
    # Try standard formats with tags first
    for pattern in [
        r"<labels>(.*?)</labels>",  # Standard tags
        r"<labels>(.*?)<\/labels>",  # Escaped closing tag
        r"<\\labels>(.*?)</\\labels>",  # Escaped opening and closing tags
    ]:
        labels_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if labels_match:
            labels_text = labels_match.group(1).strip()
            try:
                # Remove any surrounding backticks from JSON code blocks
                labels_text = re.sub(r"^```(json)?\s*|\s*```$", "", labels_text)
                return json.loads(labels_text)
            except json.JSONDecodeError:
                # Try one simple fix - replace single quotes with double quotes
                try:
                    fixed_text = labels_text.replace("'", '"')
                    return json.loads(fixed_text)
                except:
                    pass  # Continue to next approach

    # If no tagged labels found, look for JSON-like structure in the text
    # This handles cases where the model provides labels but without proper tags
    json_patterns = [
        r'({[\s\n]*"disaster_type"[\s\n]*:.*?"humanitarian"[\s\n]*:.*?})',
        r'({[\s\n]*"damage_severity"[\s\n]*:.*?"disaster_type"[\s\n]*:.*?})',
        r'(\{[^{}]*"disaster_type"[^{}]*"damage_severity"[^{}]*"informative"[^{}]*"humanitarian"[^{}]*\})',
        r"the labels would be:[\s\n]*(\{.*?\})",
        r"labels are:[\s\n]*(\{.*?\})",
    ]

    for pattern in json_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                # Clean up the matched text (replace newlines, fix quotes)
                json_text = match.group(1).strip()
                json_text = re.sub(r"[\n\r]", " ", json_text)
                json_text = json_text.replace("'", '"')
                return json.loads(json_text)
            except:
                pass  # Try next pattern if this one fails

    return {}


# Helper function to check if the response is valid in the EXACT required format
def is_valid_response(resp_dict, mappings):
    required_keys = [
        "damage_severity",
        "informative",
        "humanitarian",
        "disaster_type",
    ]  # Note: changed from disaster_types to disaster_type

    # Must have all required keys
    for key in required_keys:
        if key not in resp_dict:
            return False
        # Must be among the allowed values in label_mappings
        # Handle the field name difference between prompt and code
        mapping_key = "disaster_types" if key == "disaster_type" else key
        valid_values = list(mappings[mapping_key].values())
        if resp_dict[key] not in valid_values:
            return False

    return True


# Helper function to detect if the LLM is refusing to analyze the image
# 1. First let's improve the extract_labels function to handle untagged JSON
def extract_labels(text):
    # Try standard formats with tags first
    for pattern in [
        r"<labels>(.*?)</labels>",  # Standard tags
        r"<labels>(.*?)<\/labels>",  # Escaped closing tag
        r"<\\labels>(.*?)</\\labels>",  # Escaped opening and closing tags
    ]:
        labels_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if labels_match:
            labels_text = labels_match.group(1).strip()
            try:
                # Remove any surrounding backticks from JSON code blocks
                labels_text = re.sub(r"^```(json)?\s*|\s*```$", "", labels_text)
                return json.loads(labels_text)
            except json.JSONDecodeError:
                # Try one simple fix - replace single quotes with double quotes
                try:
                    fixed_text = labels_text.replace("'", '"')
                    return json.loads(fixed_text)
                except:
                    pass  # Continue to next approach

    # If no tagged labels found, look for JSON-like structure in the text
    # This handles cases where the model provides labels but without proper tags
    json_patterns = [
        r'({[\s\n]*"disaster_type"[\s\n]*:.*?"humanitarian"[\s\n]*:.*?})',
        r'({[\s\n]*"damage_severity"[\s\n]*:.*?"disaster_type"[\s\n]*:.*?})',
        r'(\{[^{}]*"disaster_type"[^{}]*"damage_severity"[^{}]*"informative"[^{}]*"humanitarian"[^{}]*\})',
        r"the labels would be:[\s\n]*(\{.*?\})",
        r"labels are:[\s\n]*(\{.*?\})",
    ]

    for pattern in json_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                # Clean up the matched text (replace newlines, fix quotes)
                json_text = match.group(1).strip()
                json_text = re.sub(r"[\n\r]", " ", json_text)
                json_text = json_text.replace("'", '"')
                return json.loads(json_text)
            except:
                pass  # Try next pattern if this one fails

    return {}


# 2. Updated is_refusal function
def is_refusal(text):
    # More specific refusal patterns that indicate complete task refusal
    explicit_refusal_patterns = [
        # Complete task refusals
        "I cannot analyze this image",
        "I'm unable to analyze this image",
        "I cannot provide the requested analysis",
        "I cannot classify this image",
        "I apologize, but I cannot comply with this request",
        "I'm not able to complete this task",
        "I cannot in good conscience",
        "would violate ethical guidelines",
        "cannot perform this analysis due to",
        "I must decline to analyze",
        # Policy-based refusals
        "policy does not allow me to",
        "against my content policy",
        "violates content policy",
        # Sensitive content refusals (more specific)
        "contains sensitive content that I cannot",
        "image appears to contain graphic",
        "image may contain content that",
        "this image appears to show sensitive",
    ]

    # Use our improved extraction to get any labels
    labels = extract_labels(text)

    # Check if we have valid labels with all required fields
    required_fields = [
        "damage_severity",
        "informative",
        "humanitarian",
        "disaster_type",
    ]
    all_fields_valid = all(
        key in labels and labels[key] != "N/A" and labels[key] != ""
        for key in required_fields
    )

    # If we have all valid fields, it's definitely not a refusal
    if all_fields_valid:
        return False

    # Check if the response starts with a clear refusal
    first_100_chars = text[:100].lower()
    starting_refusals = ["i apologize", "i'm sorry, but i", "i'm unable to", "i cannot"]
    starts_with_refusal = any(
        first_100_chars.startswith(phrase) for phrase in starting_refusals
    )

    # Check for explicit refusal patterns
    contains_explicit_refusal = any(
        pattern.lower() in text.lower() for pattern in explicit_refusal_patterns
    )

    # Only consider it a refusal if it contains an explicit refusal pattern
    # AND doesn't have valid labels
    return contains_explicit_refusal and not all_fields_valid


# Worker function to process a single image
def process_image(args):
    # If args is passed as a tuple by apply_async, extract it
    if isinstance(args, tuple) and len(args) == 1 and isinstance(args[0], tuple):
        args = args[0]

    (
        idx,
        row,
        model_name,
        main_prompt,
        format_reminder,
        fallback_prompts,
        label_mappings,
        data_dir,
    ) = args

    try:
        # Create a new model instance for each process to avoid conflicts
        llm = get_vision_llm(model_name)

        relative_path = row["image_path"]
        full_path = os.path.join(data_dir, relative_path)

        # Make sure the image exists
        if not os.path.exists(full_path):
            return {
                "idx": idx,
                "analysis": f"Error: Image file not found at {full_path}",
                "refusal": "Error",
                "fallback_attempts": 0,
                "parsed_resp": {
                    "damage_severity": "N/A",
                    "informative": "N/A",
                    "humanitarian": "N/A",
                    "disaster_type": "N/A",
                },
            }

        # Track refusal handling
        refusal_attempts = []
        all_responses = []

        # Call the LLM for the first time
        result = llm.classify_image(image_path=full_path, prompt=main_prompt)

        # We expect a text response that contains <analysis> and <labels> blocks
        raw_response = result.get("summary", "")
        all_responses.append(f"INITIAL RESPONSE:\n{raw_response}")

        # Extract the analysis and labels from the response
        analysis_text = extract_analysis(raw_response)
        parsed_resp = extract_labels(raw_response)

        # Check if this is a refusal
        is_refused = is_refusal(raw_response)

        # Check if the response has the EXACT valid format according to label_mappings
        has_valid_format = is_valid_response(parsed_resp, label_mappings)

        # Try fallback prompts if the LLM refuses or didn't format properly
        current_prompt = main_prompt

        # Only try fallbacks if we have a refusal
        if is_refused:
            refusal_attempts.append("Initial prompt: LLM refused to analyze the image")

            # Try each fallback prompt
            for fallback_name, fallback_text in fallback_prompts.items():
                fallback_prompt = f"{current_prompt}\n\n{fallback_text}"
                fallback_result = llm.classify_image(
                    image_path=full_path, prompt=fallback_prompt
                )
                fallback_response = fallback_result.get("summary", "")
                all_responses.append(
                    f"\n\n{fallback_name.upper()}:\n{fallback_response}"
                )

                # Check if fallback worked
                if not is_refusal(fallback_response):
                    refusal_attempts.append(f"Fallback {fallback_name}: Successful")
                    # Extract the analysis and labels from the fallback response
                    fallback_analysis = extract_analysis(fallback_response)
                    fallback_parsed = extract_labels(fallback_response)

                    # If we got valid analysis and labels, use them
                    if fallback_analysis:
                        analysis_text = fallback_analysis
                    if is_valid_response(fallback_parsed, label_mappings):
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
            corrected_prompt = f"{current_prompt}\n\n{format_reminder}"

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

            if is_valid_response(second_parsed_resp, label_mappings):
                parsed_resp = second_parsed_resp
                has_valid_format = True

        # If it's still invalid after all attempts, store placeholders
        if not has_valid_format:
            parsed_resp = {
                "damage_severity": "N/A",
                "informative": "N/A",
                "humanitarian": "N/A",
                "disaster_type": "N/A",  # Note: using disaster_type here
            }

        # Combine all raw responses
        combined_raw_responses = "\n".join(all_responses)

        # Create a result dictionary to return
        return {
            "idx": idx,
            "analysis": combined_raw_responses,  # Now storing all raw responses
            "refusal": "Yes" if is_refused else "No",
            "fallback_attempts": len(refusal_attempts),
            "parsed_resp": parsed_resp,
        }

    except Exception as e:
        # Return error information if any exception occurs
        error_message = (
            f"Error processing {row['image_path']}: {str(e)}\n{traceback.format_exc()}"
        )
        return {
            "idx": idx,
            "analysis": error_message,
            "refusal": "Error",
            "fallback_attempts": 0,
            "parsed_resp": {
                "damage_severity": "N/A",
                "informative": "N/A",
                "humanitarian": "N/A",
                "disaster_type": "N/A",
            },
        }


import os
import json
import pandas as pd
import re
from tqdm.auto import tqdm
import multiprocessing
from functools import partial
import time
import traceback
from typing import Union, List

# Relative import for your model factory, config, etc.
from src.zero_shot.testing.models import get_vision_llm
import config

# Example of how label mappings might appear in code.
LABEL_MAPPINGS = {
    "damage_severity": {0: "little_or_none", 1: "mild", 2: "severe"},
    "informative": {0: "not_informative", 1: "informative"},
    "humanitarian": {
        0: "affected_injured_or_dead_people",
        1: "infrastructure_and_utility_damage",
        2: "not_humanitarian",
        3: "rescue_volunteering_or_donation_effort",
    },
    "disaster_types": {
        0: "earthquake",
        1: "fire",
        2: "flood",
        3: "hurricane",
        4: "landslide",
        5: "not_disaster",
        6: "other_disaster",
    },
}


# Helper functions to extract analysis and labels sections from response
def extract_analysis(text):
    # Try multiple tag formats
    for pattern in [
        r"<analysis>(.*?)</analysis>",  # Standard tags
        r"<analysis>(.*?)<\/analysis>",  # Escaped closing tag
        r"<\\analysis>(.*?)</\\analysis>",  # Escaped opening and closing tags
    ]:
        analysis_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if analysis_match:
            return analysis_match.group(1).strip()
    return ""


def extract_labels(text):
    # Try standard formats with tags first
    for pattern in [
        r"<labels>(.*?)</labels>",  # Standard tags
        r"<labels>(.*?)<\/labels>",  # Escaped closing tag
        r"<\\labels>(.*?)</\\labels>",  # Escaped opening and closing tags
    ]:
        labels_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if labels_match:
            labels_text = labels_match.group(1).strip()
            try:
                # Remove any surrounding backticks from JSON code blocks
                labels_text = re.sub(r"^```(json)?\s*|\s*```$", "", labels_text)
                return json.loads(labels_text)
            except json.JSONDecodeError:
                # Try one simple fix - replace single quotes with double quotes
                try:
                    fixed_text = labels_text.replace("'", '"')
                    return json.loads(fixed_text)
                except:
                    pass  # Continue to next approach

    # If no tagged labels found, look for JSON-like structure in the text
    # This handles cases where the model provides labels but without proper tags
    json_patterns = [
        r'({[\s\n]*"disaster_type"[\s\n]*:.*?"humanitarian"[\s\n]*:.*?})',
        r'({[\s\n]*"damage_severity"[\s\n]*:.*?"disaster_type"[\s\n]*:.*?})',
        r'(\{[^{}]*"disaster_type"[^{}]*"damage_severity"[^{}]*"informative"[^{}]*"humanitarian"[^{}]*\})',
        r"the labels would be:[\s\n]*(\{.*?\})",
        r"labels are:[\s\n]*(\{.*?\})",
        r"labels:[\s\n]*(\{.*?\})",
        r"Therefore, the labels would be:[\s\n]*(\{.*?\})",
    ]

    for pattern in json_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                # Clean up the matched text (replace newlines, fix quotes)
                json_text = match.group(1).strip()
                json_text = re.sub(r"[\n\r]", " ", json_text)
                json_text = json_text.replace("'", '"')
                return json.loads(json_text)
            except:
                pass  # Try next pattern if this one fails

    return {}


# Helper function to check if the response is valid in the EXACT required format
def is_valid_response(resp_dict, mappings):
    required_keys = [
        "damage_severity",
        "informative",
        "humanitarian",
        "disaster_type",
    ]  # Note: changed from disaster_types to disaster_type

    # Must have all required keys
    for key in required_keys:
        if key not in resp_dict:
            return False
        # Must be among the allowed values in label_mappings
        # Handle the field name difference between prompt and code
        mapping_key = "disaster_types" if key == "disaster_type" else key
        valid_values = list(mappings[mapping_key].values())
        if resp_dict[key] not in valid_values:
            return False

    return True


# Helper function to detect if the LLM is refusing to analyze the image
def is_refusal(text):
    # More specific refusal patterns that indicate complete task refusal
    explicit_refusal_patterns = [
        # Complete task refusals
        "I cannot analyze this image",
        "I'm unable to analyze this image",
        "I cannot provide the requested analysis",
        "I cannot classify this image",
        "I apologize, but I cannot comply with this request",
        "I'm not able to complete this task",
        "I cannot in good conscience",
        "would violate ethical guidelines",
        "cannot perform this analysis due to",
        "I must decline to analyze",
        # Policy-based refusals
        "policy does not allow me to",
        "against my content policy",
        "violates content policy",
        # Sensitive content refusals (more specific)
        "contains sensitive content that I cannot",
        "image appears to contain graphic",
        "image may contain content that",
        "this image appears to show sensitive",
    ]

    # Use our improved extraction to get any labels
    labels = extract_labels(text)

    # Check if we have valid labels with all required fields
    required_fields = [
        "damage_severity",
        "informative",
        "humanitarian",
        "disaster_type",
    ]
    all_fields_valid = all(
        key in labels and labels[key] != "N/A" and labels[key] != ""
        for key in required_fields
    )

    # If we have all valid fields, it's definitely not a refusal
    if all_fields_valid:
        return False

    # Check if the response starts with a clear refusal
    first_100_chars = text[:100].lower()
    starting_refusals = ["i apologize", "i'm sorry, but i", "i'm unable to", "i cannot"]
    starts_with_refusal = any(
        first_100_chars.startswith(phrase) for phrase in starting_refusals
    )

    # Check for explicit refusal patterns
    contains_explicit_refusal = any(
        pattern.lower() in text.lower() for pattern in explicit_refusal_patterns
    )

    # Only consider it a refusal if it contains an explicit refusal pattern
    # AND doesn't have valid labels
    return contains_explicit_refusal and not all_fields_valid


# Worker function to process a single image
def process_image(args):
    # If args is passed as a tuple by apply_async, extract it
    if isinstance(args, tuple) and len(args) == 1 and isinstance(args[0], tuple):
        args = args[0]

    (
        idx,
        row,
        model_name,
        main_prompt,
        format_reminder,
        fallback_prompts,
        label_mappings,
        data_dir,
    ) = args

    try:
        # Create a new model instance for each process to avoid conflicts
        llm = get_vision_llm(model_name)

        relative_path = row["image_path"]
        full_path = os.path.join(data_dir, relative_path)

        # Make sure the image exists
        if not os.path.exists(full_path):
            return {
                "idx": idx,
                "analysis": f"Error: Image file not found at {full_path}",
                "refusal": "Error",
                "fallback_attempts": 0,
                "parsed_resp": {
                    "damage_severity": "N/A",
                    "informative": "N/A",
                    "humanitarian": "N/A",
                    "disaster_type": "N/A",
                },
            }

        # Track refusal handling
        refusal_attempts = []
        all_responses = []

        # Call the LLM for the first time
        result = llm.classify_image(image_path=full_path, prompt=main_prompt)

        # We expect a text response that contains <analysis> and <labels> blocks
        raw_response = result.get("summary", "")
        all_responses.append(f"INITIAL RESPONSE:\n{raw_response}")

        # Extract the analysis and labels from the response
        analysis_text = extract_analysis(raw_response)
        parsed_resp = extract_labels(raw_response)

        # Check if this is a refusal
        is_refused = is_refusal(raw_response)

        # Check if the response has the EXACT valid format according to label_mappings
        has_valid_format = is_valid_response(parsed_resp, label_mappings)

        # Try fallback prompts if the LLM refuses or didn't format properly
        current_prompt = main_prompt

        # Only try fallbacks if we have a refusal
        if is_refused:
            refusal_attempts.append("Initial prompt: LLM refused to analyze the image")

            # Try each fallback prompt
            for fallback_name, fallback_text in fallback_prompts.items():
                fallback_prompt = f"{current_prompt}\n\n{fallback_text}"
                fallback_result = llm.classify_image(
                    image_path=full_path, prompt=fallback_prompt
                )
                fallback_response = fallback_result.get("summary", "")
                all_responses.append(
                    f"\n\n{fallback_name.upper()}:\n{fallback_response}"
                )

                # Check if fallback worked
                if not is_refusal(fallback_response):
                    refusal_attempts.append(f"Fallback {fallback_name}: Successful")
                    # Extract the analysis and labels from the fallback response
                    fallback_analysis = extract_analysis(fallback_response)
                    fallback_parsed = extract_labels(fallback_response)

                    # If we got valid analysis and labels, use them
                    if fallback_analysis:
                        analysis_text = fallback_analysis
                    if is_valid_response(fallback_parsed, label_mappings):
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
            corrected_prompt = f"{current_prompt}\n\n{format_reminder}"

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

            if is_valid_response(second_parsed_resp, label_mappings):
                parsed_resp = second_parsed_resp
                has_valid_format = True

        # If it's still invalid after all attempts, store placeholders
        if not has_valid_format:
            parsed_resp = {
                "damage_severity": "N/A",
                "informative": "N/A",
                "humanitarian": "N/A",
                "disaster_type": "N/A",  # Note: using disaster_type here
            }

        # Combine all raw responses
        combined_raw_responses = "\n".join(all_responses)

        # Create a result dictionary to return
        return {
            "idx": idx,
            "analysis": combined_raw_responses,  # Now storing all raw responses
            "refusal": "Yes" if is_refused else "No",
            "fallback_attempts": len(refusal_attempts),
            "parsed_resp": parsed_resp,
        }

    except Exception as e:
        # Return error information if any exception occurs
        error_message = (
            f"Error processing {row['image_path']}: {str(e)}\n{traceback.format_exc()}"
        )
        return {
            "idx": idx,
            "analysis": error_message,
            "refusal": "Error",
            "fallback_attempts": 0,
            "parsed_resp": {
                "damage_severity": "N/A",
                "informative": "N/A",
                "humanitarian": "N/A",
                "disaster_type": "N/A",
            },
        }


def run_zero_shot_inference(
    test_csv_filename: str,
    model_name: Union[str, List[str]],  # Now accepts either string or list
    prompt_name: Union[str, List[str]],  # Now accepts either string or list
    label_mappings: dict = None,
    num_workers: int = 1,
):
    """
    Runs zero-shot inference on a CSV file of test images using one or multiple
    vision LLMs and one or multiple prompts, with multiprocessing support.

    Metadata:
        Args:
            test_csv_filename (str):
                Name of the CSV file (e.g. "small_test.csv" or "large_test.csv"),
                relative to the directory from which this function is called.

            model_name (Union[str, List[str]]):
                Either a single model identifier as a string (e.g. "gpt4v") or
                a list of model identifiers to run inference with multiple models.

            prompt_name (Union[str, List[str]]):
                Either a single prompt identifier as a string (e.g. "basic_prompt") or
                a list of prompt identifiers to run inference with multiple prompts.

            label_mappings (dict, optional):
                Dictionary specifying allowed labels for each task. If None,
                uses the default LABEL_MAPPINGS defined in this file.

            num_workers (int, optional):
                Number of worker processes to use for parallel inference.
                Default is 1 (sequential processing).

        Returns:
            None
            (Saves a CSV file in the "results" directory under src/zero_shot/testing,
             with an appropriate name based on the combination of models and prompts used.)
    """
    start_time = time.time()

    # Convert single model name and prompt name to lists for consistent processing
    if isinstance(model_name, str):
        model_names = [model_name]
        single_model = True
    else:
        model_names = model_name
        single_model = False

    if isinstance(prompt_name, str):
        prompt_names = [prompt_name]
        single_prompt = True
    else:
        prompt_names = prompt_name
        single_prompt = False

    if label_mappings is None:
        label_mappings = LABEL_MAPPINGS

    # -------------------------------------------------------------------
    # 1) Read the test CSV using pandas
    # -------------------------------------------------------------------
    df = pd.read_csv(test_csv_filename)
    if df.empty:
        print(f"Warning: The CSV '{test_csv_filename}' is empty or invalid.")
        return

    # -------------------------------------------------------------------
    # 2) Load all prompt texts from prompts.json
    # -------------------------------------------------------------------
    prompts_json_path = os.path.join(os.path.dirname(__file__), "prompts.json")
    if not os.path.exists(prompts_json_path):
        raise FileNotFoundError(f"Could not find prompts.json at: {prompts_json_path}")

    with open(prompts_json_path, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)

    # Verify all requested prompts exist
    for p_name in prompt_names:
        if p_name not in all_prompts:
            raise ValueError(f"No prompt named '{p_name}' in prompts.json")

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
    # 4) Create the results directory if it doesn't exist
    # -------------------------------------------------------------------
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory at: {results_dir}")

    # -------------------------------------------------------------------
    # 5) Process each model-prompt combination and collect results
    # -------------------------------------------------------------------
    # Create a DataFrame to store all results
    all_results = pd.DataFrame()

    # Process each model-prompt combination in turn
    total_images = len(df) * len(model_names) * len(prompt_names)
    total_refusals = 0

    # Track overall stats
    model_prompt_stats = {}

    for current_model in model_names:
        model_prompt_stats[current_model] = {}

        for current_prompt in prompt_names:
            print(f"\n{'=' * 50}")
            print(f"Processing model: {current_model} with prompt: {current_prompt}")
            print(f"{'=' * 50}")

            # Get the prompt text from the dictionary
            main_prompt = all_prompts[current_prompt]

            # Create a working copy of the original DataFrame for this model-prompt combination
            combo_df = df.copy()

            # Add model and prompt specific columns
            combo_df["model_name"] = current_model
            combo_df["prompt_name"] = current_prompt
            combo_df["LLM_analysis"] = None
            combo_df["refusal_detected"] = None
            combo_df["fallback_attempts"] = 0
            combo_df["damage_severity_prediction"] = None
            combo_df["informative_prediction"] = None
            combo_df["humanitarian_prediction"] = None
            combo_df["disaster_types_prediction"] = None

            # Create a model instance for reference (will be recreated in workers)
            llm = get_vision_llm(current_model)

            # Track refusals for this model-prompt combination
            combo_refusals = 0

            # -------------------------------------------------------------------
            # 6) Process images - either sequentially or with multiprocessing
            # -------------------------------------------------------------------
            if num_workers > 1:
                print(f"Using multiprocessing with {num_workers} workers")

                # Create a list of arguments for each image
                process_args = [
                    (
                        idx,
                        combo_df.iloc[idx],
                        current_model,
                        main_prompt,
                        format_reminder,
                        fallback_prompts,
                        label_mappings,
                        config.DATA_DIR,
                    )
                    for idx in range(len(combo_df))
                ]

                # Initialize results list
                results = [None] * len(combo_df)

                # Create a single shared progress bar
                with tqdm(
                    total=len(combo_df),
                    desc=f"Processing {current_model} with {current_prompt}",
                    dynamic_ncols=True,
                    position=0,
                    leave=True,
                ) as progress_bar:

                    # Function to process and update progress
                    def update_progress(result):
                        idx = result["idx"]
                        is_refused = result["refusal"] == "Yes"
                        if is_refused:
                            nonlocal combo_refusals
                            combo_refusals += 1

                        # Store result at correct position in results list
                        results[idx] = result

                        # Update progress bar
                        relative_path = combo_df.iloc[idx]["image_path"]
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
                    combo_df.at[idx, "LLM_analysis"] = result["analysis"]
                    combo_df.at[idx, "refusal_detected"] = result["refusal"]
                    combo_df.at[idx, "fallback_attempts"] = result["fallback_attempts"]

                    parsed_resp = result["parsed_resp"]
                    combo_df.at[idx, "damage_severity_prediction"] = parsed_resp[
                        "damage_severity"
                    ]
                    combo_df.at[idx, "informative_prediction"] = parsed_resp[
                        "informative"
                    ]
                    combo_df.at[idx, "humanitarian_prediction"] = parsed_resp[
                        "humanitarian"
                    ]

                    # Handle the field name difference between prompt and code
                    disaster_type_key = (
                        "disaster_type"
                        if "disaster_type" in parsed_resp
                        else "disaster_types"
                    )
                    combo_df.at[idx, "disaster_types_prediction"] = parsed_resp[
                        disaster_type_key
                    ]

            else:
                # Sequential processing with progress bar
                progress_bar = tqdm(
                    total=len(combo_df),
                    desc=f"Processing {current_model} with {current_prompt}",
                )

                for idx in range(len(combo_df)):
                    row = combo_df.iloc[idx]
                    relative_path = row["image_path"]
                    full_path = os.path.join(config.DATA_DIR, relative_path)

                    # Track refusal handling
                    refusal_attempts = []
                    all_responses = []

                    # Call the LLM for the first time
                    result = llm.classify_image(
                        image_path=full_path, prompt=main_prompt
                    )

                    # We expect a text response that contains <analysis> and <labels> blocks
                    raw_response = result.get("summary", "")
                    all_responses.append(f"INITIAL RESPONSE:\n{raw_response}")

                    # Extract the analysis and labels from the response
                    analysis_text = extract_analysis(raw_response)
                    parsed_resp = extract_labels(raw_response)

                    # Check if this is a refusal
                    is_refused = is_refusal(raw_response)
                    if is_refused:
                        combo_refusals += 1

                    # Check if the response has the EXACT valid format according to label_mappings
                    has_valid_format = is_valid_response(parsed_resp, label_mappings)

                    # Try fallback prompts if the LLM refuses
                    current_prompt_text = main_prompt
                    if is_refused:
                        refusal_attempts.append(
                            "Initial prompt: LLM refused to analyze the image"
                        )

                        # Try each fallback prompt
                        for fallback_name, fallback_text in fallback_prompts.items():
                            fallback_prompt = (
                                f"{current_prompt_text}\n\n{fallback_text}"
                            )
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
                                if is_valid_response(fallback_parsed, label_mappings):
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

                        if is_valid_response(second_parsed_resp, label_mappings):
                            parsed_resp = second_parsed_resp
                            has_valid_format = True

                    # If it's still invalid after all attempts, store placeholders
                    if not has_valid_format:
                        parsed_resp = {
                            "damage_severity": "N/A",
                            "informative": "N/A",
                            "humanitarian": "N/A",
                            "disaster_type": "N/A",  # Note: using disaster_type here
                        }

                    # Add refusal information to analysis text
                    if refusal_attempts:
                        refusal_summary = (
                            "\n\n==== REFUSAL HANDLING ====\n"
                            + "\n".join(refusal_attempts)
                        )
                        analysis_text = (
                            analysis_text or "LLM refused to analyze the image"
                        ) + refusal_summary

                    # -------------------------------------------------------------------
                    # Store the analysis and predictions into the DataFrame
                    # -------------------------------------------------------------------
                    combo_df.at[idx, "LLM_analysis"] = "\n".join(all_responses)
                    combo_df.at[idx, "refusal_detected"] = "Yes" if is_refused else "No"
                    combo_df.at[idx, "fallback_attempts"] = len(refusal_attempts)
                    combo_df.at[idx, "damage_severity_prediction"] = parsed_resp[
                        "damage_severity"
                    ]
                    combo_df.at[idx, "informative_prediction"] = parsed_resp[
                        "informative"
                    ]
                    combo_df.at[idx, "humanitarian_prediction"] = parsed_resp[
                        "humanitarian"
                    ]
                    # Handle the field name difference between prompt and code
                    disaster_type_key = (
                        "disaster_type"
                        if "disaster_type" in parsed_resp
                        else "disaster_types"
                    )
                    combo_df.at[idx, "disaster_types_prediction"] = parsed_resp[
                        disaster_type_key
                    ]

                    # Update progress bar
                    refusal_marker = " 🚫" if is_refused else ""
                    progress_bar.set_postfix_str(
                        f"{'✓' if not is_refused else '✗'} {os.path.basename(relative_path)}{refusal_marker}"
                    )
                    progress_bar.update(1)

                # Close the progress bar
                progress_bar.close()

            # Add model-prompt combination statistics
            refusal_rate = combo_refusals / len(combo_df)
            model_prompt_stats[current_model][current_prompt] = {
                "refusals": combo_refusals,
                "refusal_rate": refusal_rate,
                "total_images": len(combo_df),
            }

            print(
                f"\nModel {current_model} with prompt {current_prompt} refusals: {combo_refusals} ({refusal_rate:.1%})"
            )

            # Update total refusals
            total_refusals += combo_refusals

            # Append this combination's results to the combined results
            all_results = pd.concat([all_results, combo_df], ignore_index=True)

    # -------------------------------------------------------------------
    # 7) Save the consolidated CSV to the results directory
    # -------------------------------------------------------------------
    if single_model and single_prompt:
        output_filename = f"{model_names[0]}_{prompt_names[0]}.csv"
    elif single_model:
        output_filename = f"{model_names[0]}_multi_prompt_comparison.csv"
    elif single_prompt:
        output_filename = f"multi_model_{prompt_names[0]}.csv"
    else:
        output_filename = f"model_prompt_comparison.csv"

    output_path = os.path.join(results_dir, output_filename)
    all_results.to_csv(output_path, index=False)

    # Calculate and display metrics
    end_time = time.time()
    elapsed_time = end_time - start_time
    images_per_second = total_images / elapsed_time

    print(f"\n{'=' * 50}")
    print(f"SUMMARY")
    print(f"{'=' * 50}")
    print(
        f"Processed {len(df)} images with {len(model_names)} models and {len(prompt_names)} prompts in {elapsed_time:.1f} seconds"
    )
    print(f"Overall speed: {images_per_second:.2f} images/sec")
    print(f"Total refusals: {total_refusals} ({total_refusals/total_images:.1%})")

    # Print refusal rates by model and prompt
    print(f"\n{'=' * 50}")
    print(f"REFUSAL RATES BY MODEL AND PROMPT")
    print(f"{'=' * 50}")
    for model in model_names:
        print(f"\nModel: {model}")
        for prompt in prompt_names:
            stats = model_prompt_stats[model][prompt]
            print(
                f"  Prompt: {prompt} - Refusals: {stats['refusals']} ({stats['refusal_rate']:.1%})"
            )

    print(f"\nResults saved to: {output_path}")


def analyse_model_performance(results_csv_filename: str, prompt_name: str = None):
    """
    Analyzes the performance of each model from a results CSV file for a specific prompt
    and displays a styled comparison table with both accuracy and F1 scores.

    Args:
        results_csv_filename (str):
            Name of the CSV file with results, located in the src/zero_shot/testing/results directory.
        prompt_name (str, optional):
            If provided, only analyze results for this specific prompt.
            If None, use all data in the CSV (assuming only one prompt was used).
    """
    import os
    import pandas as pd
    import numpy as np
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    import re
    from IPython.display import display, HTML

    # Construct the full path to the results CSV
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    results_path = os.path.join(results_dir, results_csv_filename)

    # Read the results CSV
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found at: {results_path}")

    df = pd.read_csv(results_path)

    # Filter by prompt if specified
    if prompt_name is not None:
        if "prompt_name" in df.columns:
            df = df[df["prompt_name"] == prompt_name]
        elif "prompt" in df.columns:
            df = df[df["prompt"] == prompt_name]
        else:
            # If neither column exists, check the filename
            prompt_match = re.search(
                rf"_{re.escape(prompt_name)}\.csv$", results_csv_filename
            )
            if not prompt_match:
                print(
                    f"Warning: No prompt column found and filename doesn't match '{prompt_name}'."
                )
            # Continue with all data

        if df.empty:
            raise ValueError(
                f"No data found for prompt '{prompt_name}' in the CSV file."
            )

    # Extract unique model names
    model_names = df["model_name"].unique()

    # Define the categories and their classes
    categories = {
        "Damage Severity": {
            "column": "damage_severity",
            "classes": ["little_or_none", "mild", "severe"],
        },
        "Informative": {
            "column": "informative",
            "classes": ["not_informative", "informative"],
        },
        "Humanitarian": {
            "column": "humanitarian",
            "classes": [
                "affected_injured_or_dead_people",
                "infrastructure_and_utility_damage",
                "not_humanitarian",
                "rescue_volunteering_or_donation_effort",
            ],
        },
        "Disaster Types": {
            "column": "disaster_types",
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
    accuracy_results = {}
    f1_results = {}
    all_accuracies = {model: [] for model in model_names}
    all_f1s = {model: [] for model in model_names}

    # Calculate metrics for each model and category
    for model in model_names:
        model_df = df[df["model_name"] == model]

        if model not in accuracy_results:
            accuracy_results[model] = {}
            f1_results[model] = {}

        for category_name, category_info in categories.items():
            column = category_info["column"]

            if category_name not in accuracy_results[model]:
                accuracy_results[model][category_name] = {}
                f1_results[model][category_name] = {}

            # Prepare data for metric calculation
            truth_col = column
            pred_col = f"{column}_prediction"

            # Skip N/A predictions
            valid_preds = model_df[pred_col] != "N/A"
            valid_df = model_df[valid_preds]

            if len(valid_df) > 0:
                # Convert data to consistent string type to avoid comparison issues
                y_true = valid_df[truth_col].astype(str).values
                y_pred = valid_df[pred_col].astype(str).values

                # Calculate accuracy (string comparison works fine for this)
                accuracy = (y_true == y_pred).mean() * 100
                accuracy_results[model][category_name]["Overall"] = accuracy
                all_accuracies[model].append(accuracy)

                # Calculate per-class metrics
                class_f1_values = []
                for class_name in category_info["classes"]:
                    # Get rows where the true label is this class
                    class_mask = y_true == class_name

                    if np.any(class_mask):
                        # Calculate class accuracy
                        class_accuracy = (y_pred[class_mask] == class_name).mean() * 100
                        accuracy_results[model][category_name][
                            class_name
                        ] = class_accuracy

                        # Calculate class F1 score
                        try:
                            # Convert to binary classification for this class
                            binary_true = (y_true == class_name).astype(int)
                            binary_pred = (y_pred == class_name).astype(int)

                            # Calculate F1 for this class
                            class_f1 = (
                                f1_score(binary_true, binary_pred, zero_division=0)
                                * 100
                            )
                            f1_results[model][category_name][class_name] = class_f1
                            class_f1_values.append(class_f1)
                        except Exception as e:
                            print(
                                f"Error calculating F1 for class {class_name}, model {model}: {str(e)}"
                            )
                            f1_results[model][category_name][class_name] = 0.0
                    else:
                        # No examples of this class in the dataset
                        accuracy_results[model][category_name][class_name] = float(
                            "nan"
                        )
                        f1_results[model][category_name][class_name] = float("nan")
                
                # FIXED: Calculate task F1 score as the true macro average of class F1 scores
                if class_f1_values:
                    macro_f1 = np.mean(class_f1_values)
                    f1_results[model][category_name]["Overall"] = macro_f1
                    all_f1s[model].append(macro_f1)
                else:
                    f1_results[model][category_name]["Overall"] = 0.0
                    all_f1s[model].append(0.0)
            else:
                # No valid predictions
                accuracy_results[model][category_name]["Overall"] = 0.0
                f1_results[model][category_name]["Overall"] = 0.0
                all_accuracies[model].append(0.0)
                all_f1s[model].append(0.0)

                for class_name in category_info["classes"]:
                    accuracy_results[model][category_name][class_name] = float("nan")
                    f1_results[model][category_name][class_name] = float("nan")

    # Calculate overall metrics for each model
    overall_accuracy = {
        model: np.nanmean(all_accuracies[model]) for model in model_names
    }
    overall_f1 = {model: np.nanmean(all_f1s[model]) for model in model_names}

    # Format model names for display
    formatted_model_names = [
        " ".join(word.capitalize() for word in model.split("_"))
        for model in model_names
    ]

    # Generate HTML table
    html = '<table class="styled-table" style="border-collapse: collapse; width: 100%; border: 1px solid #ddd;">'

    # Add title row if a prompt was specified
    if prompt_name:
        formatted_prompt_name = " ".join(
            word.capitalize() for word in prompt_name.split("_")
        )
        html += f'<tr><th colspan="{len(model_names) * 2 + 1}" style="padding: 12px; background-color: #222; color: white; text-align: center; font-size: 1.2em;">Results for Prompt: {formatted_prompt_name}</th></tr>'

    # Table header with model names (each model gets two columns - Acc and F1)
    html += '<tr><th style="padding: 8px; background-color: #333; color: white; text-align: left; width: 30%;">Category / Class</th>'

    for i, model_name in enumerate(formatted_model_names):
        # Add border style for columns except the last one
        border_style = (
            " border-right: 2px solid #555;"
            if i < len(formatted_model_names) - 1
            else ""
        )
        html += f'<th colspan="2" style="padding: 8px; background-color: #333; color: white; text-align: center;{border_style}">{model_name}</th>'
    html += "</tr>"

    # Sub-header for metrics
    html += '<tr><th style="padding: 8px; background-color: #333; color: white; text-align: left;"></th>'

    for i, _ in enumerate(model_names):
        # First column (Acc)
        html += f'<th style="padding: 8px; background-color: #444; color: white; text-align: center;">Acc (%)</th>'

        # Second column (F1)
        # Add border style for F1 columns except the last one
        border_style = (
            " border-right: 2px solid #555;" if i < len(model_names) - 1 else ""
        )
        html += f'<th style="padding: 8px; background-color: #444; color: white; text-align: center;{border_style}">F1 (%)</th>'

    html += "</tr>"

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

        # Find best performer for this category
        acc_values = {
            model: accuracy_results[model][category_name].get("Overall", 0.0)
            for model in model_names
        }
        f1_values = {
            model: f1_results[model][category_name].get("Overall", 0.0)
            for model in model_names
        }
        best_acc = max(acc_values.values())
        best_f1 = max(f1_values.values())

        for i, model in enumerate(model_names):
            # Accuracy value
            acc = accuracy_results[model][category_name].get("Overall", 0.0)
            is_best_acc = acc == best_acc and not pd.isna(acc) and acc > 0
            acc_color = "#4CAF50" if is_best_acc else "white"

            html += f'<td style="padding: 8px; text-align: center; color: {acc_color};">{format_metric(acc)}</td>'

            # F1 value
            f1 = f1_results[model][category_name].get("Overall", 0.0)
            is_best_f1 = f1 == best_f1 and not pd.isna(f1) and f1 > 0
            f1_color = "#4CAF50" if is_best_f1 else "white"

            # Add border style for F1 columns except the last one
            border_style = (
                " border-right: 2px solid #555;" if i < len(model_names) - 1 else ""
            )
            html += f'<td style="padding: 8px; text-align: center; color: {f1_color};{border_style}">{format_metric(f1)}</td>'

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

            # Find best performer for this class
            class_acc_values = {
                model: accuracy_results[model][category_name].get(
                    class_name, float("nan")
                )
                for model in model_names
            }
            class_f1_values = {
                model: f1_results[model][category_name].get(class_name, float("nan"))
                for model in model_names
            }

            valid_acc_values = [v for v in class_acc_values.values() if not pd.isna(v)]
            valid_f1_values = [v for v in class_f1_values.values() if not pd.isna(v)]

            best_class_acc = max(valid_acc_values) if valid_acc_values else None
            best_class_f1 = max(valid_f1_values) if valid_f1_values else None

            for i, model in enumerate(model_names):
                # Class accuracy
                class_acc = accuracy_results[model][category_name].get(
                    class_name, float("nan")
                )
                is_best_acc = (
                    not pd.isna(class_acc)
                    and class_acc == best_class_acc
                    and class_acc > 0
                )
                acc_color = "#4CAF50" if is_best_acc else "white"

                html += f'<td style="padding: 8px; text-align: center; color: {acc_color};">{format_metric(class_acc)}</td>'

                # Class F1
                class_f1 = f1_results[model][category_name].get(
                    class_name, float("nan")
                )
                is_best_f1 = (
                    not pd.isna(class_f1) and class_f1 == best_class_f1 and class_f1 > 0
                )
                f1_color = "#4CAF50" if is_best_f1 else "white"

                # Add border style for F1 columns except the last one
                border_style = (
                    " border-right: 2px solid #555;" if i < len(model_names) - 1 else ""
                )
                html += f'<td style="padding: 8px; text-align: center; color: {f1_color};{border_style}">{format_metric(class_f1)}</td>'

            html += "</tr>"

        row_count += 1

    # Add overall total row
    html += f'<tr style="background-color: #222; font-weight: bold;">'
    html += f'<td style="padding: 8px; color: white; text-align: left;">OVERALL</td>'

    # Find best overall model
    best_overall_acc = max(overall_accuracy.values())
    best_overall_f1 = max(overall_f1.values())

    for i, model in enumerate(model_names):
        # Overall accuracy
        acc_value = overall_accuracy[model]
        is_best_acc = acc_value == best_overall_acc
        acc_color = "#4CAF50" if is_best_acc else "white"

        html += f'<td style="padding: 8px; text-align: center; color: {acc_color};">{format_metric(acc_value)}</td>'

        # Overall F1
        f1_value = overall_f1[model]
        is_best_f1 = f1_value == best_overall_f1
        f1_color = "#4CAF50" if is_best_f1 else "white"

        # Add border style for F1 columns except the last one
        border_style = (
            " border-right: 2px solid #555;" if i < len(model_names) - 1 else ""
        )
        html += f'<td style="padding: 8px; text-align: center; color: {f1_color};{border_style}">{format_metric(f1_value)}</td>'

    html += "</tr>"

    html += "</table>"

    # Display the table
    display(HTML(html))


def analyse_prompt_performance(results_csv_filename: str, model_name: str = None):
    """
    Analyzes the performance of each prompt from a results CSV file for a specific model
    and displays a styled comparison table with both accuracy and F1 scores.
    """
    import os
    import pandas as pd
    import numpy as np
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import LabelEncoder
    import re
    from IPython.display import display, HTML

    # Construct the full path to the results CSV
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    results_path = os.path.join(results_dir, results_csv_filename)

    # Read the results CSV
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found at: {results_path}")

    df = pd.read_csv(results_path)

    # Filter by model if specified
    if model_name is not None:
        if "model_name" in df.columns:
            df = df[df["model_name"] == model_name]
            if df.empty:
                raise ValueError(
                    f"No data found for model '{model_name}' in the CSV file."
                )
        else:
            print(f"Warning: No 'model_name' column found in CSV. Using all data.")

    # Handle prompt column name
    # [existing prompt column handling code...]

    # Extract unique prompt names
    prompt_names = df["prompt_name"].unique()

    # Define the categories and their classes
    categories = {
        "Damage Severity": {
            "column": "damage_severity",
            "classes": ["little_or_none", "mild", "severe"],
        },
        "Informative": {
            "column": "informative",
            "classes": ["not_informative", "informative"],
        },
        "Humanitarian": {
            "column": "humanitarian",
            "classes": [
                "affected_injured_or_dead_people",
                "infrastructure_and_utility_damage",
                "not_humanitarian",
                "rescue_volunteering_or_donation_effort",
            ],
        },
        "Disaster Types": {
            "column": "disaster_types",
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
    accuracy_results = {}
    f1_results = {}
    all_accuracies = {prompt: [] for prompt in prompt_names}
    all_f1s = {prompt: [] for prompt in prompt_names}

    # Calculate metrics for each prompt and category
    for prompt in prompt_names:
        prompt_df = df[df["prompt_name"] == prompt]

        if prompt not in accuracy_results:
            accuracy_results[prompt] = {}
            f1_results[prompt] = {}

        for category_name, category_info in categories.items():
            column = category_info["column"]

            if category_name not in accuracy_results[prompt]:
                accuracy_results[prompt][category_name] = {}
                f1_results[prompt][category_name] = {}

            # Prepare data for metric calculation
            truth_col = column
            pred_col = f"{column}_prediction"

            # Skip N/A predictions
            valid_preds = prompt_df[pred_col] != "N/A"
            valid_df = prompt_df[valid_preds]

            if len(valid_df) > 0:
                # Convert data to consistent string type to avoid comparison issues
                y_true = valid_df[truth_col].astype(str).values
                y_pred = valid_df[pred_col].astype(str).values

                # Calculate accuracy (string comparison works fine for this)
                accuracy = (y_true == y_pred).mean() * 100
                accuracy_results[prompt][category_name]["Overall"] = accuracy
                all_accuracies[prompt].append(accuracy)

                # Calculate per-class metrics
                class_f1_values = []
                for class_name in category_info["classes"]:
                    # Get rows where the true label is this class
                    class_mask = y_true == class_name

                    if np.any(class_mask):
                        # Calculate class accuracy
                        class_accuracy = (y_pred[class_mask] == class_name).mean() * 100
                        accuracy_results[prompt][category_name][
                            class_name
                        ] = class_accuracy

                        # Calculate class F1 score
                        try:
                            # Convert to binary classification for this class
                            binary_true = (y_true == class_name).astype(int)
                            binary_pred = (y_pred == class_name).astype(int)

                            # Calculate F1 for this class
                            class_f1 = (
                                f1_score(binary_true, binary_pred, zero_division=0)
                                * 100
                            )
                            f1_results[prompt][category_name][class_name] = class_f1
                            class_f1_values.append(class_f1)
                        except Exception as e:
                            print(
                                f"Error calculating F1 for class {class_name}, prompt {prompt}: {str(e)}"
                            )
                            f1_results[prompt][category_name][class_name] = 0.0
                    else:
                        # No examples of this class in the dataset
                        accuracy_results[prompt][category_name][class_name] = float(
                            "nan"
                        )
                        f1_results[prompt][category_name][class_name] = float("nan")
                
                # FIXED: Calculate task F1 score as the true macro average of class F1 scores
                if class_f1_values:
                    macro_f1 = np.mean(class_f1_values)
                    f1_results[prompt][category_name]["Overall"] = macro_f1
                    all_f1s[prompt].append(macro_f1)
                else:
                    f1_results[prompt][category_name]["Overall"] = 0.0
                    all_f1s[prompt].append(0.0)
            else:
                # No valid predictions
                accuracy_results[prompt][category_name]["Overall"] = 0.0
                f1_results[prompt][category_name]["Overall"] = 0.0
                all_accuracies[prompt].append(0.0)
                all_f1s[prompt].append(0.0)

                for class_name in category_info["classes"]:
                    accuracy_results[prompt][category_name][class_name] = float("nan")
                    f1_results[prompt][category_name][class_name] = float("nan")

    # Calculate overall metrics for each prompt
    overall_accuracy = {
        prompt: np.nanmean(all_accuracies[prompt]) for prompt in prompt_names
    }
    overall_f1 = {prompt: np.nanmean(all_f1s[prompt]) for prompt in prompt_names}

    # Format prompt names for display
    formatted_prompt_names = [
        " ".join(word.capitalize() for word in prompt.split("_"))
        for prompt in prompt_names
    ]

    # Generate HTML table
    html = '<table class="styled-table" style="border-collapse: collapse; width: 100%; border: 1px solid #ddd;">'

    # Add title row if a model was specified
    if model_name:
        formatted_model_name = " ".join(
            word.capitalize() for word in model_name.split("_")
        )
        html += f'<tr><th colspan="{len(prompt_names) * 2 + 1}" style="padding: 12px; background-color: #222; color: white; text-align: center; font-size: 1.2em;">Results for Model: {formatted_model_name}</th></tr>'

    # Table header with prompt names (each prompt gets two columns - Acc and F1)
    html += '<tr><th style="padding: 8px; background-color: #333; color: white; text-align: left; width: 30%;">Category / Class</th>'

    for i, prompt_name in enumerate(formatted_prompt_names):
        # Add border style for columns except the last one
        border_style = (
            " border-right: 2px solid #555;"
            if i < len(formatted_prompt_names) - 1
            else ""
        )
        html += f'<th colspan="2" style="padding: 8px; background-color: #333; color: white; text-align: center;{border_style}">{prompt_name}</th>'
    html += "</tr>"

    # Sub-header for metrics
    html += '<tr><th style="padding: 8px; background-color: #333; color: white; text-align: left;"></th>'

    for i, _ in enumerate(prompt_names):
        # First column (Acc)
        html += f'<th style="padding: 8px; background-color: #444; color: white; text-align: center;">Acc (%)</th>'

        # Second column (F1)
        # Add border style for F1 columns except the last one
        border_style = (
            " border-right: 2px solid #555;" if i < len(prompt_names) - 1 else ""
        )
        html += f'<th style="padding: 8px; background-color: #444; color: white; text-align: center;{border_style}">F1 (%)</th>'

    html += "</tr>"

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

        # Find best performer for this category
        acc_values = {
            prompt: accuracy_results[prompt][category_name].get("Overall", 0.0)
            for prompt in prompt_names
        }
        f1_values = {
            prompt: f1_results[prompt][category_name].get("Overall", 0.0)
            for prompt in prompt_names
        }
        best_acc = max(acc_values.values())
        best_f1 = max(f1_values.values())

        for i, prompt in enumerate(prompt_names):
            # Accuracy value
            acc = accuracy_results[prompt][category_name].get("Overall", 0.0)
            is_best_acc = acc == best_acc and not pd.isna(acc) and acc > 0
            acc_color = "#4CAF50" if is_best_acc else "white"

            html += f'<td style="padding: 8px; text-align: center; color: {acc_color};">{format_metric(acc)}</td>'

            # F1 value
            f1 = f1_results[prompt][category_name].get("Overall", 0.0)
            is_best_f1 = f1 == best_f1 and not pd.isna(f1) and f1 > 0
            f1_color = "#4CAF50" if is_best_f1 else "white"

            # Add border style for F1 columns except the last one
            border_style = (
                " border-right: 2px solid #555;" if i < len(prompt_names) - 1 else ""
            )
            html += f'<td style="padding: 8px; text-align: center; color: {f1_color};{border_style}">{format_metric(f1)}</td>'

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

            # Find best performer for this class
            class_acc_values = {
                prompt: accuracy_results[prompt][category_name].get(
                    class_name, float("nan")
                )
                for prompt in prompt_names
            }
            class_f1_values = {
                prompt: f1_results[prompt][category_name].get(class_name, float("nan"))
                for prompt in prompt_names
            }

            valid_acc_values = [v for v in class_acc_values.values() if not pd.isna(v)]
            valid_f1_values = [v for v in class_f1_values.values() if not pd.isna(v)]

            best_class_acc = max(valid_acc_values) if valid_acc_values else None
            best_class_f1 = max(valid_f1_values) if valid_f1_values else None

            for i, prompt in enumerate(prompt_names):
                # Class accuracy
                class_acc = accuracy_results[prompt][category_name].get(
                    class_name, float("nan")
                )
                is_best_acc = (
                    not pd.isna(class_acc)
                    and class_acc == best_class_acc
                    and class_acc > 0
                )
                acc_color = "#4CAF50" if is_best_acc else "white"

                html += f'<td style="padding: 8px; text-align: center; color: {acc_color};">{format_metric(class_acc)}</td>'

                # Class F1
                class_f1 = f1_results[prompt][category_name].get(
                    class_name, float("nan")
                )
                is_best_f1 = (
                    not pd.isna(class_f1) and class_f1 == best_class_f1 and class_f1 > 0
                )
                f1_color = "#4CAF50" if is_best_f1 else "white"

                # Add border style for F1 columns except the last one
                border_style = (
                    " border-right: 2px solid #555;"
                    if i < len(prompt_names) - 1
                    else ""
                )
                html += f'<td style="padding: 8px; text-align: center; color: {f1_color};{border_style}">{format_metric(class_f1)}</td>'

            html += "</tr>"

        row_count += 1

    # Add overall total row
    html += f'<tr style="background-color: #222; font-weight: bold;">'
    html += f'<td style="padding: 8px; color: white; text-align: left;">OVERALL</td>'

    # Find best overall prompt
    best_overall_acc = max(overall_accuracy.values())
    best_overall_f1 = max(overall_f1.values())

    for i, prompt in enumerate(prompt_names):
        # Overall accuracy
        acc_value = overall_accuracy[prompt]
        is_best_acc = acc_value == best_overall_acc
        acc_color = "#4CAF50" if is_best_acc else "white"

        html += f'<td style="padding: 8px; text-align: center; color: {acc_color};">{format_metric(acc_value)}</td>'

        # Overall F1
        f1_value = overall_f1[prompt]
        is_best_f1 = f1_value == best_overall_f1
        f1_color = "#4CAF50" if is_best_f1 else "white"

        # Add border style for F1 columns except the last one
        border_style = (
            " border-right: 2px solid #555;" if i < len(prompt_names) - 1 else ""
        )
        html += f'<td style="padding: 8px; text-align: center; color: {f1_color};{border_style}">{format_metric(f1_value)}</td>'

    html += "</tr>"

    html += "</table>"

    # Display the table
    display(HTML(html))


def analyse_model_prompt_heatmaps(results_csv_filename: str):
    """
    Analyzes the performance of all models against all prompts and displays
    the results as compact heatmaps - one for overall accuracy and one for each task.
    Each heatmap is displayed individually for better readability.

    Args:
        results_csv_filename (str):
            Name of the CSV file with results, located in the src/zero_shot/testing/results directory.
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import display

    # Construct the full path to the results CSV
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    results_path = os.path.join(results_dir, results_csv_filename)

    # Read the results CSV
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found at: {results_path}")

    df = pd.read_csv(results_path)

    # Ensure both model_name and prompt_name columns exist
    required_cols = ["model_name", "prompt_name"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the CSV file.")

    # Extract unique model and prompt names
    model_names = df["model_name"].unique()
    prompt_names = df["prompt_name"].unique()

    # Format model and prompt names for display
    def format_name(name):
        return " ".join(word.capitalize() for word in name.split("_"))

    formatted_model_names = [format_name(model) for model in model_names]
    formatted_prompt_names = [format_name(prompt) for prompt in prompt_names]

    # Define the tasks
    tasks = {
        "Damage Severity": "damage_severity",
        "Informative": "informative",
        "Humanitarian": "humanitarian",
        "Disaster Types": "disaster_types",
    }

    # Create accuracy matrices - one for overall and one for each task
    overall_accuracies = np.zeros((len(model_names), len(prompt_names)))
    task_accuracies = {
        task: np.zeros((len(model_names), len(prompt_names))) for task in tasks
    }

    # Calculate accuracies for each model-prompt combination
    for i, model in enumerate(model_names):
        for j, prompt in enumerate(prompt_names):
            # Filter data for this model-prompt combination
            combo_df = df[(df["model_name"] == model) & (df["prompt_name"] == prompt)]

            if not combo_df.empty:
                # Calculate accuracy for each task
                task_acc_values = []
                for task_name, task_col in tasks.items():
                    truth_col = task_col
                    pred_col = f"{task_col}_prediction"

                    # Skip N/A predictions
                    valid_preds = combo_df[pred_col] != "N/A"
                    if valid_preds.sum() > 0:
                        task_accuracy = (
                            combo_df[valid_preds][truth_col]
                            == combo_df[valid_preds][pred_col]
                        ).mean() * 100  # Convert to percentage
                        task_accuracies[task_name][i, j] = task_accuracy
                        task_acc_values.append(task_accuracy)
                    else:
                        task_accuracies[task_name][i, j] = np.nan

                # Calculate overall accuracy (average of task accuracies)
                if task_acc_values:
                    overall_accuracies[i, j] = np.nanmean(task_acc_values)
                else:
                    overall_accuracies[i, j] = np.nan

    # Function to create and display a single heatmap
    def create_heatmap(data, title, model_labels, prompt_labels):
        # Smaller figure size
        plt.figure(figsize=(8, 5))
        ax = plt.gca()

        # Create the heatmap without color bar
        sns.heatmap(
            data,
            annot=True,
            fmt=".1f",
            cmap=sns.diverging_palette(10, 240, s=100, l=40, n=55, sep=1),
            xticklabels=prompt_labels,
            yticklabels=model_labels,
            ax=ax,
            cbar=False,  # No color bar
            annot_kws={"fontsize": 8},
            vmin=50,
            vmax=100,
        )

        # Set title and labels
        ax.set_title(title, fontsize=8, fontweight="bold", pad=10)
        ax.set_xlabel("Prompt", fontsize=8, labelpad=10)
        ax.set_ylabel("Model", fontsize=8, labelpad=10)

        # Adjust tick labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

        # Add grid lines
        for _, spine in ax.spines.items():
            spine.set_visible(True)

        # Improve spacing
        plt.tight_layout()
        plt.show()

    # Create and display each heatmap individually

    create_heatmap(
        overall_accuracies,
        "Overall Accuracy (%)",
        formatted_model_names,
        formatted_prompt_names,
    )

    # Create individual heatmaps for each task
    for task_name, task_data in task_accuracies.items():

        create_heatmap(
            task_data,
            f"{task_name} Accuracy (%)",
            formatted_model_names,
            formatted_prompt_names,
        )

    # Return the raw data for further analysis if needed
    return {
        "overall": overall_accuracies,
        "tasks": task_accuracies,
        "models": model_names,
        "prompts": prompt_names,
    }


def llm_conf_matrices(results_csv_filename: str, model_name: str, save_dir: str = None):
    """
    Generates and displays confusion matrices for all prompts used with a specific LLM.
    Creates a separate set of four confusion matrices (one for each task) for each prompt.

    Args:
        results_csv_filename (str):
            Name of the CSV file with results, located in the src/zero_shot/testing/results directory.
        model_name (str):
            The name of the LLM model to analyze (e.g., "gpt4v", "claude_sonnet").
        save_dir (str, optional):
            Directory to save the generated confusion matrix plots. If None, plots are only displayed.
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    from collections import defaultdict

    # Construct the full path to the results CSV
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    results_path = os.path.join(results_dir, results_csv_filename)

    # Read the results CSV
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found at: {results_path}")

    df = pd.read_csv(results_path)

    # Filter for the specified model
    model_df = df[df["model_name"] == model_name]
    if model_df.empty:
        raise ValueError(f"No data found for model '{model_name}' in the CSV file.")

    # Get unique prompts for this model
    prompts = model_df["prompt_name"].unique()
    if len(prompts) == 0:
        raise ValueError(f"No prompts found for model '{model_name}' in the CSV file.")

    # Task mapping configs
    tasks = ["disaster_types", "informative", "humanitarian", "damage_severity"]

    task_names = {
        "disaster_types": ["quake", "fire", "flood", "hurr.", "land.", "none", "other"],
        "informative": ["not inf", "inf"],
        "humanitarian": ["injured", "infra", "not hum", "rescue"],
        "damage_severity": ["none", "mild", "severe"],
    }

    # Dictionary to convert task names to proper case titles
    task_titles = {
        "disaster_types": "Disaster Types",
        "informative": "Informative",
        "humanitarian": "Humanitarian",
        "damage_severity": "Damage Severity",
    }

    # Process each prompt
    for prompt in prompts:
        print(f"\nGenerating confusion matrices for {model_name} with prompt: {prompt}")

        # Filter data for this prompt
        prompt_df = model_df[model_df["prompt_name"] == prompt]

        # Create dictionaries to store true labels and predictions for each task
        true_labels_dict = {}
        predictions_dict = {}

        # Extract true labels and predictions for each task
        for task in tasks:
            # Skip N/A predictions and ensure we have both true and prediction values
            valid_preds = (
                (prompt_df[f"{task}_prediction"] != "N/A")
                & (~prompt_df[task].isna())
                & (~prompt_df[f"{task}_prediction"].isna())
            )
            valid_df = prompt_df[valid_preds]

            if len(valid_df) > 0:
                # Get true labels and predictions - ensure both are strings
                true_labels_dict[task] = valid_df[task].astype(str).values
                predictions_dict[task] = (
                    valid_df[f"{task}_prediction"].astype(str).values
                )

                # Debug check for empty or problematic values
                if len(true_labels_dict[task]) == 0 or len(predictions_dict[task]) == 0:
                    print(f"Warning: Empty arrays for task {task} with prompt {prompt}")
                    true_labels_dict[task] = []
                    predictions_dict[task] = []
            else:
                print(
                    f"Warning: No valid predictions for task {task} with prompt {prompt}"
                )
                true_labels_dict[task] = []
                predictions_dict[task] = []

        # Skip this prompt if any task has no valid predictions
        if any(len(true_labels_dict[task]) == 0 for task in tasks):
            print(f"Skipping prompt {prompt} due to missing valid predictions")
            continue

        # Set up the plotting style
        plt.style.use("default")
        sns.set_theme()

        # Create save directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Create figure with four subplots in a row
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(
            f"Confusion Matrices: {model_name} with {prompt}", fontsize=16, y=1.05
        )

        for idx, (task, ax) in enumerate(zip(tasks, axes)):
            if len(true_labels_dict[task]) > 0:
                try:
                    # Compute confusion matrix
                    cm = confusion_matrix(
                        true_labels_dict[task], predictions_dict[task]
                    )
                except Exception as e:
                    print(
                        f"Error creating confusion matrix for task {task} with prompt {prompt}: {str(e)}"
                    )
                    print(
                        f"Types - true: {type(true_labels_dict[task][0])}, pred: {type(predictions_dict[task][0])}"
                    )
                    print(
                        f"Sample values - true: {true_labels_dict[task][:3]}, pred: {predictions_dict[task][:3]}"
                    )
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
                    ax.set_title(task_titles[task])
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
                    xticklabels=task_names[task],
                    yticklabels=task_names[task],
                    ax=ax,
                    cbar=False,
                    annot_kws={"fontsize": 12},
                    vmin=-0.5,
                    center=0,
                    vmax=1.0,
                )

                # Set titles and labels
                ax.set_title(task_titles[task])
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
                ax.set_title(task_titles[task])

        # Adjust layout
        plt.tight_layout(w_pad=1.5, h_pad=1.5)

        # Save if save_dir is specified
        if save_dir:
            filename = f"{model_name}_{prompt}_confusion_matrices.png"
            plt.savefig(
                os.path.join(save_dir, filename),
                bbox_inches="tight",
                dpi=300,
            )
            print(f"Saved to {os.path.join(save_dir, filename)}")

        # Display the plot
        plt.show()
        plt.close()

    print(
        f"\nCompleted confusion matrix generation for model {model_name} across {len(prompts)} prompts"
    )


def test_prompt_significance(results_csv_filename, model_name=None, prompt_name=None):
    """
    Performs McNemar's test to determine if the differences between prompt/model strategies
    are statistically significant.

    Args:
        results_csv_filename (str): Name of the CSV file with results
        model_name (str, optional): Filter for a specific model
        prompt_name (str, optional): Filter for a specific prompt

    Returns:
        Dictionary with p-values for each task and pair
    """
    import os
    import pandas as pd
    import numpy as np
    from itertools import combinations
    from statsmodels.stats.contingency_tables import mcnemar

    # Load data
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    results_path = os.path.join(results_dir, results_csv_filename)
    df = pd.read_csv(results_path)

    # Determine whether we're comparing models or prompts
    comparing_models = prompt_name is not None

    # Filter by model or prompt if specified
    if model_name is not None:
        df = df[df["model_name"] == model_name]
    elif prompt_name is not None:
        df = df[df["prompt_name"] == prompt_name]

    # Get unique prompts or models
    if comparing_models:
        items = df["model_name"].unique()
    else:
        items = df["prompt_name"].unique()

    # Define tasks
    tasks = ["damage_severity", "informative", "humanitarian", "disaster_types"]

    # Dictionary to store results
    significance_results = {task: {} for task in tasks}

    # For each task
    for task in tasks:

        # For each pair of items (prompts or models)
        for item1, item2 in combinations(items, 2):
            # Initialize comparison name before any potential errors
            comparison_name = f"{item1} vs {item2}"

            # Create contingency table
            contingency = np.zeros((2, 2), dtype=int)

            # Get unique images
            image_paths = df["image_path"].unique()

            for img_path in image_paths:
                try:
                    # Get predictions for both items
                    if comparing_models:
                        item1_rows = df[
                            (df["model_name"] == item1) & (df["image_path"] == img_path)
                        ]
                        item2_rows = df[
                            (df["model_name"] == item2) & (df["image_path"] == img_path)
                        ]
                    else:
                        item1_rows = df[
                            (df["prompt_name"] == item1)
                            & (df["image_path"] == img_path)
                        ]
                        item2_rows = df[
                            (df["prompt_name"] == item2)
                            & (df["image_path"] == img_path)
                        ]

                    if len(item1_rows) == 0 or len(item2_rows) == 0:
                        continue

                    pred1 = item1_rows[f"{task}_prediction"].iloc[0]
                    pred2 = item2_rows[f"{task}_prediction"].iloc[0]
                    true_label = df[df["image_path"] == img_path][task].iloc[0]

                    # Skip N/A predictions
                    if pred1 == "N/A" or pred2 == "N/A" or pd.isna(true_label):
                        continue

                    # Compare with true label
                    correct1 = pred1 == true_label
                    correct2 = pred2 == true_label

                    # Update contingency table
                    if correct1 and correct2:  # Both correct
                        contingency[0, 0] += 1
                    elif correct1 and not correct2:  # Only item1 correct
                        contingency[0, 1] += 1
                    elif not correct1 and correct2:  # Only item2 correct
                        contingency[1, 0] += 1
                    else:  # Both wrong
                        contingency[1, 1] += 1
                except (IndexError, KeyError) as e:
                    continue

            # Perform McNemar's test
            if (
                contingency[0, 1] + contingency[1, 0] > 0
            ):  # Only if we have disagreements
                try:
                    result = mcnemar(contingency, exact=False, correction=True)
                    p_value = result.pvalue

                    # Store result
                    significance_results[task][comparison_name] = {
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                        "contingency": contingency.tolist(),
                    }

                except Exception as e:
                    significance_results[task][comparison_name] = {"error": str(e)}
            else:
                significance_results[task][comparison_name] = {
                    "error": "Not enough disagreements"
                }

    return significance_results


def bootstrap_confidence_intervals(results_csv_filename, model_name=None, prompt_name=None, n_bootstrap=1000, alpha=0.05):
    """
    Computes bootstrap confidence intervals for each model/prompt and task.
    
    Args:
        results_csv_filename (str): Name of the CSV file with results
        model_name (str, optional): Filter for a specific model
        prompt_name (str, optional): Filter for a specific prompt
        n_bootstrap (int): Number of bootstrap samples
        alpha (float): Significance level (e.g., 0.05 for 95% confidence)
        
    Returns:
        Dictionary with confidence intervals for each task and model/prompt
    """
    import os
    import pandas as pd
    import numpy as np
    import random
    
    # Load data
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    results_path = os.path.join(results_dir, results_csv_filename)
    df = pd.read_csv(results_path)
    
    # Determine whether we're comparing models or prompts
    comparing_models = prompt_name is not None
    
    # Filter by model or prompt if specified
    if model_name is not None:
        df = df[df["model_name"] == model_name]
    elif prompt_name is not None:
        df = df[df["prompt_name"] == prompt_name]
    
    # Get unique prompts or models
    if comparing_models:
        items = df["model_name"].unique()
    else:
        items = df["prompt_name"].unique()
    
    # Define tasks
    tasks = ["damage_severity", "informative", "humanitarian", "disaster_types"]
    
    # Dictionary to store results
    bootstrap_results = {task: {item: {} for item in items} for task in tasks}
    
    # For each task and item
    for task in tasks:
        for item in items:
            # Get data for this item
            if comparing_models:
                item_df = df[df["model_name"] == item].copy()
            else:
                item_df = df[df["prompt_name"] == item].copy()
            
            # Skip if no data
            if len(item_df) == 0:
                continue
                
            # Filter to valid predictions (not N/A)
            valid_mask = (item_df[f"{task}_prediction"] != "N/A") & (~item_df[task].isna())
            valid_df = item_df[valid_mask]
            
            # Skip if no valid predictions
            if len(valid_df) == 0:
                continue
            
            try:
                # Perform bootstrap sampling
                accuracies = []
                # Use fewer bootstrap samples for faster analysis
                for _ in range(n_bootstrap):
                    # Sample with replacement - use indices to avoid IndexError
                    indices = np.random.choice(valid_df.index, size=len(valid_df), replace=True)
                    bootstrap_sample = valid_df.loc[indices]
                    
                    # Calculate accuracy for this sample
                    accuracy = (bootstrap_sample[task] == bootstrap_sample[f"{task}_prediction"]).mean() * 100
                    accuracies.append(accuracy)
                
                # Calculate confidence intervals
                lower = np.percentile(accuracies, alpha/2 * 100)
                upper = np.percentile(accuracies, (1 - alpha/2) * 100)
                
                # Store results
                bootstrap_results[task][item] = {
                    "mean": np.mean(accuracies),
                    "lower": lower,
                    "upper": upper
                }
                
            except Exception as e:
                pass
    
    return bootstrap_results


def plot_significance_comparison(bootstrap_results):
    """
    Creates a plot showing confidence intervals for each prompt and task.

    Args:
        bootstrap_results: Output from bootstrap_confidence_intervals function
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set up plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Mapping of tasks to nice titles
    task_titles = {
        "damage_severity": "Damage Severity",
        "informative": "Informative",
        "humanitarian": "Humanitarian",
        "disaster_types": "Disaster Types",
    }

    # Plot each task
    for i, (task, ax) in enumerate(zip(bootstrap_results.keys(), axes)):
        # Extract data for this task
        prompts = []
        means = []
        lowers = []
        uppers = []

        for prompt in bootstrap_results[task]:
            result = bootstrap_results[task][prompt]
            if result:  # Check if we have valid results
                prompts.append(prompt.replace("_", " ").title())
                means.append(result["mean"])
                lowers.append(result["lower"])
                uppers.append(result["upper"])

        # Calculate error bars
        errors_low = [mean - lower for mean, lower in zip(means, lowers)]
        errors_high = [upper - mean for mean, upper in zip(means, uppers)]
        errors = [errors_low, errors_high]

        # Create plot
        bars = ax.bar(prompts, means, yerr=errors, capsize=5, alpha=0.7)

        # Add value labels on top of bars
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{mean:.1f}%",
                ha="center",
                va="bottom",
            )

        # Set title and labels
        ax.set_title(task_titles[task])
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(50, 100)  # Assuming accuracy is in 0-100 range
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.show()


def complete_statistical_analysis(
    results_csv_filename, model_name=None, prompt_name=None
):
    """
    Performs complete statistical analysis of either model or prompt performance.

    Args:
        results_csv_filename (str): Name of the CSV file with results
        model_name (str, optional): Filter for a specific model
        prompt_name (str, optional): Filter for a specific prompt

    Note:
        Specify either model_name or prompt_name, not both.
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import display, HTML
    import builtins

    # Check that we're not filtering by both model and prompt
    if model_name is not None and prompt_name is not None:
        raise ValueError("Please specify either model_name OR prompt_name, not both.")

    # Temporarily suppress detailed output
    original_print = builtins.print
    builtins.print = lambda *args, **kwargs: None

    try:
        # Determine if we're comparing models or prompts
        comparing_models = prompt_name is not None
        filter_column = "prompt_name" if comparing_models else "model_name"
        filter_value = prompt_name if comparing_models else model_name

        # Run the analysis quietly
        significance_results = test_prompt_significance(
            results_csv_filename,
            model_name=None if comparing_models else model_name,
            prompt_name=prompt_name if comparing_models else None,
        )

        bootstrap_results = bootstrap_confidence_intervals(
            results_csv_filename,
            model_name=None if comparing_models else model_name,
            prompt_name=prompt_name if comparing_models else None,
            n_bootstrap=500,
        )

        # Restore print function
        builtins.print = original_print

        # Create formatted HTML tables with transparent backgrounds and light text
        html = """
        <h2 style="text-align:center; color:#FFFFFF; margin-bottom:20px;">"""

        if comparing_models:
            html += f"""Model Performance with Prompt: {prompt_name.replace('_', ' ').title()}"""
        else:
            html += f"""Prompt Performance{' with Model: ' + model_name.replace('_', ' ').title() if model_name else ''}"""

        html += """</h2>
        """

        # Performance table - transparent with light text
        html += """
        <h3 style="color:#FFFFFF; margin-top:30px; margin-bottom:15px;">Performance with 95% Confidence Intervals</h3>
        <table style="width:100%; border-collapse:collapse; margin-bottom:30px; border:1px solid #AAAAAA;">
        <tr>
            <th style="padding:10px; border:1px solid #AAAAAA; text-align:left; font-weight:bold; color:#FFFFFF;">Task</th>
            <th style="padding:10px; border:1px solid #AAAAAA; text-align:left; font-weight:bold; color:#FFFFFF;">"""

        html += "Model" if comparing_models else "Prompt"

        html += """</th>
            <th style="padding:10px; border:1px solid #AAAAAA; text-align:center; font-weight:bold; color:#FFFFFF;">Accuracy (%)</th>
            <th style="padding:10px; border:1px solid #AAAAAA; text-align:center; font-weight:bold; color:#FFFFFF;">95% CI</th>
        </tr>
        """

        # Add rows for each task and model/prompt
        tasks = list(bootstrap_results.keys())
        for task_idx, task in enumerate(tasks):
            # Find best performer for this task
            best_item = max(
                bootstrap_results[task].items(),
                key=lambda x: x[1].get("mean", 0) if x[1] else 0,
            )[0]

            # Format task name
            task_display = task.replace("_", " ").title()
            items = list(bootstrap_results[task].keys())

            for i, item in enumerate(items):
                result = bootstrap_results[task][item]
                if not result:
                    continue

                # Format data
                item_display = item.replace("_", " ").title()
                is_best = item == best_item
                highlight = (
                    "color:#00FFCC; font-weight:bold;" if is_best else "color:#FFFFFF;"
                )
                accuracy = result.get("mean", 0)
                lower = result.get("lower", 0)
                upper = result.get("upper", 0)

                html += f"""
                <tr>
                    <td style="padding:10px; border:1px solid #AAAAAA; color:#FFFFFF;">{task_display if i == 0 else ""}</td>
                    <td style="padding:10px; border:1px solid #AAAAAA; {highlight}">{item_display}</td>
                    <td style="padding:10px; border:1px solid #AAAAAA; text-align:center; {highlight}">{accuracy:.2f}</td>
                    <td style="padding:10px; border:1px solid #AAAAAA; text-align:center; {highlight}">[{lower:.2f}, {upper:.2f}]</td>
                </tr>
                """

        html += "</table>"

        # Table for significant differences - dark theme friendly
        html += """
        <h3 style="color:#FFFFFF; margin-top:30px; margin-bottom:15px;">Statistically Significant Differences (p<0.05)</h3>
        <table style="width:100%; border-collapse:collapse; border:1px solid #AAAAAA;">
        <tr>
            <th style="padding:10px; border:1px solid #AAAAAA; text-align:left; font-weight:bold; color:#FFFFFF;">Task</th>
            <th style="padding:10px; border:1px solid #AAAAAA; text-align:left; font-weight:bold; color:#FFFFFF;">Comparison</th>
            <th style="padding:10px; border:1px solid #AAAAAA; text-align:center; font-weight:bold; color:#FFFFFF;">P-value</th>
            <th style="padding:10px; border:1px solid #AAAAAA; text-align:center; font-weight:bold; color:#FFFFFF;">Superior """

        html += "Model" if comparing_models else "Prompt"

        html += """</th>
        </tr>
        """

        # Track if we found any significant differences
        any_significant = False

        for task_idx, task in enumerate(tasks):
            task_display = task.replace("_", " ").title()

            # Get significant comparisons
            significant_pairs = []
            for comp_name, comp_data in significance_results[task].items():
                if isinstance(comp_data, dict) and comp_data.get("significant", False):
                    significant_pairs.append((comp_name, comp_data))

            # Sort by p-value
            significant_pairs.sort(key=lambda x: x[1].get("p_value", 1))

            if significant_pairs:
                any_significant = True

                for i, (comparison, data) in enumerate(significant_pairs):
                    p_value = data.get("p_value", 0)

                    # Determine superior model/prompt
                    contingency = data.get("contingency", [[0, 0], [0, 0]])
                    item1, item2 = comparison.split(" vs ")
                    item1_wins = contingency[0][1]
                    item2_wins = contingency[1][0]

                    superior = item1 if item1_wins > item2_wins else item2
                    superior_display = superior.replace("_", " ").title()

                    html += f"""
                    <tr>
                        <td style="padding:10px; border:1px solid #AAAAAA; color:#FFFFFF;">{task_display if i == 0 else ""}</td>
                        <td style="padding:10px; border:1px solid #AAAAAA; color:#FFFFFF;">{comparison}</td>
                        <td style="padding:10px; border:1px solid #AAAAAA; text-align:center; color:#FFFFFF;">{p_value:.4f}</td>
                        <td style="padding:10px; border:1px solid #AAAAAA; text-align:center; font-weight:bold; color:#00FFCC;">{superior_display}</td>
                    </tr>
                    """

        if not any_significant:
            html += f"""
            <tr>
                <td colspan="4" style="padding:15px; text-align:center; border:1px solid #AAAAAA; color:#FFFFFF;">
                    No statistically significant differences found
                </td>
            </tr>
            """

        html += "</table>"

        # Display the tables
        display(HTML(html))

        # Create visualization with dark theme style
        plt.style.use("dark_background")
        plt.figure(figsize=(14, 10))
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # Define a bright color palette for dark background
        items = list(bootstrap_results[list(bootstrap_results.keys())[0]].keys())
        colors = sns.color_palette("bright", len(items))
        item_color_map = dict(zip(items, colors))

        for i, task in enumerate(tasks):
            ax = axes[i]
            task_display = task.replace("_", " ").title()

            # Extract data for plotting
            x_labels = []
            means = []
            lowers = []
            uppers = []
            colors_list = []

            # Find best performer
            best_mean = 0
            for item, result in bootstrap_results[task].items():
                if result and result.get("mean", 0) > best_mean:
                    best_mean = result.get("mean", 0)

            # Prepare data
            for item in items:
                result = bootstrap_results[task].get(item, {})
                if not result:
                    continue

                item_display = item.replace("_", " ").title()
                x_labels.append(item_display)

                mean = result.get("mean", 0)
                lower = result.get("lower", 0)
                upper = result.get("upper", 0)

                means.append(mean)
                lowers.append(lower)
                uppers.append(upper)

                # Highlight best performer
                colors_list.append(
                    "#00FFCC" if mean == best_mean else item_color_map[item]
                )

            # Calculate error bars
            errors_low = [m - l for m, l in zip(means, lowers)]
            errors_high = [u - m for m, u in zip(means, uppers)]

            # Create bar chart
            bars = ax.bar(
                x_labels,
                means,
                yerr=[errors_low, errors_high],
                color=colors_list,
                alpha=0.8,
                capsize=5,
            )

            # Add value labels
            for bar, mean in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean + 0.5,
                    f"{mean:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                )

            # Formatting
            ax.set_title(task_display, fontsize=14, fontweight="bold")
            ax.set_ylabel("Accuracy (%)", fontsize=12)
            ax.set_ylim(
                min(means) - 5 if means else 0, max(means) + 5 if means else 100
            )
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)

        title = (
            f"Model Performance with Prompt: {prompt_name.replace('_', ' ').title()}"
            if comparing_models
            else f"Prompt Performance{' with Model: ' + model_name.replace('_', ' ').title() if model_name else ''}"
        )
        plt.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.show()

        return {
            "significance": significance_results, 
            "bootstrap": bootstrap_results
        }

    except Exception as e:
        # Restore print function in case of error
        builtins.print = original_print
        print(f"Error in statistical analysis: {e}")
        import traceback

        traceback.print_exc()
        return None
