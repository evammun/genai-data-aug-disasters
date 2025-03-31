"""
prompt_generation.py

This script provides multiprocessed caption generation for images by calling large language models (e.g., Claude, GPT).
It reads per-label allocations (synthetic_image_allocations.csv) and uses multiprocessing to distribute work.
It includes an 'extract_caption_text' function that searches for <caption> tags in LLM responses, as well as refusal detection
to handle disclaimers or refusal patterns. The 'used_fallback' indicator is recorded if a fallback prompt is appended after an LLM refusal.
"""

import os
import csv
import math
import random
import re
import time
import threading
import concurrent.futures
import logging
import mimetypes
import base64
import pandas as pd
import anthropic
from openai import OpenAI
import config


def configure_logging():
    """
    Configure the logging levels for all relevant modules.
    This function sets the root logger and the loggers for httpx, anthropic, and openai to WARNING level.
    """
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def load_prompt_candidates(prompts_file: str) -> dict:
    """
    Load a collection of prompt templates from either a JSON or CSV file.

    If the file is JSON, the function expects a dictionary structure, where each key
    may be a string representation of a tuple (e.g., '("real", "prompt_key")'),
    and the value is the prompt text. Those keys are parsed into tuples using ast.literal_eval.

    If the file is CSV, the function expects columns named 'usage_type', 'prompt_key', and 'prompt_text'.
    The combination of (usage_type, prompt_key) is used as the dictionary key, and 'prompt_text' is the value.

    Args:
        prompts_file (str): Path to the JSON or CSV file containing prompt templates.

    Returns:
        dict: A dictionary where keys are tuples (e.g. ("real", "some_key")) and values are strings (prompt text).
    """
    import json
    import ast

    _, ext = os.path.splitext(prompts_file.lower())
    if ext == ".json":
        # Handle JSON prompts
        with open(prompts_file, "r", encoding="utf-8-sig") as f:
            raw = json.load(f)
        prompts_dict = {}
        for k_str, text in raw.items():
            k_tuple = ast.literal_eval(
                k_str
            )  # Convert string representation to a tuple
            prompts_dict[k_tuple] = text
        return prompts_dict
    else:
        # Handle CSV prompts
        df = pd.read_csv(prompts_file)
        prompts_dict = {}
        for _, row in df.iterrows():
            usage = row["usage_type"].strip()
            key = row["prompt_key"].strip()
            text = row["prompt_text"]
            prompts_dict[(usage, key)] = text
        return prompts_dict


def load_medic_train_df() -> pd.DataFrame:
    """
    Load the MEDIC training dataset from a file path specified in config.

    This function reads a tab-separated file path from config.get_data_paths()["train"],
    and then ensures specific columns are cast to string type. The returned DataFrame
    contains image paths and relevant label columns (damage_severity, informative, humanitarian, disaster_types).

    Returns:
        pd.DataFrame: A dataframe with columns ["image_path", "damage_severity", "informative", "humanitarian", "disaster_types"].
    """
    data_paths = config.get_data_paths()
    df = pd.read_csv(data_paths["train"], sep="\t")
    label_cols = ["damage_severity", "informative", "humanitarian", "disaster_types"]
    for col in label_cols:
        df[col] = df[col].astype(str)
    return df[["image_path"] + label_cols]


def guess_media_type(image_path: str) -> str:
    """
    Guess the MIME type of an image file using Python's built-in 'mimetypes' module.
    If the guessed type is not found or does not start with 'image/', default to 'image/jpeg'.

    Args:
        image_path (str): Path to the image file on disk.

    Returns:
        str: The determined or default MIME type (e.g., 'image/jpeg').
    """
    mt, _ = mimetypes.guess_type(image_path)
    if not mt or not mt.startswith("image/"):
        return "image/jpeg"
    return mt


# This is a regular expression used to detect refusal or disclaimer phrases in LLM responses.
REFUSAL_REGEX = (
    r"(i[\s]*am[\s]*(sorry|unable|not[\s]*comfortable)"
    r"|i[\s]*cannot"
    r"|i[\s]*can't"
    r"|i[\s]*apologize"
    r"|i[\s]*refuse"
    r"|i[\s]*do[\s]*not[\s]*feel[\s]*comfortable"
    r"|i[\s]*am[\s]*not[\s]*comfortable"
    r"|i[\s]*won[\s]*['’]?t[\s]*(describe|provide|do|create|comply|continue)"
    r"|i[\s]*will[\s]*not[\s]*(provide|create|produce|generate|do|comply|participate|do[\s]*this|go[\s]*further)"
    r"|i[\s]*do[\s]*not[\s]*want[\s]*to[\s]*(describe|provide|create|analyze|analyse|continue|do)"
    r"|creating[\s]*detailed[\s]*(descriptions|analysis)[\s]*would[\s]*be[\s]*(inappropriate|harmful|exploitative)"
    r"|i[\s]*do[\s]*n't[\s]*feel[\s]*comfortable[\s]*providing[\s]*(a[\s]*detailed|the[\s]*requested|these[\s]*kind[\s]*of)?[\s]*(analysis|caption)?"
    r"|this[\s]*is[\s]*(disturbing|deeply[\s]*concerning|deeply[\s]*troubling|triggering)"
    r"|it[\s]*would[\s]*be[\s]*(inappropriate|harmful|exploitative)"
    r"|analyzing[\s]*such[\s]*sensitive[\s]*imagery[\s]*would[\s]*be[\s]*(inappropriate|harmful|exploitative)"
    r")"
    r".{0,500}"
)


def is_refusal(response_text: str) -> bool:
    """
    Determine if a given text string is a refusal or disclaimer
    based on a set of known refusal patterns.

    Args:
        response_text (str): The response text from an LLM.

    Returns:
        bool: True if the text matches any known refusal/disclaimer patterns, else False.
    """
    if not response_text:
        return True
    return bool(re.search(REFUSAL_REGEX, response_text.lower()))


def call_gpt_text(
    image_path: str, prompt_text: str, temperature: float = 0.4, model: str = None
):
    """
    Call a GPT-based model (e.g. GPT-4 with vision support) via the OpenAI client.
    If 'image_path' is provided, the image is opened and base64-encoded, then included
    in the user message. If the API call is successful and a valid response is returned,
    the function returns (response_text, None). Otherwise, it returns (None, error_message).

    Args:
        image_path (str): Path to the image file to include in the GPT prompt, or None.
        prompt_text (str): The text prompt to send to GPT.
        temperature (float): Sampling temperature for creative or conservative responses.
        model (str): Which GPT model to use. If None, uses config.OPENAI_MODEL from user config.

    Returns:
        tuple: (text_out, err)
            text_out: The GPT response text if successful, else None.
            err: An error message or None if no error occurred.
    """
    client = OpenAI(api_key=config.OPENAI_KEY)

    b64_data = None
    if image_path:
        # Read and encode image if provided
        try:
            with open(image_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            return None, f"FileReadError: {e}"

    # Build the message content for GPT, including the image if applicable
    user_content = []
    if b64_data:
        media_type = guess_media_type(image_path)
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{b64_data}",
                    "detail": "auto",
                },
            }
        )

    # Add the textual portion of the prompt
    user_content.append({"type": "text", "text": prompt_text})

    max_retries = 3
    attempts = 0

    # Try up to max_retries in case of rate-limiting or temporary errors
    while attempts < max_retries:
        try:
            response = client.chat.completions.create(
                model=model or config.OPENAI_MODEL,
                messages=[
                    {"role": "user", "content": user_content},
                ],
                max_tokens=5000,
                temperature=temperature,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "rate limit" in err_msg:
                time.sleep(5)
                attempts += 1
                continue
            return None, f"APICallError: {str(e)}"

        if not response.choices or len(response.choices) == 0:
            return None, "EmptyChoices"

        choice = response.choices[0]
        if not hasattr(choice, "message") or not choice.message:
            return None, "NoMessage"

        raw_reply = choice.message.content.strip()
        if not raw_reply:
            return None, "EmptyReply"

        return raw_reply, None

    return None, "APICallError: too many 429 rate-limit retries"


def call_claude_text(
    image_path: str, prompt_text: str, temperature: float = 0.4, model: str = None
):
    """
    Call an Anthropic Claude-based model via the Anthropic client.
    If 'image_path' is provided, the image is read and base64-encoded,
    then included in the user message. If the API call is successful
    and a valid response is returned, the function returns (response_text, None).
    Otherwise, it returns (None, error_message).

    Args:
        image_path (str): Path to the image file to include in the prompt, or None.
        prompt_text (str): The text prompt for Claude.
        temperature (float): Sampling temperature to control creativity/conservatism.
        model (str): Which Claude model to use. If None, uses config.ANTHROPIC_MODEL.

    Returns:
        tuple: (text_out, err)
            text_out: The Claude response text if successful, else None.
            err: An error message or None if no error occurred.
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_KEY)

    b64_data = None
    if image_path:
        try:
            with open(image_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            return None, f"FileReadError: {e}"

    user_message = []
    if b64_data:
        media_type = guess_media_type(image_path)
        user_message.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64_data,
                },
            }
        )
    user_message.append({"type": "text", "text": prompt_text})

    max_retries = 3
    attempts = 0

    # Retry logic for handling 429 rate limit errors
    while attempts < max_retries:
        try:
            response = client.messages.create(
                model=model or config.ANTHROPIC_MODEL,
                messages=[{"role": "user", "content": user_message}],
                max_tokens=5000,
                temperature=temperature,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "rate limit" in err_msg:
                time.sleep(5)
                attempts += 1
                continue
            return None, f"APICallError: {str(e)}"

        if not response.content or len(response.content) == 0:
            return None, "EmptyContent"

        raw_reply = response.content[0].text.strip()
        if not raw_reply:
            return None, "EmptyReply"

        return raw_reply, None

    return None, "APICallError: too many 429 rate-limit retries"


def call_llm_with_fallback(
    llm_name: str,
    main_prompt: str,
    image_path: str = None,
    fallback_texts=None,
    max_tries=3,
    is_hypothetical: bool = False,
    model: str = None,
):
    """
    Attempt to generate text from an LLM multiple times, optionally appending fallback prompts if needed.
    If the LLM returns a refusal or an empty output, a fallback prompt is appended to the prompt and retried.

    Args:
        llm_name (str): Either 'gpt' or 'claude' to decide which LLM calling function to use.
        main_prompt (str): The primary prompt text to be given to the LLM.
        image_path (str): Optional path to an image file, to be encoded and appended to the prompt.
        fallback_texts (list): A list of fallback prompt strings appended if the primary prompt fails or refuses.
        max_tries (int): The maximum number of attempts allowed before giving up.
        is_hypothetical (bool): If True, uses a higher temperature for more creative generation.
        model (str): Specific model name to override defaults if provided.

    Returns:
        tuple: (text_out, err, used_fallback)
            text_out (str or None): The final output from the LLM on success, else None.
            err (str or None): Any error or refusal reason; None if successful.
            used_fallback (bool): True if any fallback prompt was appended during the attempts.
    """
    if fallback_texts is None:
        fallback_texts = []

    attempt = 1
    current_prompt = main_prompt
    temperature = 0.8 if is_hypothetical else 0.4
    used_fallback = False

    while attempt <= max_tries:
        # Choose the correct function based on llm_name or model
        if llm_name.lower() == "gpt" or (model and "gpt" in model.lower()):
            text_out, err = call_gpt_text(
                image_path, current_prompt, temperature=temperature, model=model
            )
        else:
            text_out, err = call_claude_text(
                image_path, current_prompt, temperature=temperature, model=model
            )

        # If there was an API or other error, try a fallback if any left
        if err:
            if attempt < max_tries and (attempt - 1) < len(fallback_texts):
                used_fallback = True
                current_prompt += "\n\n" + fallback_texts[attempt - 1]
                attempt += 1
                continue
            return None, err, used_fallback

        # Check for refusal
        if text_out and is_refusal(text_out):
            if attempt < max_tries and (attempt - 1) < len(fallback_texts):
                used_fallback = True
                current_prompt += "\n\n" + fallback_texts[attempt - 1]
                attempt += 1
                continue
            return None, "Refusal", used_fallback

        # If we have no text output at all, consider it a failure that might trigger fallback
        if not text_out:
            if attempt < max_tries and (attempt - 1) < len(fallback_texts):
                used_fallback = True
                current_prompt += "\n\n" + fallback_texts[attempt - 1]
                attempt += 1
                continue
            return None, "EmptyOutput", used_fallback

        # If we got a valid text, return it
        return text_out, None, used_fallback

    return None, "MaxRetriesExceeded", used_fallback


# Regexes used to find <caption>...</caption> or partial <caption> in LLM outputs
CAPTION_REGEX = re.compile(r"<caption>(.*?)<\/caption>", re.IGNORECASE | re.DOTALL)
PARTIAL_CAPTION_REGEX = re.compile(r"<caption>(.*)", re.IGNORECASE | re.DOTALL)


def extract_caption_text(llm_raw_reply: str) -> str:
    """
    Extract the textual content between <caption> and </caption> tags from an LLM response.
    If no well-formed tags are found, it searches for a partial <caption> and returns all text after it.
    If neither is found, returns an empty string.

    Args:
        llm_raw_reply (str): The raw text output from the LLM.

    Returns:
        str: The extracted caption text if found, otherwise an empty string.
    """
    if not llm_raw_reply:
        return ""
    match = CAPTION_REGEX.search(llm_raw_reply)
    if match:
        return match.group(1).strip()
    partial = PARTIAL_CAPTION_REGEX.search(llm_raw_reply)
    if partial:
        return partial.group(1).strip()
    return ""


def generate_prompts_for_one_combo(
    df_train: pd.DataFrame,
    label_combo: dict,
    total_needed: int,
    real_prompt: str,
    hypothetical_prompt: str,
    proportion_real: float,
    llm_name: str,
    fallback_texts: list,
    model: str = None,
    outer_pbar=None,
):
    """
    Generate captions for a specific label combination in a dataset, using both real images
    (if available) and hypothetical descriptions. This function attempts to produce a total
    of 'total_needed' captions, allocating some fraction to real images and the remainder
    to hypothetical. If not enough real images exist, more hypothetical descriptions are generated.

    Args:
        df_train (pd.DataFrame): The training DataFrame containing image paths and labels.
        label_combo (dict): Dictionary specifying the label combination to filter (e.g., damage_severity=..., etc.).
        total_needed (int): Total number of captions desired for this combination.
        real_prompt (str): Prompt text used when real images are passed to the LLM.
        hypothetical_prompt (str): Prompt text used for hypothetical (non-image) requests.
        proportion_real (float): Fraction of total_needed that should be real images (0 to 1).
        llm_name (str): Either 'gpt' or 'claude' to determine which LLM function is used.
        fallback_texts (list): A list of fallback prompt segments appended upon refusal or error.
        model (str): Specific model name if overriding default.
        outer_pbar: Optional progress bar to update as generation proceeds.

    Returns:
        tuple: (success_records, failure_records), each a list of dictionary rows describing the outcomes.
    """
    success_records = []
    failure_records = []

    # Filter the training DataFrame to only rows matching the given label combination
    df_filtered = df_train[
        (df_train["damage_severity"] == str(label_combo["damage_severity"]))
        & (df_train["informative"] == str(label_combo["informative"]))
        & (df_train["humanitarian"] == str(label_combo["humanitarian"]))
        & (df_train["disaster_types"] == str(label_combo["disaster_types"]))
    ]
    df_filtered = df_filtered.reset_index(drop=True)
    num_available = len(df_filtered)

    # Decide how many real vs hypothetical images to generate
    desired_real = int(total_needed * proportion_real)
    if num_available == 0:
        num_real = 0
        num_hypo = total_needed
    else:
        num_real = min(desired_real, num_available)
        num_hypo = total_needed - num_real

    real_paths = []
    if num_real > 0:
        all_paths = list(df_filtered["image_path"])
        # Sample if we have more available than needed
        if num_real < num_available:
            real_paths = random.sample(all_paths, num_real)
        else:
            real_paths = all_paths.copy()

    # Part 1: Generate from real images
    for img_path_rel in real_paths:
        if outer_pbar:
            outer_pbar.update(1)

        fullpath = os.path.join(config.DATA_DIR, img_path_rel.lstrip("/"))

        # Format the real prompt with labels, similar to how you do it for hypothetical
        real_prompt_with_labels = real_prompt
        # Replace placeholders if they exist in the prompt
        if "[DISASTER_TYPE]" in real_prompt:
            real_prompt_with_labels = real_prompt_with_labels.replace(
                "[DISASTER_TYPE]", label_combo["disaster_types"]
            )
            real_prompt_with_labels = real_prompt_with_labels.replace(
                "[DAMAGE_SEVERITY]", label_combo["damage_severity"]
            )
            real_prompt_with_labels = real_prompt_with_labels.replace(
                "[HUMANITARIAN]", label_combo["humanitarian"]
            )
            real_prompt_with_labels = real_prompt_with_labels.replace(
                "[INFORMATIVE]", label_combo["informative"]
            )

        text_out, err, used_fb = call_llm_with_fallback(
            llm_name=llm_name,
            main_prompt=real_prompt_with_labels,  # Now using the version with labels
            image_path=fullpath,
            fallback_texts=fallback_texts,
            max_tries=3,
            is_hypothetical=False,
            model=model,
        )

        base_record = {
            "image_path": img_path_rel,
            "damage_severity": label_combo["damage_severity"],
            "informative": label_combo["informative"],
            "humanitarian": label_combo["humanitarian"],
            "disaster_types": label_combo["disaster_types"],
            "source_type": "real",
        }

        if text_out and not err:
            llm_output = text_out
            extracted = extract_caption_text(llm_output)
            # If we can't find <caption> tags, fallback to the entire text
            if not extracted.strip():
                extracted = llm_output

            r = dict(base_record)
            r["llm_output"] = llm_output
            r["caption_text"] = extracted
            r["failure_reason"] = ""
            r["used_fallback"] = str(used_fb)
            success_records.append(r)
        else:
            r = dict(base_record)
            r["llm_output"] = text_out if text_out else ""
            r["caption_text"] = ""
            r["failure_reason"] = err if err else "Unknown"
            r["used_fallback"] = str(False)
            failure_records.append(r)

    # Part 2: Generate hypothetical prompts
    for _ in range(num_hypo):
        if outer_pbar:
            outer_pbar.update(1)

        # A small method to create a string of random 'diversity' keywords to encourage variety
        diversity_keywords = {
            "Time of Day": [
                "Dawn",
                "Sunrise",
                "Morning",
                "Midday",
                "Afternoon",
                "Dusk",
                "Sunset",
                "Evening",
                "Night",
                "Midnight",
                "Early hours",
                "Late morning",
            ],
            "Geography": ["Northern hemisphere", "Southern hemisphere", "West", "East"],
            "Letter": [
                "Include TWO distinct words starting with letter A",
                "Include TWO distinct words starting with letter B",
                "Include TWO distinct words starting with letter C",
                "Include TWO distinct words starting with letter D",
                "Include TWO distinct words starting with letter E",
                "Include TWO distinct words starting with letter F",
                "Include TWO distinct words starting with letter G",
                "Include TWO distinct words starting with letter H",
                "Include TWO distinct words starting with letter I",
                "Include TWO distinct words starting with letter L",
                "Include TWO distinct words starting with letter M",
                "Include TWO distinct words starting with letter N",
                "Include TWO distinct words starting with letter O",
                "Include TWO distinct words starting with letter P",
                "Include TWO distinct words starting with letter R",
                "Include TWO distinct words starting with letter S",
                "Include TWO distinct words starting with letter T",
                "Include TWO distinct words starting with letter U",
                "Include TWO distinct words starting with letter W",
            ],
            "Camera Angle": [
                "Aerial view",
                "Drone view",
                "High angle",
                "Eye level",
                "Low angle",
                "Ground level",
                "Wide angle",
                "Medium shot",
                "Close-up",
                "Distance shot",
                "Tilted angle",
                "Panoramic view",
            ],
        }

        def get_random_diversity_keywords():
            """
            Select a random keyword from each diversity category.
            Returns a dictionary with the chosen keyword under each category name.
            """
            selected = {}
            for category, options in diversity_keywords.items():
                selected[category] = random.choice(options)
            return selected

        def create_diversity_keyword_string():
            """
            Assemble the selected keywords from all categories into a comma-separated string.
            """
            selected = get_random_diversity_keywords()
            keyword_string = ", ".join(selected.values())
            return keyword_string

        diversity_selection = create_diversity_keyword_string()

        # Perform the inline replacements for each label in the hypothetical prompt
        replaced_prompt = hypothetical_prompt
        replaced_prompt = replaced_prompt.replace(
            "[DISASTER_TYPE]", label_combo["disaster_types"]
        )
        replaced_prompt = replaced_prompt.replace(
            "[DAMAGE_SEVERITY]", label_combo["damage_severity"]
        )
        replaced_prompt = replaced_prompt.replace(
            "[HUMANITARIAN]", label_combo["humanitarian"]
        )
        replaced_prompt = replaced_prompt.replace(
            "[INFORMATIVE]", label_combo["informative"]
        )

        # Append the diversity keywords at the end, but remove the 'Labels:' line
        prompt_with_labels = (
            f"{replaced_prompt}\n\nInspiration keywords: {diversity_selection}"
        )

        text_out, err, used_fb = call_llm_with_fallback(
            llm_name=llm_name,
            main_prompt=prompt_with_labels,
            image_path=None,
            fallback_texts=fallback_texts,
            max_tries=3,
            is_hypothetical=True,
            model=model,
        )

        base_record = {
            "image_path": "hypothetical",
            "damage_severity": label_combo["damage_severity"],
            "informative": label_combo["informative"],
            "humanitarian": label_combo["humanitarian"],
            "disaster_types": label_combo["disaster_types"],
            "source_type": "hypothetical",
        }

        if text_out and not err:
            llm_output = text_out
            extracted = extract_caption_text(llm_output)
            if not extracted.strip():
                extracted = llm_output

            r = dict(base_record)
            r["llm_output"] = llm_output
            r["caption_text"] = extracted
            r["failure_reason"] = ""
            r["used_fallback"] = str(used_fb)
            success_records.append(r)
        else:
            r = dict(base_record)
            r["llm_output"] = text_out if text_out else ""
            r["caption_text"] = ""
            r["failure_reason"] = err if err else "Unknown"
            r["used_fallback"] = str(False)
            failure_records.append(r)

    return success_records, failure_records


import multiprocessing


def generate_captions(
    prompts_file=None,
    real_prompt_key=None,
    hypo_prompt_key=None,
    fallback_keys=None,
    llm_name="claude",
    label_combo=None,
    total_images=None,
    proportion_real=0.75,
    alloc_csv="synthetic_image_allocations.csv",
    out_csv="test_captions_output.csv",
    model=None,
    input_file=None,
    verbose=False,
):
    """
    Generate captions for disaster images or hypothetical scenarios based on label combinations or a specified dataset.

    This function can process real images (calling the LLM with an image) or hypothetical scenarios
    (calling the LLM with text only). It can also handle fallback prompts if refusals occur.

    Args:
        prompts_file (str): Path to a JSON or CSV of prompt templates;
            if None, tries to use config.AUGMENTATION_PROMPTS_FILE.
        real_prompt_key (str): Dictionary key for the real (image-based) prompt.
        hypo_prompt_key (str): Dictionary key for the hypothetical prompt.
        fallback_keys (list): List of dictionary keys for fallback prompt segments if the main prompt fails or is refused.
        llm_name (str): Indicates which LLM to call ("claude" or "gpt").
        label_combo (dict): If provided, generation is restricted to images matching this label combination.
        total_images (int): If set, controls how many images to process in total. If None, processes all.
        proportion_real (float): Fraction of total_images that should be real images; the rest are hypothetical.
        alloc_csv (str): Path to the CSV describing how many synthetic images per label combo (may not be used in all flows).
        out_csv (str): Output CSV path for successful records.
        model (str): Model override (e.g., a specific GPT or Claude variant).
        input_file (str): If provided, a custom TSV/CSV dataset is loaded from this path instead of the default MEDIC train set.
        verbose (bool): If True, prints summary logs at the end.

    Returns:
        tuple: (total_successes, total_failures), the count of successful and failed generations overall.
    """
    import pandas as pd
    from tqdm import tqdm
    import os
    import random

    from src.augmentation.prompt_generation import (
        configure_logging,
        load_prompt_candidates,
        load_medic_train_df,
        _write_records_to_csv,
        call_llm_with_fallback,
    )
    import config

    # Suppress logs
    configure_logging()

    # Prepare the output directories for success/failure CSVs
    out_dir = os.path.dirname(out_csv)
    out_name = os.path.splitext(os.path.basename(out_csv))[0]
    failures_csv = os.path.join(out_dir, f"{out_name}_failures.csv")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total_successes = 0
    total_failures = 0

    # Handle the prompts file or fall back to a config default
    if prompts_file is None:
        if hasattr(config, "AUGMENTATION_PROMPTS_FILE"):
            prompts_file = config.AUGMENTATION_PROMPTS_FILE
        else:
            raise ValueError(
                "No prompts_file provided and config.AUGMENTATION_PROMPTS_FILE not defined."
            )

    # Load the dictionary of prompt templates
    prompts_dict = load_prompt_candidates(prompts_file)

    # Retrieve the selected prompts from the dictionary
    real_prompt = (
        prompts_dict.get(("real", real_prompt_key)) if real_prompt_key else None
    )
    hypothetical_prompt = (
        prompts_dict.get(("hypothetical", hypo_prompt_key)) if hypo_prompt_key else None
    )

    # Convert fallback_keys into a list of fallback prompt strings
    fallback_texts = []
    if fallback_keys:
        fallback_texts = [
            prompts_dict.get(("fallback", fk))
            for fk in fallback_keys
            if prompts_dict.get(("fallback", fk))
        ]

    # Load the data (either from input_file or the default MEDIC dataset)
    if input_file:
        df_train = pd.read_csv(input_file, sep="\t")
    else:
        df_train = load_medic_train_df()

    # Determine how many items we will process
    if total_images is not None:
        total_ops = total_images
        num_real = int(total_images * proportion_real)
        num_hypo = total_images - num_real
    else:
        total_ops = len(df_train)
        num_real = total_ops
        num_hypo = 0

    with tqdm(
        total=total_ops, desc=f"Generating captions with {model or 'default model'}"
    ) as pbar:
        # If a specific label combo is given, restrict generation to that subset
        if label_combo is not None:
            subset = df_train[
                (
                    df_train["damage_severity"].astype(str)
                    == str(label_combo["damage_severity"])
                )
                & (
                    df_train["informative"].astype(str)
                    == str(label_combo["informative"])
                )
                & (
                    df_train["humanitarian"].astype(str)
                    == str(label_combo["humanitarian"])
                )
                & (
                    df_train["disaster_types"].astype(str)
                    == str(label_combo["disaster_types"])
                )
            ].copy()

            subset = subset.reset_index(drop=True)
            all_paths = list(subset["image_path"])
            num_available = len(all_paths)

            if num_available == 0:
                if verbose:
                    print(
                        f"No real images found for combination {label_combo}, using all hypothetical"
                    )
                num_real = 0
                num_hypo = total_images
            else:
                if num_real <= num_available:
                    real_paths = all_paths[:num_real]
                else:
                    # If we need more real images than available, we resample duplicates
                    real_paths = all_paths.copy()
                    additional_needed = num_real - num_available
                    if additional_needed > 0 and verbose:
                        print(
                            f"Only {num_available} real images available, resampling {additional_needed} images"
                        )
                    if additional_needed > 0:
                        resampled = random.choices(all_paths, k=additional_needed)
                        real_paths.extend(resampled)

            success_records = []
            failure_records = []

            # Generate from real images
            for img_path_rel in real_paths:
                pbar.update(1)

                img_path_rel = img_path_rel.lstrip("/")
                fullpath = os.path.normpath(os.path.join(config.DATA_DIR, img_path_rel))

                text_out, err, used_fallback = call_llm_with_fallback(
                    llm_name=llm_name,
                    main_prompt=real_prompt,
                    image_path=fullpath,
                    fallback_texts=fallback_texts,
                    max_tries=3,
                    is_hypothetical=False,
                    model=model,
                )

                record = {
                    "image_path": img_path_rel,
                    "damage_severity": label_combo["damage_severity"],
                    "informative": label_combo["informative"],
                    "humanitarian": label_combo["humanitarian"],
                    "disaster_types": label_combo["disaster_types"],
                    "source_type": "real",
                }

                if text_out and not err:
                    success_records.append({**record, "caption_text": text_out})
                else:
                    failure_records.append(
                        {
                            **record,
                            "caption_text": text_out if text_out else "",
                            "failure_reason": err if err else "Empty output",
                        }
                    )

            _write_records_to_csv(
                success_records,
                out_csv,
                real_prompt_key or "",
                hypo_prompt_key or "",
                ",".join(fallback_keys) if fallback_keys else "",
            )

            _write_records_to_csv(
                failure_records,
                failures_csv,
                real_prompt_key or "",
                hypo_prompt_key or "",
                ",".join(fallback_keys) if fallback_keys else "",
                is_failure=True,
            )

            total_successes += len(success_records)
            total_failures += len(failure_records)

            # Generate hypothetical if needed
            if hypothetical_prompt and num_hypo > 0:
                success_records = []
                failure_records = []

                for _ in range(num_hypo):
                    pbar.update(1)

                    combined_prompt = f"{hypothetical_prompt}\n\nLabels: {label_combo}"

                    text_out, err, used_fallback = call_llm_with_fallback(
                        llm_name=llm_name,
                        main_prompt=combined_prompt,
                        image_path=None,
                        fallback_texts=fallback_texts,
                        max_tries=3,
                        is_hypothetical=True,
                        model=model,
                    )

                    record = {
                        "image_path": "hypothetical",
                        "damage_severity": label_combo["damage_severity"],
                        "informative": label_combo["informative"],
                        "humanitarian": label_combo["humanitarian"],
                        "disaster_types": label_combo["disaster_types"],
                        "source_type": "hypothetical",
                    }

                    if text_out and not err:
                        success_records.append({**record, "caption_text": text_out})
                    else:
                        failure_records.append(
                            {
                                **record,
                                "caption_text": text_out if text_out else "",
                                "failure_reason": err if err else "Empty output",
                            }
                        )

                _write_records_to_csv(
                    success_records,
                    out_csv,
                    real_prompt_key or "",
                    hypo_prompt_key or "",
                    ",".join(fallback_keys) if fallback_keys else "",
                )

                _write_records_to_csv(
                    failure_records,
                    failures_csv,
                    real_prompt_key or "",
                    hypo_prompt_key or "",
                    ",".join(fallback_keys) if fallback_keys else "",
                    is_failure=True,
                )

                total_successes += len(success_records)
                total_failures += len(failure_records)

        else:
            # No label combo specified => either sample from entire dataset or process all
            if total_images is not None:
                num_real = int(total_images * proportion_real)

                # Sample real images if the user wants a specific number
                df_train_sample = df_train.sample(
                    n=min(num_real, len(df_train))
                ).reset_index(drop=True)

                success_records = []
                failure_records = []

                for _, row in df_train_sample.iterrows():
                    pbar.update(1)

                    img_path_rel = row["image_path"].lstrip("/")
                    fullpath = os.path.normpath(
                        os.path.join(config.DATA_DIR, img_path_rel)
                    )

                    text_out, err, used_fallback = call_llm_with_fallback(
                        llm_name=llm_name,
                        main_prompt=real_prompt,
                        image_path=fullpath,
                        fallback_texts=fallback_texts,
                        max_tries=3,
                        is_hypothetical=False,
                        model=model,
                    )

                    record = {
                        "image_path": img_path_rel,
                        "damage_severity": row["damage_severity"],
                        "informative": row["informative"],
                        "humanitarian": row["humanitarian"],
                        "disaster_types": row["disaster_types"],
                        "source_type": "real",
                    }

                    if text_out and not err:
                        success_records.append({**record, "caption_text": text_out})
                    else:
                        failure_records.append(
                            {
                                **record,
                                "caption_text": text_out if text_out else "",
                                "failure_reason": err if err else "Empty output",
                            }
                        )

                _write_records_to_csv(
                    success_records,
                    out_csv,
                    real_prompt_key or "",
                    hypo_prompt_key or "",
                    ",".join(fallback_keys) if fallback_keys else "",
                )

                _write_records_to_csv(
                    failure_records,
                    failures_csv,
                    real_prompt_key or "",
                    hypo_prompt_key or "",
                    ",".join(fallback_keys) if fallback_keys else "",
                    is_failure=True,
                )

                total_successes += len(success_records)
                total_failures += len(failure_records)

                # Now generate hypothetical for the remainder
                num_hypo = total_images - num_real
                if num_hypo > 0:
                    success_records = []
                    failure_records = []

                    # Use the labels from the first row as a template for hypothetical generation
                    row = df_train.iloc[0]
                    hypo_combo = {
                        "damage_severity": row["damage_severity"],
                        "informative": row["informative"],
                        "humanitarian": row["humanitarian"],
                        "disaster_types": row["disaster_types"],
                    }

                    for _ in range(num_hypo):
                        pbar.update(1)

                        combined_prompt = (
                            f"{hypothetical_prompt}\n\nLabels: {hypo_combo}"
                        )

                        text_out, err, used_fallback = call_llm_with_fallback(
                            llm_name=llm_name,
                            main_prompt=combined_prompt,
                            image_path=None,
                            fallback_texts=fallback_texts,
                            max_tries=3,
                            is_hypothetical=True,
                            model=model,
                        )

                        record = {
                            "image_path": "hypothetical",
                            "damage_severity": hypo_combo["damage_severity"],
                            "informative": hypo_combo["informative"],
                            "humanitarian": hypo_combo["humanitarian"],
                            "disaster_types": hypo_combo["disaster_types"],
                            "source_type": "hypothetical",
                        }

                        if text_out and not err:
                            success_records.append({**record, "caption_text": text_out})
                        else:
                            failure_records.append(
                                {
                                    **record,
                                    "caption_text": text_out if text_out else "",
                                    "failure_reason": err if err else "Empty output",
                                }
                            )

                    _write_records_to_csv(
                        success_records,
                        out_csv,
                        real_prompt_key or "",
                        hypo_prompt_key or "",
                        ",".join(fallback_keys) if fallback_keys else "",
                    )

                    _write_records_to_csv(
                        failure_records,
                        failures_csv,
                        real_prompt_key or "",
                        hypo_prompt_key or "",
                        ",".join(fallback_keys) if fallback_keys else "",
                        is_failure=True,
                    )

                    total_successes += len(success_records)
                    total_failures += len(failure_records)
            else:
                # total_images=None => process all images as "real" only
                success_records = []
                failure_records = []

                for _, row in df_train.iterrows():
                    pbar.update(1)

                    img_path_rel = row["image_path"].lstrip("/")
                    fullpath = os.path.normpath(
                        os.path.join(config.DATA_DIR, img_path_rel)
                    )

                    text_out, err, used_fallback = call_llm_with_fallback(
                        llm_name=llm_name,
                        main_prompt=real_prompt,
                        image_path=fullpath,
                        fallback_texts=fallback_texts,
                        max_tries=3,
                        is_hypothetical=False,
                        model=model,
                    )

                    record = {
                        "image_path": img_path_rel,
                        "damage_severity": row["damage_severity"],
                        "informative": row["informative"],
                        "humanitarian": row["humanitarian"],
                        "disaster_types": row["disaster_types"],
                        "source_type": "real",
                    }

                    if text_out and not err:
                        success_records.append({**record, "caption_text": text_out})
                    else:
                        failure_records.append(
                            {
                                **record,
                                "caption_text": text_out if text_out else "",
                                "failure_reason": err if err else "Empty output",
                            }
                        )

                _write_records_to_csv(
                    success_records,
                    out_csv,
                    real_prompt_key or "",
                    hypo_prompt_key or "",
                    ",".join(fallback_keys) if fallback_keys else "",
                )

                _write_records_to_csv(
                    failure_records,
                    failures_csv,
                    real_prompt_key or "",
                    hypo_prompt_key or "",
                    ",".join(fallback_keys) if fallback_keys else "",
                    is_failure=True,
                )

                total_successes += len(success_records)
                total_failures += len(failure_records)

    if verbose:
        print("\nGeneration Summary:")
        print(f"Successfully generated captions: {total_successes}")
        print(f"Failed generations: {total_failures}")
        if total_successes + total_failures > 0:
            success_rate = total_successes / (total_successes + total_failures) * 100
            print(f"Success rate: {success_rate:.1f}%")
        print(f"\nOutput files:")
        print(f"Successful generations: {out_csv}")
        print(f"Failed generations: {failures_csv}")

    return total_successes, total_failures


def _write_records_to_csv(
    records: list,
    out_csv: str,
    real_prompt_key: str,
    hypo_prompt_key: str,
    fallback_keys_str: str,
    is_failure: bool = False,
):
    """
    Append a list of result dictionaries to a CSV file. If the CSV file does not exist,
    headers are written first. Otherwise, records are appended.

    Args:
        records (list): A list of dictionaries containing caption generation results.
        out_csv (str): Path to the CSV file where results are written.
        real_prompt_key (str): Identifier for the real-image prompt used.
        hypo_prompt_key (str): Identifier for the hypothetical prompt used.
        fallback_keys_str (str): Comma-separated list of fallback prompt keys used.
        is_failure (bool): If True, writes to the 'failure_reason' column, else leaves it blank.

    Returns:
        None
    """
    if not records:
        return

    if os.path.dirname(out_csv):
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    headers = [
        "image_path",
        "damage_severity",
        "informative",
        "humanitarian",
        "disaster_types",
        "caption_text",
        "source_type",
        "real_prompt_key",
        "hypo_prompt_key",
        "fallback_keys",
    ]
    if is_failure:
        headers.append("failure_reason")

    # Open CSV in append mode and handle quotes/escape for multiline fields
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=headers, quoting=csv.QUOTE_ALL, escapechar="\\"
        )
        if f.tell() == 0:
            writer.writeheader()

        for rec in records:
            row = dict(rec)
            row["real_prompt_key"] = real_prompt_key
            row["hypo_prompt_key"] = hypo_prompt_key
            row["fallback_keys"] = fallback_keys_str
            writer.writerow(row)


csv_lock = multiprocessing.Lock()


def caption_subset_test(
    balanced_subset_path: str,
    output_csv: str,
    prompt_pairs: list,
    models: list,
    fallback_keys: list = None,
    process_all: bool = True,
    generate_hypothetical: bool = True,
    total_images: int = None,
    proportion_real: float = 0.75,
    silence_generation: bool = False,
):
    """
    A helper function that repeatedly calls 'generate_captions' for a set of prompt pairs and models,
    writing results to a combined CSV file.

    Args:
        balanced_subset_path (str): Path to a dataset file (TSV/CSV) containing balanced or curated images.
        output_csv (str): Path to the final combined results CSV.
        prompt_pairs (list): A list of tuples (real_prompt_key, hypo_prompt_key) specifying which prompts to test.
        models (list): A list of model names (e.g., "claude-v1", "gpt-4-0314") to iterate over.
        fallback_keys (list): Optional fallback keys to use if the LLM refuses or errors out.
        process_all (bool): If True, process the entire dataset; otherwise do sampling.
        generate_hypothetical (bool): If True, generate hypothetical captions as well as real ones.
        total_images (int): When sampling, how many total images to process.
        proportion_real (float): Fraction of total_images that should be real images.
        silence_generation (bool): Currently unused parameter for controlling output verbosity.

    Returns:
        pd.DataFrame or None: A combined DataFrame of all results if any were generated, otherwise None.
    """
    import os
    import pandas as pd
    import tempfile
    import time
    import datetime
    from src.augmentation.prompt_generation import generate_captions
    import sys

    if fallback_keys is None:
        fallback_keys = []

    all_dfs = []
    total_elapsed = 0
    overall_success = 0
    overall_failure = 0
    operation_stats = []

    num_prompt_pairs = len(prompt_pairs)
    num_models = len(models)
    total_operations = num_prompt_pairs * num_models

    overall_start = time.time()
    print(
        f"Processing {num_prompt_pairs} prompt pairs × {num_models} models = {total_operations} operations"
    )

    # If process_all is True, read the entire dataset up front
    if process_all:
        df_subset = pd.read_csv(balanced_subset_path, sep="\t")
        num_rows = len(df_subset)
        print(f"Dataset contains {num_rows} images")

        if generate_hypothetical:
            # Identify unique combinations of label columns for hypothetical generation
            label_cols = [
                "damage_severity",
                "informative",
                "humanitarian",
                "disaster_types",
            ]
            unique_combinations = df_subset[label_cols].drop_duplicates()

            # Store these combos temporarily if needed
            import tempfile

            with tempfile.NamedTemporaryFile(
                suffix=".tsv", delete=False, mode="w"
            ) as temp_file:
                unique_combinations.to_csv(temp_file.name, sep="\t", index=False)
                subset_for_hypothetical = temp_file.name

            num_unique_combos = len(unique_combinations)
            print(
                f"Found {num_unique_combos} unique label combinations for hypothetical scenarios"
            )
    else:
        subset_for_hypothetical = None

    # For each prompt pair and each model, run the generation
    for prompt_idx, (r_key, h_key) in enumerate(prompt_pairs):
        for model_idx, model_name in enumerate(models):
            operation_num = prompt_idx * len(models) + model_idx + 1

            is_gpt_model = "gpt" in model_name.lower()
            llm_name = "gpt" if is_gpt_model else "claude"

            if is_gpt_model:
                model_short = "GPT4V"
            else:
                parts = model_name.split("-")
                model_short = parts[-2] if len(parts) > 2 else parts[-1]

            print(
                f"\n[{operation_num}/{total_operations}] Processing {r_key}/{h_key} with {model_short} ({llm_name})"
            )
            operation_start = time.time()

            # If processing the entire dataset
            if process_all:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_csv_real = os.path.join(tmpdir, "captions_temp_real.csv")
                    tmp_csv_real_fail = os.path.join(
                        tmpdir, "captions_temp_real_failures.csv"
                    )

                    print(f"  Generating real image captions...")
                    real_start = time.time()
                    sys.stdout.flush()

                    # Real images
                    generate_captions(
                        real_prompt_key=r_key,
                        hypo_prompt_key=None,
                        fallback_keys=fallback_keys,
                        llm_name=llm_name,
                        label_combo=None,
                        total_images=None,
                        proportion_real=1.0,
                        model=model_name,
                        input_file=balanced_subset_path,
                        out_csv=tmp_csv_real,
                    )

                    real_elapsed = time.time() - real_start
                    real_success = 0
                    real_fail = 0

                    # Accumulate success data
                    if os.path.exists(tmp_csv_real):
                        df_success = pd.read_csv(tmp_csv_real, encoding="utf-8")
                        real_success = len(df_success)
                        if real_success > 0:
                            df_success["is_failure"] = False
                            df_success["model_name"] = model_name
                            df_success["real_prompt_key"] = r_key
                            df_success["hypo_prompt_key"] = h_key
                            all_dfs.append(df_success)

                    # Accumulate failure data
                    if os.path.exists(tmp_csv_real_fail):
                        df_fail = pd.read_csv(tmp_csv_real_fail, encoding="utf-8")
                        real_fail = len(df_fail)
                        if real_fail > 0:
                            df_fail["is_failure"] = True
                            df_fail["model_name"] = model_name
                            df_fail["real_prompt_key"] = r_key
                            df_fail["hypo_prompt_key"] = h_key
                            all_dfs.append(df_fail)

                    print(
                        f"  ✓ Real images: {real_success} successful, {real_fail} failed ({real_elapsed:.1f}s)"
                    )
                    overall_success += real_success
                    overall_failure += real_fail

                # If hypothetical generation is enabled
                if generate_hypothetical and subset_for_hypothetical:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_csv_hypo = os.path.join(tmpdir, "captions_temp_hypo.csv")
                        tmp_csv_hypo_fail = os.path.join(
                            tmpdir, "captions_temp_hypo_failures.csv"
                        )

                        print(f"  Generating hypothetical scenarios...")
                        hypo_start = time.time()
                        sys.stdout.flush()

                        # Hypothetical images
                        generate_captions(
                            real_prompt_key=None,
                            hypo_prompt_key=h_key,
                            fallback_keys=fallback_keys,
                            llm_name=llm_name,
                            label_combo=None,
                            total_images=num_unique_combos,
                            proportion_real=0.0,
                            model=model_name,
                            input_file=subset_for_hypothetical,
                            out_csv=tmp_csv_hypo,
                        )

                        hypo_elapsed = time.time() - hypo_start
                        hypo_success = 0
                        hypo_fail = 0

                        if os.path.exists(tmp_csv_hypo):
                            df_success = pd.read_csv(tmp_csv_hypo, encoding="utf-8")
                            hypo_success = len(df_success)
                            if hypo_success > 0:
                                df_success["is_failure"] = False
                                df_success["model_name"] = model_name
                                df_success["real_prompt_key"] = r_key
                                df_success["hypo_prompt_key"] = h_key
                                all_dfs.append(df_success)

                        if os.path.exists(tmp_csv_hypo_fail):
                            df_fail = pd.read_csv(tmp_csv_hypo_fail, encoding="utf-8")
                            hypo_fail = len(df_fail)
                            if hypo_fail > 0:
                                df_fail["is_failure"] = True
                                df_fail["model_name"] = model_name
                                df_fail["real_prompt_key"] = r_key
                                df_fail["hypo_prompt_key"] = h_key
                                all_dfs.append(df_fail)

                        print(
                            f"  ✓ Hypothetical: {hypo_success} successful, {hypo_fail} failed ({hypo_elapsed:.1f}s)"
                        )
                        overall_success += hypo_success
                        overall_failure += hypo_fail

            else:
                # If not processing the entire dataset, do a sampling approach
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_csv = os.path.join(tmpdir, "captions_temp.csv")
                    tmp_csv_fail = os.path.join(tmpdir, "captions_temp_failures.csv")

                    print(f"  Generating sampled captions...")
                    sample_start = time.time()
                    sys.stdout.flush()

                    # Generate both real and hypothetical in one step
                    generate_captions(
                        real_prompt_key=r_key,
                        hypo_prompt_key=h_key,
                        fallback_keys=fallback_keys,
                        llm_name=llm_name,
                        total_images=total_images,
                        proportion_real=proportion_real,
                        model=model_name,
                        input_file=balanced_subset_path,
                        out_csv=tmp_csv,
                    )

                    sample_elapsed = time.time() - sample_start
                    sample_success = 0
                    sample_fail = 0

                    if os.path.exists(tmp_csv):
                        df_success = pd.read_csv(tmp_csv, encoding="utf-8")
                        sample_success = len(df_success)
                        if sample_success > 0:
                            df_success["is_failure"] = False
                            df_success["model_name"] = model_name
                            df_success["real_prompt_key"] = r_key
                            df_success["hypo_prompt_key"] = h_key
                            all_dfs.append(df_success)

                    if os.path.exists(tmp_csv_fail):
                        df_fail = pd.read_csv(tmp_csv_fail, encoding="utf-8")
                        sample_fail = len(df_fail)
                        if sample_fail > 0:
                            df_fail["is_failure"] = True
                            df_fail["model_name"] = model_name
                            df_fail["real_prompt_key"] = r_key
                            df_fail["hypo_prompt_key"] = h_key
                            all_dfs.append(df_fail)

                    print(
                        f"  ✓ Sample: {sample_success} successful, {sample_fail} failed ({sample_elapsed:.1f}s)"
                    )
                    overall_success += sample_success
                    overall_failure += sample_fail

            operation_elapsed = time.time() - operation_start
            total_elapsed += operation_elapsed

            print(
                f"  Overall: {operation_num}/{total_operations} operations complete ({operation_num/total_operations*100:.1f}%)"
            )

    # Cleanup if hypothetical combos were used
    if process_all and generate_hypothetical and "subset_for_hypothetical" in locals():
        if os.path.exists(subset_for_hypothetical):
            os.remove(subset_for_hypothetical)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        final_df.to_csv(output_csv, index=False, sep="\t")

        total_rows = len(final_df)
        total_real = len(final_df[final_df["source_type"] == "real"])
        total_hypo = len(final_df[final_df["source_type"] == "hypothetical"])
        total_fail = (
            sum(final_df["is_failure"]) if "is_failure" in final_df.columns else 0
        )
        total_success = total_rows - total_fail

        overall_elapsed = time.time() - overall_start
        elapsed_str = str(datetime.timedelta(seconds=int(overall_elapsed)))

        print("\n" + "=" * 50)
        print("FINAL SUMMARY")
        print("=" * 50)
        print(f"Total runtime: {elapsed_str}")
        print(f"Total images processed: {total_rows}")
        print(f"  • Real images: {total_real}")
        print(f"  • Hypothetical scenarios: {total_hypo}")
        print(
            f"Success rate: {total_success} successful ({total_success/total_rows*100:.1f}%), {total_fail} failed"
        )
        print(f"Output file: {output_csv}")
        print("=" * 50)

        return final_df
    else:
        print("No data generated.")
        return None


def append_records_to_csv(csv_path, records):
    """
    Safely append a list of row dictionaries to a CSV file in a multiprocessing-safe manner.
    A global Lock (csv_lock) is used to ensure that concurrent writes do not conflict.

    Args:
        csv_path (str): The output CSV path.
        records (list): A list of dictionaries representing rows to append.
    """
    if not records:
        return

    fieldnames = [
        "image_path",
        "damage_severity",
        "informative",
        "humanitarian",
        "disaster_types",
        "llm_output",
        "caption_text",
        "source_type",
        "fallback_keys",
        "is_failure",
        "model_name",
        "failure_reason",
        "used_fallback",
        "aug_img_path",
    ]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with csv_lock:
        file_empty = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_ALL,
                escapechar="\\",
            )
            if file_empty:
                writer.writeheader()

            for row in records:
                writer.writerow(row)


def process_label_combo(
    row_dict: dict,
    df_train_pickled: bytes,
    real_prompt: str,
    hypothetical_prompt: str,
    fallback_texts: list,
    proportion_real: float,
    model_name: str,
    out_csv: str,
):
    """
    Generate captions for a single label combination, given row_dict details.
    This function unpickles the dataset, filters it based on the label combination,
    and calls 'generate_prompts_for_one_combo' to produce the captions.
    The results (success and failure) are then appended to 'out_csv'.

    Args:
        row_dict (dict): A dictionary representing a single row from the allocations CSV,
            containing label combination fields and how many synthetic images to create.
        df_train_pickled (bytes): A pickled version of the training DataFrame for concurrency-safe transport.
        real_prompt (str): Prompt text used for real images.
        hypothetical_prompt (str): Prompt text used for hypothetical generation.
        fallback_texts (list): A list of fallback prompts to handle refusals/errors.
        proportion_real (float): Fraction of total images that should be real vs hypothetical.
        model_name (str): Which model to use (Claude or GPT).
        out_csv (str): Path where results are appended.

    Returns:
        tuple: (success_count, failure_count) for the label combination.
    """
    import pickle

    df_train = pickle.loads(df_train_pickled)

    label_combo = {
        "damage_severity": row_dict["DamageSeverity"],
        "informative": row_dict["Informativeness"],
        "humanitarian": row_dict["Humanitarian"],
        "disaster_types": row_dict["DisasterType"],
    }
    total_needed = int(row_dict["SyntheticImages"])

    success_records, failure_records = generate_prompts_for_one_combo(
        df_train=df_train,
        label_combo=label_combo,
        total_needed=total_needed,
        real_prompt=real_prompt,
        hypothetical_prompt=hypothetical_prompt,
        proportion_real=proportion_real,
        llm_name=model_name,
        fallback_texts=fallback_texts,
        model=model_name,
        outer_pbar=None,
    )

    def unify_record(rec, is_failure):
        return {
            "image_path": rec["image_path"],
            "damage_severity": rec["damage_severity"],
            "informative": rec["informative"],
            "humanitarian": rec["humanitarian"],
            "disaster_types": rec["disaster_types"],
            "llm_output": rec.get("llm_output", ""),
            "caption_text": rec.get("caption_text", ""),
            "source_type": rec.get("source_type", ""),
            "fallback_keys": rec.get("fallback_keys", ""),
            "is_failure": str(is_failure),
            "model_name": model_name,
            "failure_reason": rec.get("failure_reason", "") if is_failure else "",
            "used_fallback": rec.get("used_fallback", "False"),
        }

    srows = [unify_record(r, False) for r in success_records]
    frows = [unify_record(r, True) for r in failure_records]

    append_records_to_csv(out_csv, srows)
    append_records_to_csv(out_csv, frows)

    return (len(srows), len(frows))


def generate_all_captions(
    prompts_file: str,
    real_prompt_key: str,
    hypo_prompt_key: str,
    fallback_keys=None,
    llm_name: str = "claude",
    proportion_real: float = 0.75,
    alloc_csv: str = "synthetic_image_allocations.csv",
    out_csv: str = "llm_captions.csv",
    verbose: bool = False,
    max_workers: int = 4,
    max_rows: int = None,
    overwrite_output: bool = True,
):
    """
    High-level function to read an allocations CSV (specifying how many synthetic images to create for each label combo),
    then generate the requested captions. This function uses multiprocessing to speed up generation across combinations.

    Args:
        prompts_file (str): Path to a JSON/CSV file containing the prompt templates.
        real_prompt_key (str): Key in the prompts dictionary for real-image-based prompts.
        hypo_prompt_key (str): Key in the prompts dictionary for hypothetical prompts.
        fallback_keys (list): A list of fallback prompt keys to handle potential refusals.
        llm_name (str): Which LLM to call ("claude" or "gpt").
        proportion_real (float): Fraction of total requested images that should come from real images rather than hypothetical.
        alloc_csv (str): CSV file that lists how many synthetic images to allocate per label combination.
        out_csv (str): Where to write all generated caption records (successes and failures).
        verbose (bool): If True, prints status updates and a final summary.
        max_workers (int): The maximum number of multiprocessing workers to use.
        max_rows (int): If specified, limit the number of label combinations processed from the allocations CSV.
        overwrite_output (bool): If True, deletes the existing output CSV before starting.

    Returns:
        None
    """
    configure_logging()
    import pickle

    if overwrite_output and os.path.exists(out_csv):
        if verbose:
            print(f"Deleting existing output file {out_csv} to start fresh.")
        os.remove(out_csv)

    prompts_dict = load_prompt_candidates(prompts_file)
    real_prompt = prompts_dict.get(("real", real_prompt_key), "")
    hypothetical_prompt = prompts_dict.get(("hypothetical", hypo_prompt_key), "")

    if fallback_keys is None:
        fallback_keys = []
    fallback_texts = []
    for fk in fallback_keys:
        fbtxt = prompts_dict.get(("fallback", fk))
        if fbtxt:
            fallback_texts.append(fbtxt)

    df_train = load_medic_train_df()
    df_train_pickled = pickle.dumps(df_train)

    df_alloc = pd.read_csv(alloc_csv)
    if max_rows is not None and max_rows < len(df_alloc):
        df_alloc = df_alloc.iloc[:max_rows].copy()
    n_rows = len(df_alloc)

    if verbose:
        print(f"Processing {n_rows} combos with {max_workers} processes.")
        print(f"Results => {out_csv}")

    total_successes = 0
    total_failures = 0

    from tqdm import tqdm

    # We process each row in the allocation CSV in parallel
    with tqdm(total=n_rows, desc="Generating combos") as combo_bar:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            future_to_idx = {}
            for i, row in df_alloc.iterrows():
                row_dict = row.to_dict()
                fut = executor.submit(
                    process_label_combo,
                    row_dict,
                    df_train_pickled,
                    real_prompt,
                    hypothetical_prompt,
                    fallback_texts,
                    proportion_real,
                    llm_name,
                    out_csv,
                )
                future_to_idx[fut] = i

            for future in concurrent.futures.as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    s_count, f_count = future.result()
                    total_successes += s_count
                    total_failures += f_count
                except Exception as e:
                    if verbose:
                        print(f"[Row {i}] error => {e}")
                combo_bar.update(1)

    if verbose:
        print("Finished all combos.")
        print(f"Successes: {total_successes}, Failures: {total_failures}")
        if (total_successes + total_failures) > 0:
            sr = 100.0 * total_successes / (total_successes + total_failures)
            print(f"Success rate: {sr:.1f}%")
        print(f"Wrote final results to {out_csv}")
