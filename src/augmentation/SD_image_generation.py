"""
SD_image_generation.py

This module provides utilities for generating images via two text-to-image APIs:

1. Stability AI's Stable Diffusion (functions generate_images_for_csv, generate_image_with_stable_diffusion)
2. Black Forest AI's FLUX.1 (functions generate_images_with_black_forest, generate_images_with_black_forest_multiprocess)

The primary goal is to take a CSV of textual descriptions or prompts, send them to the relevant
text-to-image API, and store the resulting images locally. Each function updates its CSV
with new file paths indicating where the generated images were saved.

Features:
- Prompt length truncation (to avoid exceeding API limits).
- Fallback prompts in case of errors or content moderation.
- Error handling and logging to CSV.
- (For Black Forest AI) optional multiprocessing support.
- Tools to merge original training sets with newly generated synthetic data.
"""

import os
import csv
import base64
import requests
import pandas as pd
import json
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm
import math
import concurrent.futures
import multiprocessing
import datetime
import time
import sys


import config

csv.field_size_limit(sys.maxsize)
##############################################################################
#                           GLOBAL CONSTANTS
##############################################################################

# These columns define the format of our error-logging DataFrames and CSVs.
ERROR_LOG_COLUMNS = [
    "row_index",
    "original_prompt",
    "error_type",
    "error_message",
    "http_status",
    "timestamp",
]

# A lock for concurrency in BFS CSV writes, so multiple processes do not overwrite each other.
csv_lock = multiprocessing.Lock()

# Final BFS CSV columns (these are used specifically by BFS workflows).
FIELDNAMES_BF = [
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


##############################################################################
#               STABLE DIFFUSION 1.x / 3.x: IMAGE GENERATION FUNCTIONS
##############################################################################
def get_max_prompt_length(engine_id: str) -> int:
    """
    Retrieve the maximum supported prompt length for a given Stable Diffusion engine.

    Parameters
    ----------
    engine_id : str
        Identifier for the Stable Diffusion engine, e.g. "stable-diffusion-v1-6" or "sd3.5-medium".

    Returns
    -------
    int
        The character limit for the prompt. A heuristic is used: ~2k for SD1.x, ~10k for SD3.x.
    """
    # Heuristic: "sd3" in the engine ID implies a ~10k limit; older ones have ~2k.
    if "sd3" in engine_id.lower():
        return 9900  # Leaves some buffer under 10k
    else:
        return 1900  # Leaves some buffer under 2k


def truncate_prompt(prompt: str, engine_id: str) -> str:
    """
    Truncate a text prompt if it exceeds the maximum supported length for the specified SD engine.

    Parameters
    ----------
    prompt : str
        The original text prompt to be truncated.
    engine_id : str
        A Stable Diffusion engine identifier used to look up length limits.

    Returns
    -------
    str
        Possibly truncated prompt (with '...' appended if truncation occurred).
    """
    max_length = get_max_prompt_length(engine_id)
    if len(prompt) <= max_length:
        return prompt
    return prompt[:max_length] + "..."


def generate_image_with_stable_diffusion(
    prompt: str,
    height: int = 512,
    width: int = 512,
    steps: int = 30,
    cfg_scale: float = 7.0,
    sampler: str = "K_DPMPP_2M",
    seed: int = 0,
    samples: int = 1,
    engine_id: Optional[str] = None,
    verbose: bool = False,  # Added verbose parameter to control output
) -> Tuple[Optional[bytes], Optional[Dict[str, Any]]]:
    """
    Request an image from Stability AI's text-to-image API using the provided prompt
    and generation parameters. Handles both v1 and v2beta API endpoints.

    Parameters
    ----------
    prompt : str
        The textual prompt guiding image generation.
    height : int
        Height of the generated image.
    width : int
        Width of the generated image.
    steps : int
        Number of diffusion steps (inference steps).
    cfg_scale : float
        Classifier-free guidance scale controlling adherence to the prompt.
    sampler : str
        Choice of sampler algorithm (e.g. "K_EULER_ANCESTRAL", "K_DPMPP_2M").
    seed : int
        Random seed for reproducible images. 0 indicates the API chooses a random seed.
    samples : int
        Number of images to generate in a single request; typically 1 for single image.
    engine_id : str, optional
        Identifier for the Stable Diffusion model engine. If None, uses the engine from config.
    verbose : bool, optional
        If True, print detailed information about the API calls. Default is False.

    Returns
    -------
    (image_data, error_dict) : (Optional[bytes], Optional[Dict[str, Any]])
        * image_data: The raw image data (in bytes) if successful, or None on error.
        * error_dict: A dictionary with error details if something went wrong, or None on success.
    """
    import datetime
    import requests
    import base64

    # If no engine ID is provided, use the default from config
    if engine_id is None:
        engine_id = config.STABLE_DIFFUSION_MODEL

    # Retrieve the user's Stability API key
    api_key = config.STABLE_DIFFUSION_KEY
    if not api_key:
        return None, {
            "error_type": "Configuration Error",
            "error_message": "Stability API key is missing in config.STABLE_DIFFUSION_KEY.",
            "http_status": None,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    # Determine if we should use v1 or v2beta API based on engine_id
    use_v2beta = any(id_prefix in engine_id.lower() for id_prefix in ["sd3", "sd3.5"])

    if use_v2beta:
        # For SD3/SD3.5, use the v2beta API
        if verbose:
            print(f"Using v2beta API for {engine_id}")

        # SD3/SD3.5 always uses the same endpoint regardless of the specific model
        api_host = "https://api.stability.ai"
        url = f"{api_host}/v2beta/stable-image/generate/sd3"

        # Set up parameters using the format that worked in our test
        output_format = "png"

        # Create form data - only include non-default parameters
        form_data = {
            "prompt": prompt,
            "output_format": output_format,
        }

        if height != 512:
            form_data["height"] = str(height)
        if width != 512:
            form_data["width"] = str(width)
        if steps != 30:
            form_data["steps"] = str(steps)
        if cfg_scale != 7.0:
            form_data["cfg_scale"] = str(cfg_scale)
        if seed != 0:
            form_data["seed"] = str(seed)

        # Set up headers exactly as they appear in the docs
        headers = {"authorization": f"Bearer {api_key}", "accept": "image/*"}

        try:
            # Make the API request
            response = requests.post(
                url,
                headers=headers,
                files={"none": ""},
                data=form_data,
                timeout=180,
            )

            # Check for errors
            if response.status_code != 200:
                error_message = response.text
                try:
                    error_json = response.json()
                    if isinstance(error_json, dict) and "message" in error_json:
                        error_message = error_json["message"]
                except:
                    pass

                return None, {
                    "error_type": "API Error",
                    "error_message": error_message,
                    "http_status": response.status_code,
                    "timestamp": datetime.datetime.now().isoformat(),
                }

            # For v2beta, the response is the image bytes directly
            return response.content, None

        except Exception as e:
            return None, {
                "error_type": "Request Failed",
                "error_message": str(e),
                "http_status": None,
                "timestamp": datetime.datetime.now().isoformat(),
            }
    else:
        # For older models like SD1.6 and SDXL, use the v1 API
        if verbose:
            print(f"Using v1 API for {engine_id}")

        # Apply length-based truncation to the prompt for v1 API
        safe_prompt = truncate_prompt(prompt, engine_id)

        # Construct the request payload
        api_host = "https://api.stability.ai"
        payload = {
            "text_prompts": [{"text": safe_prompt}],
            "cfg_scale": cfg_scale,
            "height": height,
            "width": width,
            "samples": samples,
            "steps": steps,
            "sampler": sampler,
            "seed": seed,
        }

        # Set up the request headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        # Make the POST request to the Stability API
        try:
            response = requests.post(
                f"{api_host}/v1/generation/{engine_id}/text-to-image",
                headers=headers,
                json=payload,
                timeout=180,  # seconds
            )
        except requests.exceptions.RequestException as e:
            # Any connectivity or HTTP-level failure
            return None, {
                "error_type": "Request Failed",
                "error_message": str(e),
                "http_status": None,
                "timestamp": datetime.datetime.now().isoformat(),
            }

        # If the response status code is not 200, treat it as an error
        if response.status_code != 200:
            error_message = response.text
            # Attempt to parse JSON and extract a user-friendly message
            try:
                error_json = response.json()
                if isinstance(error_json, dict) and "message" in error_json:
                    error_message = error_json["message"]
            except:
                pass
            return None, {
                "error_type": "API Error",
                "error_message": error_message,
                "http_status": response.status_code,
                "timestamp": datetime.datetime.now().isoformat(),
            }

        # Parse the JSON response to extract the generated image
        try:
            data = response.json()
            artifacts = data.get("artifacts", [])
            if not artifacts:
                return None, {
                    "error_type": "No Artifacts",
                    "error_message": "No artifacts received from the Stability API.",
                    "http_status": 200,
                    "timestamp": datetime.datetime.now().isoformat(),
                }

            base64_img = artifacts[0].get("base64")
            if not base64_img:
                return None, {
                    "error_type": "No Base64 Data",
                    "error_message": "The first artifact lacks base64 image data.",
                    "http_status": 200,
                    "timestamp": datetime.datetime.now().isoformat(),
                }

            # Decode the base64-encoded image data
            return base64.b64decode(base64_img), None

        except Exception as e:
            # Could be JSON parsing, KeyError, or base64 decode error, etc.
            return None, {
                "error_type": "Processing Error",
                "error_message": str(e),
                "http_status": 200,
                "timestamp": datetime.datetime.now().isoformat(),
            }


def log_error(
    error_log_df: pd.DataFrame, row_index: int, prompt: str, the_error: Dict[str, Any]
) -> pd.DataFrame:
    """
    Append a new error entry to the given DataFrame and return the updated DataFrame.

    Parameters
    ----------
    error_log_df : pd.DataFrame
        A DataFrame for accumulating errors, with columns defined by ERROR_LOG_COLUMNS.
    row_index : int
        The row index in the original dataset where this error occurred.
    prompt : str
        The text prompt that triggered the error.
    the_error : Dict[str, Any]
        A dictionary describing the error (error_type, error_message, HTTP status, etc.).

    Returns
    -------
    pd.DataFrame
        The updated error_log_df with the new error appended.
    """
    if error_log_df is None:
        error_log_df = pd.DataFrame(columns=ERROR_LOG_COLUMNS)

    new_row = pd.DataFrame(
        [
            {
                "row_index": row_index,
                "original_prompt": prompt,
                "error_type": the_error.get("error_type", "Unknown"),
                "error_message": the_error.get("error_message", "No message"),
                "http_status": the_error.get("http_status", None),
                "timestamp": the_error.get("timestamp", None),
            }
        ]
    )
    return pd.concat([error_log_df, new_row], ignore_index=True)


def generate_images_for_csv(
    csv_in: str,
    csv_out: str,
    height: int = 512,
    width: int = 512,
    steps: int = 30,
    cfg_scale: float = 7.0,
    sampler: str = "K_DPMPP_2M",
    seed: int = 0,
    no_save_overwrite: bool = False,
    error_log_csv: Optional[str] = None,
    fallback_prompt: Optional[str] = "enable",
    engine_id: Optional[str] = None,
) -> None:
    """
    Generate images from textual prompts in a CSV file using Stability AI's API,
    saving images locally, and recording their file paths in the CSV.

    Images are stored under config.SYNTHETIC_DATA_DIR / <model_engine_id>.

    Parameters
    ----------
    csv_in : str
        Path to the input CSV, which must contain a column 'caption_text'.
    csv_out : str
        Where to write the updated CSV (can be the same as csv_in).
    height : int
        Image height (multiple of 64, within [320..1536]).
    width : int
        Image width (multiple of 64, within [320..1536]).
    steps : int
        Number of diffusion sampling steps.
    cfg_scale : float
        Classifier-free guidance scale for controlling adherence to the prompt.
    sampler : str
        Diffusion sampler identifier, e.g. "K_EULER_ANCESTRAL".
    seed : int
        RNG seed (0 => random seed chosen by the API).
    no_save_overwrite : bool
        If True, do not regenerate images for rows that already have a file in 'aug_img_path'.
    error_log_csv : str, optional
        If provided, errors encountered during generation are appended to this CSV file.
    fallback_prompt : str, optional
        If provided (non-None), a second attempt is made with a “safer” fallback prompt
        if the original prompt fails or is moderated. If None, no fallback attempt is made.
    engine_id : str, optional
        Identifier for the stable diffusion engine. If None, uses config.STABLE_DIFFUSION_MODEL.

    Returns
    -------
    None
        Updates csv_out with new columns:
          * 'aug_img_path' - file path of the newly generated image (or "CONTENT_REFUSAL" on failure).
          * 'used_fallback' - whether a fallback prompt was used.
    """
    import os
    import csv
    import pandas as pd
    from tqdm import tqdm
    import datetime

    # Ensure the synthetic data directory exists
    os.makedirs(config.SYNTHETIC_DATA_DIR, exist_ok=True)

    # Read the input CSV, attempting ',' then '\t' delimiters
    try:
        df = pd.read_csv(csv_in, sep=",", quoting=csv.QUOTE_MINIMAL)
    except:
        df = pd.read_csv(csv_in, sep="\t", quoting=csv.QUOTE_MINIMAL)

    # Initialise output columns if they do not already exist
    if "aug_img_path" not in df.columns:
        df["aug_img_path"] = ""
    if "used_fallback" not in df.columns:
        df["used_fallback"] = False

    # If no_save_overwrite is True, skip rows that already have 'aug_img_path' set
    if no_save_overwrite:
        to_process = df[df["aug_img_path"].isna() | (df["aug_img_path"] == "")]
        skip_count = len(df) - len(to_process)
        if skip_count > 0:
            print(f"Skipping {skip_count} rows that already have images.")
    else:
        to_process = df

    success_count = 0
    fallback_success_count = 0
    failure_count = 0
    skip_count = 0
    error_log_df = pd.DataFrame(columns=ERROR_LOG_COLUMNS)

    # Determine model engine for folder naming
    model_engine_id = engine_id if engine_id else config.STABLE_DIFFUSION_MODEL
    # Create subfolder in the synthetic directory named after the model
    model_subdir_path = os.path.join(config.SYNTHETIC_DATA_DIR, model_engine_id)
    os.makedirs(model_subdir_path, exist_ok=True)

    print(f"Generating {len(to_process)} images with Stable Diffusion...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        existing_path = str(row.get("aug_img_path", "")).strip()
        # If skipping overwrites and an image already exists, skip
        if no_save_overwrite and existing_path:
            skip_count += 1
            continue

        prompt_text = str(row.get("caption_text", "")).strip()
        if not prompt_text:
            # If we have no prompt text, log an error, mark as refusal
            error_info = {
                "error_type": "Empty Prompt",
                "error_message": "Caption text is empty",
                "http_status": None,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            error_log_df = log_error(error_log_df, idx, prompt_text, error_info)
            df.at[idx, "aug_img_path"] = "CONTENT_REFUSAL"
            failure_count += 1
            continue

        # Attempt to generate an image with the user-supplied prompt
        image_data, error_info = generate_image_with_stable_diffusion(
            prompt=prompt_text,
            height=height,
            width=width,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler=sampler,
            seed=seed,
            samples=1,
            engine_id=model_engine_id,
        )

        used_fallback = False
        # If first attempt fails and fallback_prompt is set, try again
        if image_data is None and fallback_prompt:
            # Log the original failure
            if error_info:
                error_log_df = log_error(error_log_df, idx, prompt_text, error_info)

            # Construct a fallback prompt
            fallback_text = (
                f"Academic disaster response research. Create an image of: {prompt_text} "
                f"Avoid gore/explicit content but maintain disaster context. "
                f"For CNN training purposes only. Show disaster elements clearly but policy-compliant."
            )

            # Attempt generation using fallback prompt
            image_data, fallback_error_info = generate_image_with_stable_diffusion(
                prompt=truncate_prompt(fallback_text, model_engine_id),
                height=height,
                width=width,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler=sampler,
                seed=seed,
                samples=1,
                engine_id=model_engine_id,
            )
            if image_data is None and fallback_error_info:
                error_log_df = log_error(
                    error_log_df, idx, f"FALLBACK: {fallback_text}", fallback_error_info
                )
            used_fallback = image_data is not None

        # If still no image, mark refusal
        if image_data is None:
            if not fallback_prompt and error_info:
                # If we have no fallback, log the original error
                error_log_df = log_error(error_log_df, idx, prompt_text, error_info)

            df.at[idx, "aug_img_path"] = "CONTENT_REFUSAL"
            if error_info:
                error_type = error_info.get("error_type", "Unknown")
                df.at[idx, "aug_img_path"] = f"CONTENT_REFUSAL: {error_type}"
            failure_count += 1
            continue

        # If we have valid image data, write it to disk in the subfolder
        out_filename = f"sdimg_{idx}.png"
        out_path = os.path.join(model_subdir_path, out_filename)
        try:
            with open(out_path, "wb") as f:
                f.write(image_data)

            df.at[idx, "aug_img_path"] = out_path
            df.at[idx, "used_fallback"] = used_fallback

            if used_fallback:
                fallback_success_count += 1
            else:
                success_count += 1

        except OSError as e:
            # Could not write the file to disk
            error_info = {
                "error_type": "File Write Error",
                "error_message": str(e),
                "http_status": None,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            error_log_df = log_error(error_log_df, idx, prompt_text, error_info)
            df.at[idx, "aug_img_path"] = "CONTENT_REFUSAL"
            failure_count += 1
            continue

    # Save error log if requested
    if error_log_csv and not error_log_df.empty:
        error_log_df.to_csv(error_log_csv, index=False)
        print(f"Error log saved to: {error_log_csv}")

    # Output the updated DataFrame to CSV, preserving delimiter style
    if "," in open(csv_in, "r").readline():
        df.to_csv(csv_out, index=False, sep=",")
    else:
        df.to_csv(csv_out, index=False, sep="\t")

    # Print summary for user feedback
    print(f"\nImage generation complete (engine={model_engine_id}):")
    print(f"  • Success (original prompt): {success_count} images")
    if fallback_prompt:
        print(f"  • Success (fallback prompt): {fallback_success_count} images")
    print(f"  • Failed/Refused: {failure_count} images")
    print(f"  • Skipped: {skip_count} images")

    content_refusals = df[
        df["aug_img_path"].str.contains("CONTENT_REFUSAL", na=False)
    ].shape[0]
    if content_refusals > 0:
        print(f"  • Content refusals: {content_refusals} images")

    if error_log_csv:
        print(f"  • Errors logged to: {error_log_csv}")

    print(f"Saved images to: {model_subdir_path}")
    print(f"Output CSV saved to: {csv_out}")


##############################################################################
#                 BLACK FOREST AI (FLUX.1) IMAGE GENERATION FUNCTIONS
##############################################################################
def get_max_prompt_length_bf() -> int:
    """
    Return the maximum prompt length for Black Forest AI's FLUX.1 service.

    This is a placeholder that might be enforced more strictly in the future.

    Returns
    -------
    int
        The maximum length of prompt text the BFS API is expected to handle.
    """
    return 10000  # Large prompt limit, as described by FLUX.1 docs.


def truncate_prompt_bf(prompt: str) -> str:
    """
    Truncate a text prompt if it exceeds the maximum length for FLUX.1.
    (Currently not invoked by default BFS calls, but available for use if needed.)

    Parameters
    ----------
    prompt : str
        The original text prompt to be truncated for BFS.

    Returns
    -------
    str
        Possibly truncated prompt (with '...' appended if truncation occurred).
    """
    max_length = get_max_prompt_length_bf()
    if len(prompt) <= max_length:
        return prompt
    return prompt[:max_length] + "..."


import os
import csv
import time
import math
import base64
import random
import datetime
import requests
import pandas as pd
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

# Constants
ERROR_LOG_COLUMNS = [
    "row_index",
    "original_prompt",
    "error_type",
    "error_message",
    "http_status",
    "timestamp",
]

FIELDNAMES_BF = [
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

# Global lock for thread-safe CSV writing
import threading

csv_lock = threading.Lock()


def generate_image_with_black_forest(
    prompt: str,
    height: int = 768,
    width: int = 1024,
    steps: int = 28,
    guidance: float = 3.0,
    safety_tolerance: int = 2,
    seed: Optional[int] = None,
    output_format: str = "png",
    api_key: Optional[str] = None,
) -> Tuple[Optional[bytes], Optional[Dict[str, Any]]]:
    """
    Calls Black Forest AI's FLUX.1 text-to-image API (which may be asynchronous).
    Returns (image_data, error_dict) upon completion or failure.

    Parameters
    ----------
    prompt : str
        The textual prompt describing what should be generated.
    height : int
        Image height in pixels (forced to a multiple of 32, clamped to [256..1440]).
    width : int
        Image width in pixels (forced to a multiple of 32, clamped to [256..1440]).
    steps : int
        Diffusion or sampling steps for the BFS model.
    guidance : float
        Guidance strength controlling adherence to the prompt.
    safety_tolerance : int
        Safety moderation tolerance for BFS content filters (0..6).
    seed : int, optional
        Random seed for reproducibility (if provided).
    output_format : str
        Expected file format (e.g. "png" or "jpg").
    api_key : str, optional
        Black Forest AI API key. If not provided, attempts to load from config.

    Returns
    -------
    (image_data, error_dict) : (Optional[bytes], Optional[Dict[str, Any]])
        * image_data: Raw bytes of the generated image (PNG/JPG) if successful, else None.
        * error_dict: A dictionary of error details if something went wrong, else None.
    """
    import datetime

    # Force multiples of 32 for BFS
    height = max(256, min(1440, (height // 32) * 32))
    width = max(256, min(1440, (width // 32) * 32))

    # Attempt to retrieve BFS key from config if none given
    if api_key is None or not api_key:
        try:
            api_key = config.BLACK_FOREST_KEY
        except:
            return None, {
                "error_type": "Configuration Error",
                "error_message": "No BFS key found in config.",
                "timestamp": datetime.datetime.now().isoformat(),
            }
    if not api_key:
        return None, {
            "error_type": "Configuration Error",
            "error_message": "No BFS key found.",
            "timestamp": datetime.datetime.now().isoformat(),
        }

    # Prepare the payload for BFS generation
    payload = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "steps": steps,
        "guidance": guidance,
        "safety_tolerance": safety_tolerance,
        "output_format": output_format,
    }
    if seed is not None:
        payload["seed"] = seed

    api_url = "https://api.us1.bfl.ai/v1/flux-dev"
    headers = {"Content-Type": "application/json", "x-key": api_key}

    # Send a POST request to BFS
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=180)
    except requests.exceptions.RequestException as e:
        # Networking or other I/O errors
        return None, {
            "error_type": "Request Failed",
            "error_message": str(e),
            "timestamp": datetime.datetime.now().isoformat(),
        }

    # If BFS does not return 200, treat that as an error
    if response.status_code != 200:
        err_msg = response.text
        try:
            j = response.json()
            if isinstance(j, dict) and "detail" in j:
                err_msg = j["detail"]
        except:
            pass
        return None, {
            "error_type": "API Error",
            "error_message": err_msg,
            "http_status": response.status_code,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    # Try to parse the BFS response as JSON
    try:
        data = response.json()
    except Exception as e:
        return None, {
            "error_type": "Processing Error",
            "error_message": f"Could not parse BFS JSON: {str(e)}",
            "timestamp": datetime.datetime.now().isoformat(),
        }

    # BFS may return an async task with an ID; poll for results
    if "id" in data:
        task_id = data["id"]
        polling_url = data.get(
            "polling_url", f"https://api.us1.bfl.ai/v1/get_result?id={task_id}"
        )
        max_attempts = 30
        for attempt in range(max_attempts):
            time.sleep(10)  # Wait 10 seconds between polls
            try:
                st_resp = requests.get(polling_url, headers=headers, timeout=60)
            except requests.exceptions.RequestException as e:
                return None, {
                    "error_type": "Status Check Failed",
                    "error_message": str(e),
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            if st_resp.status_code != 200:
                return None, {
                    "error_type": "Status Check Error",
                    "error_message": st_resp.text,
                    "http_status": st_resp.status_code,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            st_data = st_resp.json()
            status = st_data.get("status")
            if status == "Ready":
                image_url = st_data.get("result", {}).get("sample")
                if not image_url:
                    return None, {
                        "error_type": "Missing Image URL",
                        "error_message": "Task completed but sample not found",
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                try:
                    img_resp = requests.get(image_url, timeout=60)
                except requests.exceptions.RequestException as e:
                    return None, {
                        "error_type": "Image Download Failed",
                        "error_message": str(e),
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                if img_resp.status_code != 200:
                    return None, {
                        "error_type": "Image Download Error",
                        "error_message": img_resp.text,
                        "http_status": img_resp.status_code,
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                return img_resp.content, None
            elif status in ["Error", "Request Moderated", "Content Moderated"]:
                return None, {
                    "error_type": f"Task {status}",
                    "error_message": f"Task ended with status {status}",
                    "timestamp": datetime.datetime.now().isoformat(),
                }
        # If we exhaust max_attempts, assume BFS timed out
        return None, {
            "error_type": "Timeout",
            "error_message": f"BFS not ready after {max_attempts*10} seconds",
            "timestamp": datetime.datetime.now().isoformat(),
        }

    # If BFS returned a synchronous result:
    if "image" in data:
        raw_img = data["image"]
        if raw_img.startswith("data:image"):
            base64_part = raw_img.split(",", 1)[-1]
        else:
            base64_part = raw_img
        try:
            return base64.b64decode(base64_part), None
        except Exception as e:
            return None, {
                "error_type": "Processing Error",
                "error_message": f"Decode base64 error: {str(e)}",
                "timestamp": datetime.datetime.now().isoformat(),
            }
    elif "image_url" in data:
        try:
            img_resp = requests.get(data["image_url"], timeout=60)
        except requests.exceptions.RequestException as e:
            return None, {
                "error_type": "Image Download Failed",
                "error_message": str(e),
                "timestamp": datetime.datetime.now().isoformat(),
            }
        if img_resp.status_code != 200:
            return None, {
                "error_type": "Image Download Error",
                "error_message": img_resp.text,
                "http_status": img_resp.status_code,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        return img_resp.content, None

    # If none of the above conditions are met, BFS returned something unexpected
    return None, {
        "error_type": "No Image Data",
        "error_message": "Response had neither 'id', 'image', nor 'image_url'.",
        "timestamp": datetime.datetime.now().isoformat(),
    }


def log_error(
    error_log_df: pd.DataFrame, row_index: int, prompt: str, the_error: Dict[str, Any]
) -> pd.DataFrame:
    """
    Appends one error record to error_log_df. (Duplicate name as used by SD code, but the BFS
    pipeline also needs to log errors in the same format or a similar format.)

    Parameters
    ----------
    error_log_df : pd.DataFrame
        DataFrame with columns = ERROR_LOG_COLUMNS.
    row_index : int
        Index or ID of the row that generated the error.
    prompt : str
        The textual prompt that produced this error.
    the_error : Dict[str, Any]
        Dictionary describing the error (type, message, HTTP status, etc.).

    Returns
    -------
    pd.DataFrame
        The updated error_log_df with the new error appended.
    """
    if error_log_df is None:
        error_log_df = pd.DataFrame(columns=ERROR_LOG_COLUMNS)

    entry = {
        "row_index": row_index,
        "original_prompt": prompt,
        "error_type": the_error.get("error_type", "Unknown"),
        "error_message": the_error.get("error_message", "No message"),
        "http_status": the_error.get("http_status", None),
        "timestamp": the_error.get("timestamp", None),
    }
    error_log_df = pd.concat([error_log_df, pd.DataFrame([entry])], ignore_index=True)
    return error_log_df


def append_records_to_csv(csv_path: str, records: list):
    """
    Immediately append a list of row-dictionaries to 'csv_path' in a concurrency-safe manner,
    with strict handling of multiline fields.

    Each row in 'records' must contain exactly the columns in FIELDNAMES_BF, or a mismatch occurs.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file to which the rows will be appended.
    records : list of dict
        Each dict representing a row, must have exactly the keys in FIELDNAMES_BF.
    """
    if not records:
        return

    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Check if this is a new file or if we're appending
    is_new_file = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0

    # Ensure all values are strings to avoid type issues
    sanitized_records = []
    for row in records:
        clean_row = {}
        for key in FIELDNAMES_BF:
            value = row.get(key, "")
            clean_row[key] = str(value) if value is not None else ""
        sanitized_records.append(clean_row)

    # Use a temporary file to avoid issues with concurrent writes
    temp_path = f"{csv_path}.{time.time()}.{random.randint(1000, 9999)}.tmp"

    try:
        with csv_lock:
            # Write to a temporary file first
            with open(temp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=FIELDNAMES_BF,
                    delimiter="\t",  # Use tab as delimiter consistently
                    quoting=csv.QUOTE_ALL,  # Quote everything for safety
                    quotechar='"',
                    escapechar='"',
                )

                if is_new_file:
                    writer.writeheader()

                for row in sanitized_records:
                    writer.writerow(row)

            # If this is the first write, just rename the temp file
            if is_new_file:
                os.replace(temp_path, csv_path)
            else:
                # Otherwise append its contents (excluding header) to the main file
                with open(temp_path, "r", newline="", encoding="utf-8") as temp_file:
                    lines = temp_file.readlines()

                    # Skip header line if this is not a new file
                    start_line = 1 if len(lines) > 0 else 0

                    if len(lines) > start_line:
                        with open(
                            csv_path, "a", newline="", encoding="utf-8"
                        ) as main_file:
                            main_file.writelines(lines[start_line:])

                # Remove the temp file after appending
                os.remove(temp_path)
    except Exception as e:
        # Clean up the temp file if something goes wrong
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise e


# The critical fix: Increase the CSV field size limit to handle large caption fields
csv.field_size_limit(sys.maxsize)  # Set to maximum allowed by the system


def generate_images_with_black_forest(
    csv_in: str,
    csv_out: str = None,
    height: int = 512,
    width: int = 512,
    steps: int = 30,
    guidance: float = 2.5,
    safety_tolerance: int = 6,
    seed: Optional[int] = None,
    output_format: str = "png",
    no_save_overwrite: bool = False,
    error_log_csv: Optional[str] = None,
    fallback_prompt: Optional[str] = "enable",
    api_key: Optional[str] = None,
    max_workers: int = 8,
) -> None:
    """
    Multiprocessing BFS generation. The CSV is split into chunks, each processed in parallel.
    Images are stored under config.SYNTHETIC_DATA_DIR / "bfai".
    """
    # Set default output path if none provided
    if csv_out is None:
        output_dir = "src/augmentation/model_comparison_outputs"
        os.makedirs(output_dir, exist_ok=True)
        csv_out = os.path.join(output_dir, "test_captions_bfai.csv")

    # Set default error log path if none provided
    if error_log_csv is None:
        error_log_dir = os.path.dirname(csv_out)
        error_log_csv = os.path.join(error_log_dir, "test_captions_bfai_errors.csv")

    synthetic_data_dir = config.SYNTHETIC_DATA_DIR
    os.makedirs(synthetic_data_dir, exist_ok=True)

    # CRITICAL FIX: Properly read the input CSV with correct handling of multiline fields
    print(f"Reading input CSV file: {csv_in}")
    try:
        # Open with proper settings to handle multiline fields
        with open(csv_in, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_ALL)
            # Load all rows into memory with proper field handling
            rows = list(reader)

        print(f"Successfully read {len(rows)} rows from input CSV")

        # Convert to DataFrame for further processing
        df = pd.DataFrame(rows)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print("Attempting alternative reading method...")

        # Fallback to pandas if the csv module has issues
        df = pd.read_csv(
            csv_in,
            sep="\t",
            quoting=csv.QUOTE_ALL,
            encoding="utf-8",
            engine="python",  # Python engine is more robust for complex CSV files
        )
        print(f"Successfully read {len(df)} rows using pandas")

    # Create output directory if needed
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    # Create fresh output CSV with correct headers
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=FIELDNAMES_BF, delimiter="\t", quoting=csv.QUOTE_ALL
        )
        writer.writeheader()

    # Ensure expected columns exist
    if "aug_img_path" not in df.columns:
        df["aug_img_path"] = ""
    if "used_fallback" not in df.columns:
        df["used_fallback"] = False
    if "caption_text" not in df.columns:
        df["caption_text"] = ""

    # Filter out rows to process if no_save_overwrite is True
    if no_save_overwrite:
        df_to_process = df[df["aug_img_path"].isna() | (df["aug_img_path"] == "")]
    else:
        df_to_process = df

    n_rows = len(df_to_process)
    print(f"Found {n_rows} rows to process in parallel with {max_workers} processes.")
    if n_rows == 0:
        print("No rows require BFS generation. Exiting.")
        return

    # Break rows into chunks for parallel processing
    indexes_to_process = df_to_process.index.tolist()
    chunk_size = max(1, math.ceil(n_rows / max_workers))
    chunks = [
        indexes_to_process[i : i + chunk_size]
        for i in range(0, len(indexes_to_process), chunk_size)
    ]

    print(
        f"Split data into {len(chunks)} chunks of approximately {chunk_size} rows each"
    )

    partial_results = []
    error_dfs = []

    # Execute BFS generation in parallel
    with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            future_to_idx = {}
            for i, cindexes in enumerate(chunks):
                df_chunk = df.loc[cindexes].copy()
                if "aug_img_path" in df_chunk.columns:
                    df_chunk["aug_img_path"] = df_chunk["aug_img_path"].astype(str)
                chunk_start_idx = cindexes[0] if cindexes else 0

                # Use the modified _process_csv_rows_bf function with fixed image folder
                fut = executor.submit(
                    _process_csv_rows_bf,
                    df_chunk,
                    chunk_start_idx,
                    csv_out,
                    height,
                    width,
                    steps,
                    guidance,
                    safety_tolerance,
                    seed,
                    output_format,
                    no_save_overwrite,
                    fallback_prompt,
                    api_key,
                    synthetic_data_dir,
                )
                future_to_idx[fut] = i

            # Collect results from each completed chunk
            for fut in concurrent.futures.as_completed(future_to_idx):
                i = future_to_idx[fut]
                try:
                    updated_chunk, error_df = fut.result()
                    partial_results.append(updated_chunk)
                    if not error_df.empty:
                        error_dfs.append(error_df)
                except Exception as e:
                    print(f"[Chunk {i}] encountered error: {str(e)}")
                pbar.update(1)

    # Merge partial results
    if partial_results:
        updated_df = pd.concat(partial_results, axis=0)
        for idx in updated_df.index:
            if idx in df.index:
                df.loc[idx, updated_df.columns] = updated_df.loc[idx]
    else:
        updated_df = df

    # Merge error logs
    if error_dfs:
        final_error_df = pd.concat(error_dfs, axis=0)
    else:
        final_error_df = pd.DataFrame(columns=ERROR_LOG_COLUMNS)

    # Write error log if provided
    if error_log_csv and not final_error_df.empty:
        final_error_df.to_csv(
            error_log_csv, index=False, sep="\t", quoting=csv.QUOTE_ALL
        )
        print(f"Error log saved to: {error_log_csv}")

    print(f"Done. Wrote updated BFS CSV to {csv_out}.")
    print(f"Images were saved to: {os.path.join(synthetic_data_dir, 'bfai')}")


import os
import sys
import csv
import math
import time
import random
import pandas as pd
import datetime
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

# Ensure large fields are allowed
csv.field_size_limit(sys.maxsize)

ERROR_LOG_COLUMNS = [
    "row_index",
    "original_prompt",
    "error_type",
    "error_message",
    "http_status",
    "timestamp",
]
FIELDNAMES_BF = [
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

import threading

csv_lock = threading.Lock()


import os
import csv
import math
import time
import random
import pandas as pd
import datetime
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

csv.field_size_limit(csv.field_size_limit())  # Ensure large fields are allowed

ERROR_LOG_COLUMNS = [
    "row_index",
    "original_prompt",
    "error_type",
    "error_message",
    "http_status",
    "timestamp",
]
FIELDNAMES_BF = [
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

import threading
csv_lock = threading.Lock()


def generate_images_with_black_forest(
    csv_in: str,
    csv_out: str = None,
    height: int = 512,
    width: int = 512,
    steps: int = 30,
    guidance: float = 2.5,
    safety_tolerance: int = 6,
    seed: Optional[int] = None,
    output_format: str = "png",
    no_save_overwrite: bool = False,
    error_log_csv: Optional[str] = None,
    fallback_prompt: Optional[str] = "enable",
    api_key: Optional[str] = None,
    max_workers: int = 8,
) -> None:
    """
    Reads an input CSV (in whatever format), generates images, 
    and writes a tab-delimited output CSV with QUOTES AROUND EVERY FIELD
    (including multiline 'llm_output' / 'caption_text'). That way you avoid
    losing multiline data in subsequent processing, and also have each column
    header in its own tab-separated column.
    """
    if csv_out is None:
        output_dir = "src/augmentation/model_comparison_outputs"
        os.makedirs(output_dir, exist_ok=True)
        csv_out = os.path.join(output_dir, "test_captions_bfai.tsv")

    if error_log_csv is None:
        error_log_dir = os.path.dirname(csv_out)
        error_log_csv = os.path.join(error_log_dir, "test_captions_bfai_errors.tsv")

    synthetic_data_dir = config.SYNTHETIC_DATA_DIR
    os.makedirs(synthetic_data_dir, exist_ok=True)

    # Example read: if your input is also tab-delimited with multiline quotes:
    # adjust as necessary. The important point is we just read it into df somehow:
    print(f"Reading input CSV file: {csv_in}")
    try:
        df = pd.read_csv(
            csv_in,
            sep=",",            # or sep="\t", or sep=None for sniffing—whatever your file is
            engine="python",
            on_bad_lines="warn",
            keep_default_na=False,
            encoding="utf-8",
        )
        print(f"Successfully read {len(df)} rows from input CSV")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Reindex to ensure only the 13 columns we expect
    df = df.reindex(columns=FIELDNAMES_BF, fill_value="")

    # Possibly skip rows with existing images
    if no_save_overwrite:
        df_to_process = df[df["aug_img_path"].isna() | (df["aug_img_path"] == "")]
    else:
        df_to_process = df

    n_rows = len(df_to_process)
    print(f"Found {n_rows} rows to process in parallel with {max_workers} processes.")
    if n_rows == 0:
        print("No rows need BFS generation. Exiting.")
        return

    # --- Write a tab-delimited CSV, QUOTE_ALL ensures multiline text is kept as one field. ---
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES_BF,
            delimiter="\t",          # separate columns with tabs
            quotechar='"',           # use standard double quotes
            quoting=csv.QUOTE_ALL,   # wrap *every* field in quotes
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()

    # Chunk the rows for parallel processing
    indexes_to_process = df_to_process.index.tolist()
    chunk_size = max(1, math.ceil(n_rows / max_workers))
    chunks = [
        indexes_to_process[i : i + chunk_size]
        for i in range(0, len(indexes_to_process), chunk_size)
    ]

    partial_results = []
    error_dfs = []

    # BFS generation in parallel
    with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for i, cindexes in enumerate(chunks):
                df_chunk = df.loc[cindexes].copy()

                fut = executor.submit(
                    _process_csv_rows_bf,
                    df_chunk,
                    cindexes[0] if cindexes else 0,
                    csv_out,
                    height,
                    width,
                    steps,
                    guidance,
                    safety_tolerance,
                    seed,
                    output_format,
                    no_save_overwrite,
                    fallback_prompt,
                    api_key,
                    synthetic_data_dir,
                )
                future_map[fut] = i

            for fut in concurrent.futures.as_completed(future_map):
                i = future_map[fut]
                try:
                    updated_chunk, error_df = fut.result()
                    partial_results.append(updated_chunk)
                    if not error_df.empty:
                        error_dfs.append(error_df)
                except Exception as exc:
                    print(f"[Chunk {i}] encountered error: {str(exc)}")
                pbar.update(1)

    # Merge partial results
    if partial_results:
        updated_df = pd.concat(partial_results, axis=0)
        for idx in updated_df.index:
            if idx in df.index:
                df.loc[idx, updated_df.columns] = updated_df.loc[idx]
    else:
        updated_df = df

    # Merge errors
    if error_dfs:
        final_error_df = pd.concat(error_dfs, axis=0)
    else:
        final_error_df = pd.DataFrame(columns=ERROR_LOG_COLUMNS)

    # If we have errors, write them out (also tab-delimited, quote-all)
    if error_log_csv and not final_error_df.empty:
        final_error_df.to_csv(
            error_log_csv,
            index=False,
            sep="\t",
            quotechar='"',
            quoting=csv.QUOTE_ALL,
            line_terminator="\n",
        )
        print(f"Error log saved to: {error_log_csv}")

    print(f"Done. Wrote updated BFS CSV to {csv_out}.")
    print(f"Images were saved to: {os.path.join(synthetic_data_dir, 'bfai')}")


def _process_csv_rows_bf(
    df_chunk: pd.DataFrame,
    chunk_start_idx: int,
    csv_out: str,
    height: int,
    width: int,
    steps: int,
    guidance: float,
    safety_tolerance: int,
    seed: Optional[int],
    output_format: str,
    no_save_overwrite: bool,
    fallback_prompt: Optional[str],
    api_key: Optional[str],
    synthetic_data_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Worker process for BFS generation in multiprocess mode.
    Images are stored under synthetic_data_dir / "bfai".
    """
    error_log_df = pd.DataFrame(columns=ERROR_LOG_COLUMNS)

    # Always use a single consistent folder name for all images
    bfs_model_name = "bfai"  # Fixed folder name for diffusion model output
    bfs_model_dir = os.path.join(synthetic_data_dir, bfs_model_name)
    os.makedirs(bfs_model_dir, exist_ok=True)

    for local_idx in range(len(df_chunk)):
        row_idx = df_chunk.index[local_idx]
        row = df_chunk.iloc[local_idx]

        # Skip if we already have an image and no_save_overwrite is True
        existing_path = str(row.get("aug_img_path", "")).strip()
        if no_save_overwrite and existing_path:
            continue

        # Get the caption text (important to get the full text properly)
        prompt_text = str(row.get("caption_text", "")).strip()
        # If empty, try llm_output as fallback
        if not prompt_text and "llm_output" in row:
            prompt_text = str(row.get("llm_output", "")).strip()

        if not prompt_text:
            # Empty prompt => error
            err_info = {
                "error_type": "Empty Prompt",
                "error_message": "No caption_text or llm_output found",
                "http_status": None,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            error_log_df = log_error(error_log_df, row_idx, prompt_text, err_info)
            df_chunk.at[row_idx, "aug_img_path"] = "CONTENT_REFUSAL:EmptyPrompt"

            # Build row data for output
            row_data = {
                "image_path": row.get("image_path", ""),
                "damage_severity": row.get("damage_severity", ""),
                "informative": row.get("informative", ""),
                "humanitarian": row.get("humanitarian", ""),
                "disaster_types": row.get("disaster_types", ""),
                "llm_output": row.get("llm_output", ""),
                "caption_text": prompt_text,
                "source_type": row.get("source_type", ""),
                "fallback_keys": row.get("fallback_keys", ""),
                "is_failure": "TRUE",
                "model_name": row.get("model_name", ""),
                "failure_reason": err_info["error_type"],
                "used_fallback": "FALSE",
                "aug_img_path": "CONTENT_REFUSAL:EmptyPrompt",
            }

            # Write row to output CSV
            with csv_lock:
                with open(csv_out, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=FIELDNAMES_BF,
                        delimiter="\t",
                        quoting=csv.QUOTE_ALL,
                    )
                    writer.writerow(row_data)
            continue

        # Attempt BFS generation
        data, err = generate_image_with_black_forest(
            prompt=prompt_text,  # This should now be the full caption text
            height=height,
            width=width,
            steps=steps,
            guidance=guidance,
            safety_tolerance=safety_tolerance,
            seed=seed,
            output_format=output_format,
            api_key=api_key,
        )

        used_fallback = False

        # If BFS fails initially, try fallback
        if data is None and fallback_prompt:
            if err:
                error_log_df = log_error(error_log_df, row_idx, prompt_text, err)

            fallback_text = (
                f"{prompt_text}\n\n"
                "Academic disclaimers...toning down content but preserving scenario..."
            )
            fallback_safety = min(safety_tolerance + 1, 6)
            data_fb, err_fb = generate_image_with_black_forest(
                prompt=fallback_text,
                height=height,
                width=width,
                steps=steps,
                guidance=guidance,
                safety_tolerance=fallback_safety,
                seed=seed,
                output_format=output_format,
                api_key=api_key,
            )
            if data_fb is not None:
                data = data_fb
                used_fallback = True
            else:
                if err_fb:
                    error_log_df = log_error(
                        error_log_df, row_idx, f"FALLBACK: {fallback_text}", err_fb
                    )

        if data is None:
            # Final refusal
            if not fallback_prompt and err:
                error_log_df = log_error(error_log_df, row_idx, prompt_text, err)

            df_chunk.at[row_idx, "aug_img_path"] = "CONTENT_REFUSAL"
            row_data = {
                "image_path": row.get("image_path", ""),
                "damage_severity": row.get("damage_severity", ""),
                "informative": row.get("informative", ""),
                "humanitarian": row.get("humanitarian", ""),
                "disaster_types": row.get("disaster_types", ""),
                "llm_output": row.get("llm_output", ""),
                "caption_text": prompt_text,
                "source_type": row.get("source_type", ""),
                "fallback_keys": row.get("fallback_keys", ""),
                "is_failure": "TRUE",
                "model_name": row.get("model_name", ""),
                "failure_reason": err["error_type"] if err else "Refusal",
                "used_fallback": str(used_fallback),
                "aug_img_path": "CONTENT_REFUSAL",
            }

            # Write row to output CSV
            with csv_lock:
                with open(csv_out, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=FIELDNAMES_BF,
                        delimiter="\t",
                        quoting=csv.QUOTE_ALL,
                    )
                    writer.writerow(row_data)
            continue

        # If BFS returned valid image bytes, store them
        out_filename = f"bfimg_{row_idx}.{output_format}"
        out_path = os.path.join(bfs_model_dir, out_filename)

        try:
            # Write image to disk
            with open(out_path, "wb") as f:
                f.write(data)

            df_chunk.at[row_idx, "aug_img_path"] = out_path
            df_chunk.at[row_idx, "used_fallback"] = used_fallback

            row_data = {
                "image_path": row.get("image_path", ""),
                "damage_severity": row.get("damage_severity", ""),
                "informative": row.get("informative", ""),
                "humanitarian": row.get("humanitarian", ""),
                "disaster_types": row.get("disaster_types", ""),
                "llm_output": row.get("llm_output", ""),
                "caption_text": prompt_text,
                "source_type": row.get("source_type", ""),
                "fallback_keys": row.get("fallback_keys", ""),
                "is_failure": "FALSE",
                "model_name": row.get("model_name", ""),
                "failure_reason": "",
                "used_fallback": str(used_fallback),
                "aug_img_path": out_path,
            }

            # Write row to output CSV
            with csv_lock:
                with open(csv_out, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=FIELDNAMES_BF,
                        delimiter="\t",
                        quoting=csv.QUOTE_ALL,
                    )
                    writer.writerow(row_data)

        except OSError as e:
            # Error writing file
            err_info = {
                "error_type": "File Write Error",
                "error_message": str(e),
                "http_status": None,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            error_log_df = log_error(error_log_df, row_idx, prompt_text, err_info)
            df_chunk.at[row_idx, "aug_img_path"] = "CONTENT_REFUSAL:WriteErr"
            row_data = {
                "image_path": row.get("image_path", ""),
                "damage_severity": row.get("damage_severity", ""),
                "informative": row.get("informative", ""),
                "humanitarian": row.get("humanitarian", ""),
                "disaster_types": row.get("disaster_types", ""),
                "llm_output": row.get("llm_output", ""),
                "caption_text": prompt_text,
                "source_type": row.get("source_type", ""),
                "fallback_keys": row.get("fallback_keys", ""),
                "is_failure": "TRUE",
                "model_name": row.get("model_name", ""),
                "failure_reason": "File Write Error",
                "used_fallback": str(used_fallback),
                "aug_img_path": "CONTENT_REFUSAL:WriteErr",
            }

            # Write row to output CSV
            with csv_lock:
                with open(csv_out, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=FIELDNAMES_BF,
                        delimiter="\t",
                        quoting=csv.QUOTE_ALL,
                    )
                    writer.writerow(row_data)

    return df_chunk, error_log_df


def create_augmented_training_file(
    relabelled_train_tsv: str,
    synthetic_csv: str,
    augmented_tsv: str,
    dataset_prefix_to_strip: str = "/home/evammun/Thesis/Dataset/",
    quoting=csv.QUOTE_ALL,
):
    """
    Merge a relabelled MEDIC training set (TSV) with newly generated synthetic data (CSV).

    Steps:
      1) Load MEDIC relabelled training (e.g. MEDIC_train_relabelled.tsv).
      2) Keep only columns:
         [image_id, event_name, image_path, damage_severity, informative, humanitarian, disaster_types].
      3) Load the synthetic CSV (e.g. final_synthetic.csv).
      4) Filter it to only rows where is_failure == "False" AND aug_img_path does NOT contain "CONTENT_REFUSAL".
      5) For each synthetic row, create a new record:
           image_id => "syn_{row index}"
           event_name => "synthetic"
           image_path => aug_img_path minus the dataset_prefix_to_strip (if present)
           damage_severity, informative, humanitarian, disaster_types => copied from that row
      6) Concatenate original rows with these new synthetic rows.
      7) Write out the result to augmented_tsv (tab-delimited, quoting=csv.QUOTE_ALL).

    Parameters
    ----------
    relabelled_train_tsv : str
        Path to the relabelled MEDIC training TSV file.
    synthetic_csv : str
        Path to the CSV containing BFS or SD synthetic data, with columns including:
        [damage_severity, informative, humanitarian, disaster_types, is_failure, aug_img_path].
    augmented_tsv : str
        Path where the merged TSV will be saved.
    dataset_prefix_to_strip : str
        If synthetic 'aug_img_path' starts with this prefix, remove that prefix from the final path.
    quoting : optional
        The quoting style used in reading/writing TSV (defaults to csv.QUOTE_ALL).

    Returns
    -------
    None
        Creates a merged TSV containing the original data plus the newly added synthetic rows.
    """
    # 1) Load relabelled training (TSV)
    df_relab = pd.read_csv(
        relabelled_train_tsv,
        sep="\t",
        quoting=quoting,
        dtype=str,  # keep all fields as strings to avoid dtype issues
    )

    # 2) Keep only relevant columns
    keep_cols = [
        "image_id",
        "event_name",
        "image_path",
        "damage_severity",
        "informative",
        "humanitarian",
        "disaster_types",
    ]
    df_relab_min = df_relab[keep_cols].copy()

    # 3) Load synthetic CSV (with proper quoting for multiline fields)
    df_syn = pd.read_csv(
        synthetic_csv,
        sep="\t",
        quoting=quoting,
        dtype=str,
        on_bad_lines="warn",
        engine="python",  # More robust parsing
    )

    # 4) Filter to successful synthetic entries (is_failure='False') AND no "CONTENT_REFUSAL"
    df_syn_ok = df_syn[
        (df_syn["is_failure"].str.lower() == "false")
        & (~df_syn["aug_img_path"].str.contains("CONTENT_REFUSAL", na=False))
    ].copy()

    if df_syn_ok.empty:
        print("No synthetic rows found that are successful and not content refusals.")
    else:
        print(
            f"Found {len(df_syn_ok)} synthetic rows that are is_failure='False' and aug_img_path != 'CONTENT_REFUSAL'."
        )

    # 5) Build new rows from the filtered synthetic set
    new_rows = []
    for idx, row in df_syn_ok.iterrows():
        aug_path = str(row.get("aug_img_path", ""))
        # Strip the user-supplied prefix if present
        if aug_path.startswith(dataset_prefix_to_strip):
            new_path = aug_path[len(dataset_prefix_to_strip) :]
        else:
            new_path = aug_path

        new_rec = {
            "image_id": f"syn_{idx}",
            "event_name": "synthetic",
            "image_path": new_path,
            "damage_severity": row.get("damage_severity", ""),
            "informative": row.get("informative", ""),
            "humanitarian": row.get("humanitarian", ""),
            "disaster_types": row.get("disaster_types", ""),
        }
        new_rows.append(new_rec)

    df_syn_new = pd.DataFrame(new_rows, columns=keep_cols)

    # 6) Concatenate original with new synthetic rows
    df_aug = pd.concat([df_relab_min, df_syn_new], ignore_index=True)

    # Write out final augmented TSV
    os.makedirs(os.path.dirname(augmented_tsv), exist_ok=True)
    df_aug.to_csv(
        augmented_tsv,
        sep="\t",
        quoting=quoting,
        index=False,
    )

    print(f"Augmented file created with {len(df_aug)} total rows:")
    print(f"  Original relabelled: {len(df_relab_min)}")
    print(f"  Synthetic added: {len(df_syn_new)}")
    print(f"Saved to: {augmented_tsv}")
