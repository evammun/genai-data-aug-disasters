# src/relabelling/llm_calls.py
import os
import json
import base64
import re
import mimetypes

from config import (
    ANTHROPIC_KEY,
    OPENAI_KEY,
    ANTHROPIC_MODEL,
    OPENAI_MODEL,
    RELABELLING_PROMPT,
)

import anthropic
from openai import OpenAI
import config


def extract_json_from_code_block(text: str) -> str:
    """
    Enhanced version that handles JSON extraction in various formats:
    1. Code fences (```json { ... } ```)
    2. Plain JSON objects with explanatory text before/after
    3. Plain text with embedded JSON object

    Args:
        text (str): Text that may contain a JSON object

    Returns:
        str: The extracted JSON string, or the original text if no valid JSON found
    """
    import re
    import json

    # Strategy 1: Original functionality - Look for content between code blocks
    match = re.search(r"```(?:json)?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        potential_json = match.group(1).strip()
        try:
            # Validate that this is actual JSON
            json.loads(potential_json)
            return potential_json
        except:
            pass  # If not valid JSON, continue to next strategy

    # Strategy 2: Look for content between { and } brackets (full object)
    match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if match:
        potential_json = match.group(1).strip()
        try:
            # Validate that this is actual JSON
            json.loads(potential_json)
            return potential_json
        except:
            pass  # If not valid JSON, continue to next strategy

    # Strategy 3: Try to find a JSON-like structure by looking for lines
    lines = text.split("\n")

    # Find the first line containing an opening brace
    start_line = -1
    for i, line in enumerate(lines):
        if "{" in line:
            start_line = i
            break

    if start_line >= 0:
        # Find the last line containing a closing brace
        end_line = -1
        for i in range(len(lines) - 1, start_line - 1, -1):
            if "}" in lines[i]:
                end_line = i
                break

        if end_line >= 0:
            # Extract the text between these lines
            json_block = "\n".join(lines[start_line : end_line + 1])

            # Further clean by taking just from first { to last }
            start_char = json_block.find("{")
            end_char = json_block.rfind("}") + 1
            if start_char >= 0 and end_char > start_char:
                potential_json = json_block[start_char:end_char]
                try:
                    # Validate that this is actual JSON
                    json.loads(potential_json)
                    return potential_json
                except:
                    pass  # If not valid, continue

    # If all strategies fail, return the original text as before
    return text.strip()


def guess_media_type(image_path: str) -> str:
    """
    Guesses the media type (e.g. image/jpeg, image/png) for the given file path.
    If unknown, defaults to image/jpeg.
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith("image/"):
        return "image/jpeg"
    return mime_type


def call_claude_api(image_path):
    """
    Calls Anthropic (Claude) using the RELABELLING_PROMPT for instructions.
    Returns (labels_dict, None) if successful, or (None, fail_record) if something fails.

    The fail_record is a dict with keys:
      {
        "model": "Claude",
        "image_path": <str>,
        "reason": <str>,
        "raw_content": <str>,
      }
    """
    # 1) Read & encode image
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        reason = f"FileReadError: {e}"
        fail_record = {
            "model": "Claude",
            "image_path": image_path,
            "reason": reason,
            "raw_content": "",
        }
        return None, fail_record

    # 2) Guess correct media type (image/jpeg or image/png, etc.)
    media_type = guess_media_type(image_path)

    # 3) Build user message
    user_message = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,  # use the guessed type
                "data": image_data,
            },
        },
        {
            "type": "text",
            "text": RELABELLING_PROMPT,
        },
    ]

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=512,
            temperature=0.0,
        )
    except Exception as e:
        # Capture full error in raw_content for debugging
        fail_record = {
            "model": "Claude",
            "image_path": image_path,
            "reason": f"APICallError: {str(e)}",
            "raw_content": f"{str(e)}",
        }
        return None, fail_record

    if not response.content or len(response.content) == 0:
        fail_record = {
            "model": "Claude",
            "image_path": image_path,
            "reason": "EmptyContent",
            "raw_content": "",
        }
        return None, fail_record

    raw_reply = response.content[0].text.strip()
    if not raw_reply:
        fail_record = {
            "model": "Claude",
            "image_path": image_path,
            "reason": "EmptyReply",
            "raw_content": "",
        }
        return None, fail_record

    cleaned = extract_json_from_code_block(raw_reply)
    try:
        labels_dict = json.loads(cleaned)
    except json.JSONDecodeError:
        fail_record = {
            "model": "Claude",
            "image_path": image_path,
            "reason": "JSONParseError",
            "raw_content": raw_reply[:300],
        }
        return None, fail_record

    return labels_dict, None


def call_gpt_api(image_path):
    """
    Calls GPT with the same RELABELLING_PROMPT instructions.
    Returns (labels_dict, None) if success, else (None, fail_record).
    """
    # 1) Read & encode
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        fail_record = {
            "model": "GPT",
            "image_path": image_path,
            "reason": f"FileReadError: {e}",
            "raw_content": "",
        }
        return None, fail_record

    # 2) Guess correct media type
    media_type = guess_media_type(image_path)

    system_prompt = RELABELLING_PROMPT
    user_message_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{image_data}",
                "detail": "auto",
            },
        }
    ]

    client = OpenAI(api_key=config.OPENAI_KEY)
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message_content},
            ],
            max_tokens=512,
            temperature=0.0,
        )
    except Exception as e:
        fail_record = {
            "model": "GPT",
            "image_path": image_path,
            "reason": f"APICallError: {str(e)}",
            "raw_content": f"{str(e)}",
        }
        return None, fail_record

    if not response.choices:
        fail_record = {
            "model": "GPT",
            "image_path": image_path,
            "reason": "NoChoices",
            "raw_content": "",
        }
        return None, fail_record

    choice = response.choices[0]
    msg_obj = getattr(choice, "message", None)
    if not msg_obj:
        fail_record = {
            "model": "GPT",
            "image_path": image_path,
            "reason": "NoMessage",
            "raw_content": "",
        }
        return None, fail_record

    raw_reply = msg_obj.content.strip()
    if not raw_reply:
        fail_record = {
            "model": "GPT",
            "image_path": image_path,
            "reason": "EmptyReply",
            "raw_content": "",
        }
        return None, fail_record

    cleaned = extract_json_from_code_block(raw_reply)
    try:
        labels_dict = json.loads(cleaned)
    except json.JSONDecodeError:
        fail_record = {
            "model": "GPT",
            "image_path": image_path,
            "reason": "JSONParseError",
            "raw_content": raw_reply[:300],
        }
        return None, fail_record

    return labels_dict, None
