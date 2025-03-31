import os
import base64

# Third-party libraries for each LLM provider
import openai
import anthropic
from mistralai.client import MistralClient

# Your project's config module with API keys/model IDs. Make sure these exist:
import config


def get_vision_llm(model_name: str = "gpt4v"):
    """
    Factory function for creating vision LLM instances by name.

    Metadata:
        Args:
            model_name (str): Identifier for the vision model. Examples include:
                              "gpt4v", "claude_sonnet", "claude_haiku",
                              "pixtral_large", "pixtral_small", etc.
        Returns:
            VisionLLMBase: Instance of a vision LLM subclass.
        Raises:
            ValueError: If the model_name is not recognized.
    """
    model_name = model_name.lower().strip()

    if model_name == "gpt4v":
        return GPT4VisionLLM(
            model_id=config.OPENAI_MODEL,  # e.g. "gpt-4o-2024-08-06"
            api_key=config.OPENAI_KEY,
        )
    elif model_name in ("claude_sonnet", "claude-3-7-sonnet-latest"):
        return ClaudeVisionLLM(
            model_id=config.ANTHROPIC_MODEL,  # e.g. "claude-3-7-sonnet-20250219"
            api_key=config.ANTHROPIC_KEY,
        )
    elif model_name == "claude_haiku":
        return ClaudeVisionLLM(
            model_id=config.HAIKU_MODEL,  # e.g. "claude-2-haiku-2024"
            api_key=config.ANTHROPIC_KEY,
        )
    elif model_name == "pixtral_large":
        # Make sure MISTRAL_API_KEY is set in the environment or in config
        api_key = os.environ.get(
            "MISTRAL_API_KEY",
            config.MISTRAL_KEY if hasattr(config, "MISTRAL_KEY") else "",
        )
        if not api_key:
            raise ValueError("No Mistral API key found in environment or config")

        return PixtralVisionLLM(
            model_id=config.PIXTRAL_LARGE_MODEL,  # e.g. "pixtral-12b-2409"
            api_key=api_key,
        )
    elif model_name == "pixtral_small":
        # Make sure MISTRAL_API_KEY is set in the environment or in config
        api_key = os.environ.get(
            "MISTRAL_API_KEY",
            config.MISTRAL_KEY if hasattr(config, "MISTRAL_KEY") else "",
        )
        if not api_key:
            raise ValueError("No Mistral API key found in environment or config")

        return PixtralVisionLLM(
            model_id=config.PIXTRAL_SMALL_MODEL,  # e.g. "pixtral-6b-1234"
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unrecognised vision model name: {model_name}")


class VisionLLMBase:
    """
    Base class for vision-capable LLMs. Subclasses must implement
    classify_image().
    """

    def __init__(self, model_id: str, api_key: str):
        """
        Constructor for VisionLLMBase.

        Metadata:
            Args:
                model_id (str): The model identifier (e.g. GPT-4 or Claude version).
                api_key (str): The API key for authentication.
        """
        self.model_id = model_id
        self.api_key = api_key

    def classify_image(self, image_path: str, prompt: str, **kwargs):
        """
        Classify an image using a textual prompt.

        Metadata:
            Args:
                image_path (str): Path to the local image file.
                prompt (str): The text prompt to guide classification.
                **kwargs: Additional parameters for extended functionality.

            Returns:
                dict: Dictionary of classification or description results.
        """
        raise NotImplementedError("Subclasses must provide a classify_image() method.")


class GPT4VisionLLM(VisionLLMBase):
    """
    A class integrating GPT-4 Vision from OpenAI to classify or describe an image.
    """

    def classify_image(self, image_path: str, prompt: str, **kwargs):
        """
        Classify/describe an image using the GPT-4 Vision endpoint.

        Metadata:
            Args:
                image_path (str): Path to the local image file.
                prompt (str): Prompt to guide the classification.

            Returns:
                dict: {
                    "summary": "Descriptive summary or classification from GPT-4V",
                    "raw_response": The entire LLM response for debugging
                }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Setup client for OpenAI v1.0+
        client = openai.OpenAI(api_key=self.api_key)

        # Encode image as base64
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        # Create the user message with the updated structure
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ]

        # Make the API call with the updated client approach
        response = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.0),
        )

        # Extract content from the response with the updated structure
        raw_output = response.choices[0].message.content

        # Return a simple dictionary
        return {"summary": raw_output, "raw_response": response}


class ClaudeVisionLLM(VisionLLMBase):
    """
    A class integrating Claude Vision from Anthropic.
    """

    def classify_image(self, image_path: str, prompt: str, **kwargs):
        """
        Classify/describe an image using Claude Vision (Anthropic) API.

        Metadata:
            Args:
                image_path (str): Path to the local image file.
                prompt (str): Prompt to guide classification.

            Returns:
                dict: {
                    "summary": "Descriptive summary or classification from Claude",
                    "raw_response": ...
                }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Encode the image in base64
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Configure Anthropic client
        client = anthropic.Anthropic(api_key=self.api_key)

        # Build messages for the Anthropic API
        user_message = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64,
                },
            },
            {"type": "text", "text": prompt},
        ]

        # Call the Anthropic messages endpoint
        response = client.messages.create(
            model=self.model_id,  # e.g. config.ANTHROPIC_MODEL
            max_tokens=kwargs.get("max_tokens", 1024),
            messages=[{"role": "user", "content": user_message}],
        )

        # Extract the content from the response
        # In the newer Claude API, content is directly accessible as a property
        content = response.content[0].text if response.content else "[No content found]"

        # Return a dictionary with the summary and raw response
        return {"summary": content, "raw_response": response}


class PixtralVisionLLM(VisionLLMBase):
    """
    A class integrating a hypothetical 'Pixtral' vision LLM using the Mistral client.
    We'll call it 'pixtral_large' or 'pixtral_small' per config references.
    """

    def classify_image(self, image_path: str, prompt: str, **kwargs):
        """
        Classify/describe an image using the Pixtral variant via Mistral.

        Metadata:
            Args:
                image_path (str): Path to the local image file.
                prompt (str): Prompt to guide the classification.

            Returns:
                dict: {
                    "summary": "Descriptive summary or classification from Pixtral",
                    "raw_response": ...
                }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Verify we have a valid API key
        if not self.api_key or self.api_key == "":
            raise ValueError(
                "No API key provided for Mistral. Check your configuration."
            )

        # Encode image in base64
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        # Initialize the Mistral client with a valid API key
        client = MistralClient(api_key=self.api_key)

        # Build the message content for Pixtral
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ]

        try:
            # Make the request to the Mistral API
            # Use proper error handling
            response = client.chat(
                model=self.model_id,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 1024),
            )

            # Extract the content from the response
            # The response format depends on the MistralClient's API
            # Check for the specific structure of the response
            chat_content = ""
            if hasattr(response, "choices") and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, "message") and hasattr(
                    first_choice.message, "content"
                ):
                    chat_content = first_choice.message.content

            # If we couldn't extract content in a structured way, convert to string as fallback
            if not chat_content and hasattr(response, "__str__"):
                chat_content = str(response)

            return {"summary": chat_content, "raw_response": response}

        except Exception as e:
            # Return an error message if the API call fails
            error_message = f"Error calling Mistral API: {str(e)}"
            return {"summary": error_message, "raw_response": None}
