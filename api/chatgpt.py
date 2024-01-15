"""
This module provides utility functions for interacting with the OpenAI API and Gradio interfaces.
"""
import os
from typing import Optional
import openai
from openai import OpenAI
import gradio as gr


def url_to_gradio_image_name(url: str) -> str:
    """
    Converts an OpenAI generated URL to a Gradio-compatible image name.

    This function extracts the portion of the URL after the last forward slash ('/'). It removes special characters
    often found in URLs such as '%', '&', and '='. The resulting string is truncated to a maximum length of 200
    characters to prevent issues with file name length limitations.

    :param url: The URL containing the image name.
    :returns: A cleaned and truncated version of the image name suitable for use with Gradio.
    """

    # Get the part after the final `/` in the URL
    image_name = url.rsplit('/', 1)[-1]

    # Remove any '%', "&", "=" from the image name
    image_name = image_name.replace("%", "")
    image_name = image_name.replace("&", "")
    image_name = image_name.replace("=", "")

    # Only get UP TO the first 200 characters of the image name. At least on my computer the file names get cut off at
    # 200 causing referencing them from json (which has the complete 200+ character name) to fail, as there's no match.
    image_name = image_name[:200]
    return image_name


def get_openai_client(api_key: Optional[str] = None) -> Optional[OpenAI]:
    """
    Creates and returns an OpenAI client object configured with the given API key.

    This function initializes an OpenAI client using the provided API key. If the provided API key is None or empty,
    it attempts to retrieve the API key from the environment variable 'OPENAI_API_KEY'. If the environment variable is
    also not set, it raises a warning and returns None.

    :param api_key: The API key for OpenAI. If not provided, the function will try to use the API key from the
    environment variable.
    :returns: An instance of the OpenAI client configured with the API key, or None if no valid API key is provided.
    """
    if api_key is None or api_key == "":
        api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None or api_key == "":
        gr.Warning("No OPENAI_API_KEY environment variable set.")
        return None

    return OpenAI(api_key=api_key)


def get_chat_response(client: openai.Client, api_model: str, role: str, prompt: str, context: Optional[list] = None,
                      as_json: bool = False) -> Optional[str]:
    """
    Generates a chat response using the OpenAI API based on the provided parameters.

    This function sends a message to the OpenAI API using the specified client and model. It constructs a message with
    a role (system or user) and the provided prompt. It also optionally includes previous chat context. The response
    can be returned in JSON format if specified.

    :param client: The OpenAI client to use for making the API call.
    :param api_model: The model to use for the chat completion (e.g., 'davinci-codex').
    :param role: The role the AI should assume.
    :param prompt: The message prompt to send to the chat model.
    :param context: A list of previous chat messages to provide context. Default is None.
    :param as_json: A flag to specify if the response should be in JSON format. Default is False.

    :returns: The chat response as a string, or None if there was an error or no response generated.
    """
    message = [
        {"role": "system",
         "content": role},
    ]

    # Give the model previous chat context
    if context is not None and len(context) > 0:
        for curr_context in context:
            message.append(curr_context)

    message.append({
        "role": "user",
        "content": prompt,
    })

    if as_json:
        response = client.chat.completions.create(
            model=api_model,
            response_format={"type": "json_object"},
            messages=message,
        )
    else:
        response = client.chat.completions.create(
            model=api_model,
            messages=message,
        )

    response = response.choices[0]
    if response.finish_reason != "stop":
        match response.finish_reason:
            case "length":
                gr.Warning(
                    f"finish_reason: {response.finish_reason}. The maximum number of tokens specified in the request "
                    f"was reached.")
                return None
            case "content_filter":
                gr.Warning(
                    f"finish_reason: {response.finish_reason}. The content was omitted due to a flag from OpenAI's "
                    f"content filters.")
                return None

    content = response.message.content
    if content is None or content == "":
        gr.Warning("No content was generated.")
        return None

    return content


def get_image_response(client: openai.Client, api_model: str, prompt: str, portrait=False) -> Optional[str]:
    """
    Generates an image response using the OpenAI API based on a given prompt and specified parameters.

    This function requests the OpenAI API to generate an image based on the provided text prompt. It allows
    specification of the model to use and whether the generated image should be in a portrait format. For 'dall-e-3'
    model, it supports high-definition (HD) quality image generation.

    :param client: The OpenAI client to use for making the API call.
    :param api_model: The model to use for image generation (e.g., 'dall-e-3').
    :param prompt: The text prompt based on which the image is generated.
    :param portrait: A flag to specify if the generated image should be in portrait orientation. Default is False.

    :returns: The URL of the generated image, or None if no image was generated or if there was an error.
    """
    image_size = "1024x1024"
    if portrait and api_model == "dall-e-3":
        image_size = "1024x1792"

    image_response = client.images.generate(
        prompt=prompt,
        model=api_model,
        size=image_size,
        n=1,
        quality="hd" if api_model == "dall-e-3" else "standard",
        response_format="url",
    )

    if image_response.data is None or len(image_response.data) == 0:
        gr.Warning("No image was generated.")
        return None

    if image_response.data[0].url is None or image_response.data[0].url == "":
        gr.Warning("No image url was generated.")
        return None

    return image_response.data[0].url
