import openai
from openai import OpenAI
import os
import gradio as gr


# The actual gradio image name (+ orig_name) is  <>.png, but the tmp file created and sent to
# batch is based on the portion after the last `/` in the url without the '%' (looks url encoded).
def url_to_gradio_image_name(url):
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


def get_openai_client(api_key):
    if api_key is None or api_key == "":
        api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None or api_key == "":
        gr.Warning("No OPENAI_API_KEY environment variable set.")
        return None

    return OpenAI(api_key=api_key)


def get_chat_response(client: openai.Client, api_model: str, role: str, prompt: str, context: list = None, as_json: bool= False):
    message = [
        {"role": "system",
         "content": role},
    ]

    # Give the model previous chat context
    if context is not None and len(context) > 0:
        for c in context:
            message.append(c)

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
        if response.finish_reason == "length":
            gr.Warning(
                f"finish_reason: {response.finish_reason}. The maximum number of tokens specified in the request was reached.")
            return None, None, None
        elif response.finish_reason == "content_filter":
            gr.Warning(
                f"finish_reason: {response.finish_reason}. The content was omitted due to a flag from OpenAI's content filters.")
            return None, None, None

    content = response.message.content
    if content is None or content == "":
        gr.Warning("No content was generated.")
        return None, None

    return content


def get_image_response(client: openai.Client, api_model: str, prompt: str, portrait=False):
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
