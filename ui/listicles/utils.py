"""
This file contains the functions that are used by the Gradio UI to generate listicles.
"""
import os
import json
from typing import Optional, Any, List
import gradio as gr
import numpy as np
import processing.image as image_processing
from utils import font_manager, image as image_utils, dataclasses
import api.chatgpt as chatgpt_api


# pylint: disable=too-many-locals
def process(image_files: list[Any], json_data: str,
            nf_family: str, nf_style: str, nfs: int, nfc: dataclasses.RGBColor, nfo: int, nse: bool,
            nsc: dataclasses.RGBColor, nso: int, nsr, nbe: bool, nbc: dataclasses.RGBColor, nbo: int,
            df_family: str, df_style: str, dfs: int, dfc: dataclasses.RGBColor, dfo: int, dse: bool,
            dsc: dataclasses.RGBColor, dso: int, dsr, dbe: bool, dbc: dataclasses.RGBColor, dbo: int,
            mf_family: str, mf_style: str, mfs: int, mfc: dataclasses.RGBColor, mfo: int, mse: bool,
            msc: dataclasses.RGBColor, mso: int, msr, mbe: bool, mbc: dataclasses.RGBColor, mbo: int,
            rf_family: str, rf_style: str, rfs: int, rfc: dataclasses.RGBColor, rfo: int, rse: bool,
            rsc: dataclasses.RGBColor, rso: int, rsr, rbe: bool, rbc: dataclasses.RGBColor, rbo: int) \
        -> Optional[List[np.ndarray]]:
    """
    Processes the images and JSON data to generate the listicle images.
    :param image_files: The list of images to process. This is a gradio File.
    :param json_data: The JSON data to process.
    :param nf_family: The font family for the name.
    :param nf_style: The font style for the name.
    :param nfs: The font size for the name.
    :param nfc: The font color for the name.
    :param nfo: The font opacity for the name.
    :param nse: Whether to show the shadow for the name.
    :param nsc: The shadow color for the name.
    :param nso: The shadow opacity for the name.
    :param nsr: The shadow radius for the name.
    :param nbe: Whether to show the background for the name.
    :param nbc: The background color for the name.
    :param nbo: The background opacity for the name.
    :param df_family: The font family for the description.
    :param df_style: The font style for the description.
    :param dfs: The font size for the description.
    :param dfc: The font color for the description.
    :param dfo: The font opacity for the description.
    :param dse: Whether to show the shadow for the description.
    :param dsc: The shadow color for the description.
    :param dso: The shadow opacity for the description.
    :param dsr: The shadow radius for the description.
    :param dbe: Whether to show the background for the description.
    :param dbc: The background color for the description.
    :param dbo: The background opacity for the description.
    :param mf_family: The font family for the association.
    :param mf_style: The font style for the association.
    :param mfs: The font size for the association.
    :param mfc: The font color for the association.
    :param mfo: The font opacity for the association.
    :param mse: Whether to show the shadow for the association.
    :param msc: The shadow color for the association.
    :param mso: The shadow opacity for the association.
    :param msr: The shadow radius for the association.
    :param mbe: Whether to show the background for the association.
    :param mbc: The background color for the association.
    :param mbo: The background opacity for the association.
    :param rf_family: The font family for the rating.
    :param rf_style: The font style for the rating.
    :param rfs: The font size for the rating.
    :param rfc: The font color for the rating.
    :param rfo: The font opacity for the rating.
    :param rse: Whether to show the shadow for the rating.
    :param rsc: The shadow color for the rating.
    :param rso: The shadow opacity for the rating.
    :param rsr: The shadow radius for the rating.
    :param rbe: Whether to show the background for the rating.
    :param rbc: The background color for the rating.
    :param rbo: The background opacity for the rating.
    :return: The list of processed images as numpy arrays. If there was an error, returns None.
    """
    if not json_data:
        print("No JSON file uploaded.")
        return None
    if not image_files:
        print("No images uploaded.")
        return None

    font_families = font_manager.get_fonts()

    images = []

    json_data = json.loads(json_data)
    if len(image_files) != len(json_data["items"]):
        gr.Warning(
            f"Number of images ({len(image_files)}) does not match the number of items in the JSON ({len(json_data)}).")

    # We skip any entries that don't have an image field.
    json_dict = {item["image"]: item for item in json_data["items"] if "image" in item}

    for image_file in image_files:
        img_name = os.path.basename(image_file.name)
        if img_name not in json_dict:
            gr.Warning(
                f"Image {img_name} not found in the JSON list. Make sure the JSON contains a reference to this image.")
            continue

        img = image_processing.read_image_from_disk(image_file.name, size=(1080, 1920))
        item = json_dict[img_name]

        # Calculate y-positions for the text
        top_center = (0, int(img.shape[0] * 0.13))
        bottom_center = (0, int(img.shape[0] * 0.70))

        # Add association and rating at the top center, one above the other
        img, (_, association_height) = image_processing.add_text(img, item["association"], top_center,
                                                                 font_families[mf_family][mf_style], font_size=mfs,
                                                                 font_color=image_utils.get_rgba(mfc, mfo),
                                                                 show_shadow=mse, shadow_radius=msr,
                                                                 shadow_color=image_utils.get_rgba(msc, mso),
                                                                 show_background=mbe,
                                                                 background_color=image_utils.get_rgba(mbc, mbo),
                                                                 x_center=True)

        img, (_, _) = image_processing.add_text(img, f'{json_data["rating_type"]}: {item["rating"]}%',
                                                (0, top_center[1] + association_height + 34),
                                                font_families[rf_family][rf_style], font_size=rfs,
                                                font_color=image_utils.get_rgba(rfc, rfo), show_shadow=rse,
                                                shadow_radius=rsr, shadow_color=image_utils.get_rgba(rsc, rso),
                                                show_background=rbe, background_color=image_utils.get_rgba(rbc, rbo),
                                                x_center=True)

        # Add name and description at the bottom center, one above the other
        img, (_, name_height) = image_processing.add_text(img, item["name"], bottom_center,
                                                          font_families[nf_family][nf_style], font_size=nfs,
                                                          font_color=image_utils.get_rgba(nfc, nfo),
                                                          max_width=15,
                                                          show_shadow=nse, shadow_radius=nsr,
                                                          shadow_color=image_utils.get_rgba(nsc, nso),
                                                          show_background=nbe,
                                                          background_color=image_utils.get_rgba(nbc, nbo),
                                                          x_center=True)
        img, (_, _) = image_processing.add_text(img, f'"{item["description"]}"',
                                                (0, bottom_center[1] + name_height + 49),
                                                font_families[df_family][df_style], font_size=dfs,
                                                font_color=image_utils.get_rgba(dfc, dfo), show_shadow=dse,
                                                shadow_radius=dsr, shadow_color=image_utils.get_rgba(dsc, dso),
                                                show_background=dbe, background_color=image_utils.get_rgba(dbc, dbo),
                                                max_width=43, x_center=True)

        images += [img]

    return images


def validate_json(json_file: str) -> None:
    """
    Validates the JSON file to make sure it has the required fields.
    :param json_file: The JSON file to validate.
    :return: None
    """
    if not json_file or len(json_file) == 0:
        gr.Warning("No JSON in the code block.")
        return

    json_data = json.loads(json_file)

    if "items" not in json_data or len(json_data["items"]) == 0:
        gr.Warning("JSON is missing the 'items' field.")
        return

    # Make sure that the JSON has the required fields
    required_fields = ["image"]
    warnings = 0
    for index, item in enumerate(json_data["items"]):
        for field in required_fields:
            if field not in item:
                gr.Warning(f"JSON is missing an important field '{field}' at item index {index}.")
                warnings += 1

    if warnings == 0:
        gr.Info("JSON is valid!")


def send_artifacts_to_batch(listicle_images: gr.data_classes.RootModel, json_data: str) \
        -> (Optional[list], Optional[str]):
    """
    Sends the artifacts to the batch processing section.
    :param listicle_images: The list of images to send. This is a Gradio Gallery.
    :param json_data: The JSON data to send.
    :return: The list of images and the JSON data sent.
    """
    if not listicle_images or len(listicle_images.root) == 0:
        gr.Warning("No images to send.")
        return None, None
    if not json_data or len(json_data) == 0:
        gr.Warning("No JSON data to send.")
        return None, None
    # Parse the listicle_images GalleryData to get file paths
    listicle_images = listicle_images.root
    listicle_images = [image.image.path for image in listicle_images]
    return listicle_images, json_data


def save_artifacts(listicle_images: gr.data_classes.RootModel, image_type: gr.Dropdown, json_data: str) -> None:
    """
    Saves the artifacts to disk.
    :param listicle_images: The list of images to save. This is a Gradio Gallery.
    :param image_type: The type of image to save.
    :param json_data: The JSON data to save.
    :return: None
    """
    if not json_data or len(json_data) == 0:
        gr.Warning("No JSON data to save.")
        return None

    # Save the images
    save_dir = image_processing.save_images_to_disk(listicle_images, image_type)

    # Save the JSON data
    if save_dir is not None and save_dir != "":
        json_filepath = os.path.join(save_dir, "data.json")
        with open(json_filepath, "w", encoding="utf-8") as file:
            json_data = json.loads(json_data)
            json.dump(json_data, file, indent=4)

        gr.Info(f"Saved generated artifacts to {save_dir}.")

    return None


def generate_listicle(api_key: str, api_text_model: str, api_image_model: str, number_of_items: int, topic: str,
                      association: str, rating_type: str, details: str = "", generate_artifacts: bool = False) \
        -> (Optional[str], Optional[str], Optional[list[str]]):
    """
    Generates a listicle using the OpenAI API.
    :param api_key: The OpenAI API key to use.
    :param api_text_model: The OpenAI API text model to use (e.g. 'gpt-4').
    :param api_image_model: The OpenAI API image model to use (e.g. 'dall-e-3').
    :param number_of_items: The number of items to generate.
    :param topic: The topic of the listicle.
    :param association: What each item is associated with.
    :param rating_type: What the rating represents.
    :param details: Additional details about the listicle you want to generate.
    :param generate_artifacts: Whether to generate artifacts (images and JSON) for the listicle.
    :return: The listicle content, the listicle JSON, and the listicle images.
    """
    openai = chatgpt_api.get_openai_client(api_key)
    if openai is None:
        gr.Warning("No OpenAI client. Cannot generate listicle.")
        return None, None, None

    listicle_images = []
    listicle_json_data = ""
    additional_details = ""
    if association is not None and association != "":
        additional_details += f"Associate each item with a(n) {association}."
    additional_details += details

    role = f"You are a TikTok creator that is creating a listicle of {topic}."
    prompt = f"Generate a list of {number_of_items} {topic}. ONLY generate {number_of_items} items. " \
             f"For each item, add a unique name and description, and provide" \
             f" a rating from 0-100 for each based off {rating_type}. Make " \
             f"sure that the description is no longer than 344 characters " \
             f"long. {additional_details}"

    listicle_content = chatgpt_api.get_chat_response(openai, api_text_model, role, prompt=prompt)
    if listicle_content is None or listicle_content == "":
        return None, None, None

    if generate_artifacts:
        # https://github.com/openai/openai-python/issues/887#issuecomment-1829085545
        json_model = "gpt-4-1106-preview" if api_text_model == "gpt-4" else "gpt-3.5-turbo-1106"

        role = "You are a master at formatting pre-generated listicles into JSON."
        listicle_json_context = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": listicle_content}
        ]

        json_format = "{name: <string>, description: <string>, rating: <int>"
        if association is not None and association != "":  # Add association field if provided
            json_format += ", association: <string>"
        json_format += "}"
        message = (f"Format the listicle into JSON. For the items, store as a list named 'items' with the content "
                   f"format: {json_format}.")
        if rating_type is not None and rating_type != "":
            message += "Include a top-level field `rating_type: <string>` with what the rating represents."

        listicle_json = chatgpt_api.get_chat_response(openai, json_model, role, prompt=message,
                                                      context=listicle_json_context, as_json=True)
        if listicle_json is None or listicle_json == "":
            return listicle_content, None, None
        listicle_json_data = json.loads(listicle_json)

        # This may be wonky in the case of 'glasses' -> 'glass'. The description should still be
        #   valid and fix it. I chose this path because it would be confusing to ChatGPT if it said
        #   something like 'Generate an image of a monsters...' which is grammatically incorrect.
        p = font_manager.get_inflect()
        singular_topic = p.singular_noun(topic)
        if singular_topic is False:  # If it was singular, it returns False. Keep the original.
            singular_topic = topic

        # Generate images for each item in the listicle.
        for item in listicle_json_data["items"]:
            description = item["description"]
            name = item["name"]
            prompt = (f"Generate an image of a {str.lower(singular_topic)} known as '{name}.' "
                      f"Described as: '{description}'. Please ensure there are no words or text on "
                      f"the image.")
            image_url = chatgpt_api.get_image_response(openai, api_image_model, prompt, portrait=True)
            if image_url is None or image_url == "":
                continue

            item["image"] = chatgpt_api.url_to_gradio_image_name(image_url)
            listicle_images.append(image_url)

    # Because the json data was updated above, we need to re-serialize it.
    listicle_json = json.dumps(listicle_json_data, indent=4)

    return listicle_content, listicle_json, listicle_images
