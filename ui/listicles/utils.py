import gradio as gr
import json
import utils.image as image_utils
import processing.image as image_processing
import os
import utils.font_manager as font_manager
import api.chatgpt as chatgpt_api


def process(image_files, json_data,
            nf_family, nf_style, nfs, nfc, nfo, nse, nsc, nso, nsr, nbe, nbc, nbo,
            df_family, df_style, dfs, dfc, dfo, dse, dsc, dso, dsr, dbe, dbc, dbo,
            mf_family, mf_style, mfs, mfc, mfo, mse, msc, mso, msr, mbe, mbc, mbo,
            rf_family, rf_style, rfs, rfc, rfo, rse, rsc, rso, rsr, rbe, rbc, rbo):
    if not json_data:
        print("No JSON file uploaded.")
        return
    if not image_files:
        print("No images uploaded.")
        return

    font_families = font_manager.get_fonts()
    nff = font_families[nf_family][nf_style]
    dff = font_families[df_family][df_style]
    mff = font_families[mf_family][mf_style]
    rff = font_families[rf_family][rf_style]

    images = []

    rating_offset = 34
    text_offset = 49
    json_data = json.loads(json_data)

    if len(image_files) != len(json_data["items"]):
        gr.Warning(
            f"Number of images ({len(image_files)}) does not match the number of items in the JSON ({len(json_data)}).")

    # We skip any entries that don't have an image field.
    json_data_items = json_data["items"]
    json_dict = {item["image"]: item for item in json_data_items if "image" in item}

    for image_file in image_files:
        img_name = os.path.basename(image_file.name)
        if img_name not in json_dict:
            gr.Warning(
                f"Image {img_name} not found in the JSON list. Make sure the JSON contains a reference to this image.")
            continue

        img = image_processing.read_image_from_disk(image_file.name, size=(1080, 1920))
        item = json_dict[img_name]

        # Calculate positions for the text
        top_center = (0, int(img.shape[0] * 0.13))
        bottom_center = (0, int(img.shape[0] * 0.70))

        # Add association and rating at the top center, one above the other
        img, (_, association_height) = image_processing.add_text(img, item["association"], top_center, mff,
                                                                 font_size=mfs,
                                                                 font_color=image_utils.get_rgba(mfc, mfo),
                                                                 show_shadow=mse,
                                                                 shadow_radius=msr,
                                                                 shadow_color=image_utils.get_rgba(msc, mso),
                                                                 show_background=mbe,
                                                                 background_color=image_utils.get_rgba(mbc, mbo),
                                                                 x_center=True)

        img, (_, _) = image_processing.add_text(img, f'{json_data["rating_type"]}: {item["rating"]}%',
                                                (0, top_center[1] + association_height + rating_offset),
                                                rff, font_size=rfs, font_color=image_utils.get_rgba(rfc, rfo),
                                                show_shadow=rse, shadow_radius=rsr,
                                                shadow_color=image_utils.get_rgba(rsc, rso),
                                                show_background=rbe, background_color=image_utils.get_rgba(rbc, rbo),
                                                x_center=True)

        # Add name and description at the bottom center, one above the other
        img, (_, name_height) = image_processing.add_text(img, item["name"], bottom_center, nff, font_size=nfs,
                                                          font_color=image_utils.get_rgba(nfc, nfo),
                                                          max_width=15,
                                                          show_shadow=nse, shadow_radius=nsr,
                                                          shadow_color=image_utils.get_rgba(nsc, nso),
                                                          show_background=nbe,
                                                          background_color=image_utils.get_rgba(nbc, nbo),
                                                          x_center=True)
        img, (_, _) = image_processing.add_text(img, f'"{item["description"]}"',
                                                (0, bottom_center[1] + name_height + text_offset), dff,
                                                font_size=dfs, font_color=image_utils.get_rgba(dfc, dfo),
                                                show_shadow=dse, shadow_radius=dsr,
                                                shadow_color=image_utils.get_rgba(dsc, dso),
                                                show_background=dbe, background_color=image_utils.get_rgba(dbc, dbo),
                                                max_width=43, x_center=True)

        images += [img]

    return images


def validate_json(json_file):
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


def send_artifacts_to_batch(listicle_images, json_data):
    if not listicle_images or len(listicle_images.root) == 0:
        gr.Warning("No images to send.")
        return
    if not json_data or len(json_data) == 0:
        gr.Warning("No JSON data to send.")
        return
    # Parse the listicle_images GalleryData to get file paths
    listicle_images = listicle_images.root
    listicle_images = [image.image.path for image in listicle_images]
    return listicle_images, json_data


def save_artifacts(listicle_images, image_type, json_data):
    if not json_data or len(json_data) == 0:
        gr.Warning("No JSON data to save.")
        return

    # Save the images
    save_dir = image_processing.save_images_to_disk(listicle_images, image_type)

    # Save the JSON data
    if save_dir is not None and save_dir != "":
        json_filepath = os.path.join(save_dir, "data.json")
        with open(json_filepath, "w") as file:
            json_data = json.loads(json_data)
            json.dump(json_data, file, indent=4)

        gr.Info(f"Saved generated artifacts to {save_dir}.")


def generate_listicle(api_key, api_text_model, api_image_model, number_of_items, topic, association,
                      rating_type, details="", generate_artifacts=False):
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
            message += (f"Include a top-level field `rating_type: <string>` with what the rating "
                        f"represents.")

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
