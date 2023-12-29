import base64
import json

import cv2
import gradio as gr
import os
import glob
from PIL import ImageFont, ImageDraw, Image, ImageFilter
import numpy as np
import textwrap
from datetime import datetime
from openai import OpenAI
import uuid

# TODO: Add support to save a template for later use.
#   This would allow for multiple saved templates for quick style switching.
# TODO: Update batch processing to use parse fields (ex. association instead of months)

font_files = []
# TODO: Add support for Windows and Linux.
fonts_dirs = ["/Library/Fonts", "~/Library/Fonts", "System/Library/Fonts"]
for fonts_dir in fonts_dirs:
    fonts_dir = os.path.expanduser(fonts_dir)
    if not os.path.exists(fonts_dir):
        continue
    font_files += glob.glob(os.path.join(fonts_dir, "*.ttf"))
    font_files += glob.glob(os.path.join(fonts_dir, "*.otf"))
# Create a dictionary mapping font names to file paths
font_dict = {os.path.basename(font_file).rsplit('.', 1)[0]: font_file for font_file in font_files}
font_names = list(font_dict.keys())


def color_picker_with_opacity():
    with gr.Group():
        with gr.Row():
            color = gr.ColorPicker(label="Font Color", info=f'The color of the text.', scale=1)
            opacity = gr.Slider(0, 100, value=100, label="Opacity",
                                info=f'How opaque the object is. 0 = transparent, 100 = solid.', scale=2)

    return color, opacity


def add_visibility(checkbox, group):
    def update_visibility(checkbox_state):
        return gr.Group(visible=checkbox_state)

    checkbox.change(
        update_visibility,
        inputs=checkbox,
        outputs=group
    )


def image_editor_parameters(name, default_font_size=55, pre_opt_render_fn=None, post_opt_render_fn=None):
    pre_opt = None
    post_opt = None
    with gr.Accordion(label=name):
        with gr.Column():
            font_font = gr.Dropdown(font_names, value=font_names[0], label="Font", info=f'The font used for the text.')
            if pre_opt_render_fn:
                with gr.Group():
                    pre_opt = pre_opt_render_fn()
            with gr.Group():
                font_color, font_opacity = color_picker_with_opacity()
                font_size = gr.Number(default_font_size, label="Font Size", info=f'The size of the font.')
            with gr.Group():
                drop_shadow_checkbox = gr.Checkbox(False, label="Enable",
                                                   info=f'Whether or not to add a drop shadow to the text.')
                with gr.Group(visible=drop_shadow_checkbox.value) as additional_options:
                    drop_shadow_color, drop_shadow_opacity = color_picker_with_opacity()
                    drop_shadow_radius = gr.Number(0, label="Shadow Radius", info=f'The radius of the drop shadow.')
                    add_visibility(drop_shadow_checkbox, additional_options)
            with gr.Group():
                background_checkbox = gr.Checkbox(False, label="Enable",
                                                  info=f'Whether or not to add a background to the text.')
                with gr.Group(visible=background_checkbox.value) as additional_options:
                    background_color, background_opacity = color_picker_with_opacity()
                    add_visibility(background_checkbox, additional_options)
            if post_opt_render_fn:
                with gr.Group():
                    post_opt = post_opt_render_fn()

    return (pre_opt, post_opt, (font_font, font_size, font_color, font_opacity),
            (drop_shadow_checkbox, drop_shadow_color, drop_shadow_opacity, drop_shadow_radius),
            (background_checkbox, background_color, background_opacity))


def add_background(image_pil, draw, position, text, font, padding=(15, 5), fill_color=(0, 0, 0, 255), border_radius=0):
    # Calculate width and height of text with padding
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x1 = position[0] - padding[0]  # left
    y1 = position[1] - padding[1]  # top
    x2 = x1 + text_width + 2 * padding[0]  # right
    y2 = y1 + text_height + 2 * padding[1]  # bottom

    rect_img = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
    rect_draw = ImageDraw.Draw(rect_img)
    rect_draw.rounded_rectangle([x1, y1, x2, y2], fill=fill_color, radius=border_radius)
    image_pil.paste(rect_img, (0, 0), rect_img)

    return (x1 + padding[0], y1 + padding[1]), (x2 - x1, y2 - y1)


def add_blurred_shadow(image_pil, text, position, font, shadow_color=(0, 0, 0), shadow_offset=(0, 0),
                       blur_radius=1):
    # Create an image for the shadow
    shadow_image = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_image)

    # Draw the shadow text
    shadow_draw.text((position[0] + shadow_offset[0], position[1] + shadow_offset[1]), text, font=font,
                     fill=shadow_color)

    # Apply a Gaussian blur to the shadow image
    blurred_shadow = shadow_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Composite the blurred shadow with the original image
    image_pil.paste(blurred_shadow, (0, 0), blurred_shadow)


# Function to add text to an image with custom font, size, and wrapping
def add_text(image, text, position, font_path, font_size, font_color=(255, 255, 255, 255), shadow_color=(255, 255, 255),
             shadow_radius=None, max_width=None, show_background=False, show_shadow=False,
             background_color=(0, 0, 0, 255)):
    # Convert OpenCV image to PIL image
    image_pil = Image.fromarray(image).convert("RGBA")

    txt_layer = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(txt_layer)

    img_width, img_height = image.shape[1], image.shape[0]

    if max_width:  # Prepare for text wrapping if max_width is provided
        wrapped_text = textwrap.fill(text, width=max_width)
    else:
        wrapped_text = text

    lines = wrapped_text.split('\n')

    y_offset = 0
    max_line_width = 0  # Keep track of the widest line
    total_height = 0  # Accumulate total height of text block
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        max_line_width = max(max_line_width, line_width)
        total_height += line_height

        text_x = (img_width - line_width) / 2  # Adjusted to use numpy width
        line_y = position + y_offset
        y_offset += (line_height + 6)
        # TODO: This is bad for multiline text, since the background is drawn for each line and overlaps previous lines.
        if show_background:
            (text_x, line_y), _ = add_background(image_pil, draw, (text_x, line_y), line, font,
                                                 fill_color=background_color, border_radius=10)

        if show_shadow:
            shadow_position = (text_x, line_y)
            add_blurred_shadow(image_pil, line, shadow_position, font, shadow_color=shadow_color,
                               blur_radius=shadow_radius)

        draw.text((text_x, line_y), line, font=font, fill=font_color)

    image_pil = Image.alpha_composite(image_pil, txt_layer)
    return np.array(image_pil), (max_line_width, total_height)


def process_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # Convert to RGBA for PIL usage
    return cv2.resize(img, (1080, 1920), interpolation=cv2.INTER_AREA)


def print_parameters(name, ff, fs, fc, fo, se, sc, so, sr, be, bc, bo):
    return f"""- {name}
    - Font
        - Type: {ff}
        - Size: {fs}
        - Color: {fc}
        - Opacity: {fo}
    - Drop Shadow
        - Enable: {se}
        - Color: {sc}
        - Opacity: {so}
        - Radius: {sr}
    - Background
        - Enabled: {be}
        - Color: {bc}
        - Opacity: {bo}
    """


def get_rgba(color, opacity):
    # Opacity should be 0 -> 0, 100 -> 255
    alpha = int(opacity * 255 / 100)

    # if color is hex, convert to rgb
    if color.startswith("#"):
        color = color.lstrip("#")
        color = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    return color[0], color[1], color[2], alpha


def validate_json(json_file):
    if not json_file:
        gr.Warning("No JSON file uploaded.")
        return

    with open(json_file) as file:
        json_data = json.load(file)

    # Make sure that the JSON is a list
    if not isinstance(json_data, list):
        gr.Warning("JSON is not a list.")
        return

    if len(json_data) == 0:
        gr.Warning("JSON is empty.")
        return

    # Make sure that the JSON has the required fields
    required_fields = ["image"]
    warnings = 0
    for index, item in enumerate(json_data):
        for field in required_fields:
            if field not in item:
                gr.Warning(f"JSON is missing an important field '{field}' at item index {index}.")
                warnings += 1

    if warnings == 0:
        gr.Info("JSON is valid!")


# this is maybe the ugliest code I've written, but gradio inputs didn't allow me to pass in a class, namedtuple, or
# tuple to clean this up. Need to find a better/cleaner way to do this.
def process(image_files, json_file,
            nff, nfs, nfc, nfo, nse, nsc, nso, nsr, nbe, nbc, nbo,
            dff, dfs, dfc, dfo, dse, dsc, dso, dsr, dbe, dbc, dbo,
            mff, mfs, mfc, mfo, mse, msc, mso, msr, mbe, mbc, mbo,
            rating_text, rff, rfs, rfc, rfo, rse, rsc, rso, rsr, rbe, rbc, rbo):
    if not json_file:
        print("No JSON file uploaded.")
        return
    if not image_files:
        print("No images uploaded.")
        return

    if False:  # TODO: Unskip in future
        print(f"""Beginning processing with the following parameters...
        {print_parameters("Name", nff, nfs, nfc, nfo, nse, nsc, nso, nsr, nbe, nbc, nbo)}
        {print_parameters("Description", dff, dfs, dfc, dfo, dse, dsc, dso, dsr, dbe, dbc, dbo)}
        {print_parameters("Month", mff, mfs, mfc, mfo, mse, msc, mso, msr, mbe, mbc, mbo)}
        {print_parameters("Rating", rff, rfs, rfc, rfo, rse, rsc, rso, rsr, rbe, rbc, rbo)}
        """)

    images = []

    rating_offset = 34
    text_offset = 49
    with open(json_file) as file:
        json_data = json.load(file)

    if len(image_files) != len(json_data):
        gr.Warning(
            f"Number of images ({len(image_files)}) does not match the number of items in the JSON ({len(json_data)}).")

    # We skip any entries that don't have an image field.
    json_dict = {item["image"]: item for item in json_data if "image" in item}

    for image_file in image_files:
        img_name = os.path.basename(image_file.name)
        if img_name not in json_dict:
            gr.Warning(
                f"Image {img_name} not found in the JSON list. Make sure the JSON contains a reference to this image.")
            continue

        img = process_image(image_file.name)
        item = json_dict[img_name]

        # Calculate positions for the text
        top_center = int(img.shape[0] * 0.13)
        bottom_center = int(img.shape[0] * 0.70)

        # Add month and rating at the top center, one above the other
        img, (_, month_height) = add_text(img, item["month"], top_center, mff, font_size=mfs,
                                          font_color=get_rgba(mfc, mfo),
                                          show_shadow=mse, shadow_radius=msr, shadow_color=get_rgba(msc, mso),
                                          show_background=mbe, background_color=get_rgba(mbc, mbo))

        img, (_, _) = add_text(img, f'{rating_text}: {item["rating"]}%', top_center + month_height + rating_offset,
                               rff, font_size=rfs, font_color=get_rgba(rfc, rfo),
                               show_shadow=rse, shadow_radius=rsr, shadow_color=get_rgba(rsc, rso),
                               show_background=rbe, background_color=get_rgba(rbc, rbo))

        # Add name and description at the bottom center, one above the other
        img, (_, name_height) = add_text(img, item["name"], bottom_center, nff, font_size=nfs,
                                         font_color=get_rgba(nfc, nfo),
                                         max_width=15,
                                         show_shadow=nse, shadow_radius=nsr, shadow_color=get_rgba(nsc, nso),
                                         show_background=nbe, background_color=get_rgba(nbc, nbo))
        img, (_, _) = add_text(img, f'"{item["description"]}"', bottom_center + name_height + text_offset, dff,
                               font_size=dfs, font_color=get_rgba(dfc, dfo),
                               show_shadow=dse, shadow_radius=dsr, shadow_color=get_rgba(dsc, dso),
                               show_background=dbe, background_color=get_rgba(dbc, dbo),
                               max_width=43)  # Adjust for wrapped text

        images += [img]

    return images


def save_to_disk(images, image_type, dir="images/output"):
    if not images:
        gr.Warning("No images to save.")
        return

    date = datetime.now().strftime("%m%d%Y")

    dir = f"{dir}/{date}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    for index, image in enumerate(images.root):
        image = image.image
        if image.path is None:
            gr.Warning(f"Image at index {index} has no path... this shouldn't happen.")
            continue

        filename = image.orig_name
        if filename is None:
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"output_{index}_{timestamp}"
        filename += f".{image_type}"

        filepath = os.path.join(dir, filename)

        img = cv2.imread(image.path, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(filepath, img)

    gr.Info(f"Saved generated images to {dir}.")


def reset_parameters():
    pass


with gr.Blocks() as demo:
    gr.Markdown("# inf0 TikTok Tools")

    with gr.Tab("Listicle Template"):
        gr.Markdown("Create images in the style of those 'Your birth month is your ___' TikToks.")
        with gr.Tab("Generate"):
            gr.Markdown("Generate the listicle, JSON file, and images to use here using Chat-GPT.")

            with gr.Row():
                api_key = gr.Textbox(label="OpenAI API Key",
                                     placeholder="Leave empty to use the OPENAI_API_KEY environment variable.",
                                     lines=1, interactive=True)
                # TODO: Update with gpt-4?
                api_text_model = gr.Dropdown(["gpt-3.5-turbo", "gpt-4"], label="API Model", value="gpt-3.5-turbo",
                                             interactive=True)
                # TODO: Update with dall-e-3?
                api_image_model = gr.Dropdown(["dall-e-2", "dall-e-3"], label="API Image Model", value="dall-e-2",
                                              interactive=True)

            with gr.Row(equal_height=False):
                with gr.Group():
                    with gr.Group():
                        with gr.Row():
                            topic = gr.Dropdown(["scary rooms", "fantasy environments"], label="Topic",
                                                value="scary rooms", interactive=True, info="The topic of the listicle (keep it short).")
                            association = gr.Dropdown(["birth month", "astrological sign"], label="Association", value="birth month", info="What to associate each item with.")
                            rating_type = gr.Dropdown(["survivability", "comfortability"], label="Rating",
                                                      value="comfortability", interactive=True, allow_custom_value=True)
                            num_items = gr.Number(12, label="Number of list items", minimum=1, maximum=25, step=1,
                                                  interactive=True)
                        details = gr.TextArea(label="Additional Details",
                                              placeholder="Additional details about the listicle.",
                                              lines=3)
                        generate_artifacts = gr.Checkbox(False, label="Generate Artifacts", interactive=True, info="Generate JSON and images for the listicle.")

                    generate_listicle_button = gr.Button("Generate Listicle", variant="primary")


                    def generate_listicle(api_key, api_text_model, api_image_model, number_of_items, topic, association,
                                          rating_type, details="", generate_artifacts=False):
                        if api_key is None or api_key == "":
                            api_key = os.environ.get("OPENAI_API_KEY")
                        if api_key is None or api_key == "":
                            gr.Warning("No OPENAI_API_KEY environment variable set.")
                            return None, None, None
                        listicle_json = None
                        listicle_images = []
                        openai = OpenAI(api_key=api_key)

                        # TODO: Make prompts more efficient (token) to save on costs.
                        additional_details = ""
                        if association is not None and association != "":
                            additional_details += f"Associate each item with a(n) {association}."
                        additional_details += details

                        messages = [
                            {"role": "user", "content": f"Generate a list of {number_of_items} {topic}. ONLY generate {number_of_items} items. "
                                                        f"For each item, add a unique name and description, and provide"
                                                        f" a rating from 0-100 for each based off {rating_type}. "
                                                        f"{additional_details}"},
                        ]

                        listicle_response_message = [
                            {"role": "system",
                             "content": f"You are a TikTok creator that is creating a listicle of {topic}."},
                        ]
                        listicle_response_message.extend(messages)
                        listicle_response = openai.chat.completions.create(
                            model=api_text_model,
                            messages=listicle_response_message,
                        )

                        resp = listicle_response.choices[0]
                        if resp.finish_reason != "stop":
                            if resp.finish_reason == "length":
                                gr.Warning(
                                    f"finish_reason: {resp.finish_reason}. The maximum number of tokens specified in the request was reached.")
                                return None, None, None
                            elif resp.finish_reason == "content_filter":
                                gr.Warning(
                                    f"finish_reason: {resp.finish_reason}. The content was omitted due to a flag from OpenAI's content filters.")
                                return None, None, None

                        listicle_content = resp.message.content
                        if listicle_content is None or listicle_content == "":
                            gr.Warning("No listicle content was generated.")
                            return None, None, None

                        messages.append({"role": "assistant", "content": listicle_content})

                        if generate_artifacts:
                            # TODO: Remove if/once not needed. Use the 1106 previews to use json response formatting.
                            # https://github.com/openai/openai-python/issues/887#issuecomment-1829085545
                            json_model = "gpt-4-1106-preview" if api_text_model == "gpt-4" else "gpt-3.5-turbo-1106"

                            listicle_json_messages = [
                                {"role": "system",
                                 "content": "You are a master at formatting pre-generated listicles into JSON."},
                            ]

                            listicle_json_messages.extend(messages)
                            json_format = "{name: <string>, description: <string>, rating: <int>"  # TODO: image names, when saved, should be kebab-cased 'name' field (i'll need to update the image processing code as well)
                            if association is not None and association != "":  # Add association field if provided
                                json_format += ", association: <string>"
                            json_format += "}"
                            message = f"Format the listicle into JSON. For the items, store as a list named 'items' with the content format: {json_format}."
                            if rating_type is not None and rating_type != "":
                                message += (f"Include a top-level field `rating_type: <string>` with what the rating "
                                            f"represents.")
                            listicle_json_messages.extend([{
                                "role": "user",
                                "content": message,
                            }])
                            listicle_json_response = openai.chat.completions.create(
                                model=json_model,
                                response_format={"type": "json_object"},
                                messages=listicle_json_messages,
                            )
                            resp = listicle_json_response.choices[0]
                            if resp.finish_reason != "stop":
                                if resp.finish_reason == "length":
                                    gr.Warning(
                                        f"finish_reason: {resp.finish_reason}. The maximum number of tokens specified in the request was reached.")
                                    return listicle_content, None, None
                                elif resp.finish_reason == "content_filter":
                                    gr.Warning(
                                        f"finish_reason: {resp.finish_reason}. The content was omitted due to a flag from OpenAI's content filters.")
                                    return listicle_content, None, None

                            if resp.message.content is None or resp.message.content == "":
                                gr.Warning("No listicle JSON was generated.")
                                return listicle_content, None, None

                            listicle_json = resp.message.content

                            # TODO - image generation
                            # Parse the listicle_json string as JSON and generate images for each 'item's 'description' field.
                            listicle_json_data = json.loads(listicle_json)

                            for item in listicle_json_data["items"]:
                                description = item["description"]
                                name = item["name"]

                                listicle_image_response = openai.images.generate(
                                    prompt=f"Generate an image depicting {name}. Described as: {description}. NO TEXT.",
                                    model=api_image_model,
                                    size="1024x1792" if api_image_model == "dall-e-3" else "1024x1024",
                                    n=1,
                                    quality="hd" if api_image_model == "dall-e-3" else "standard",
                                    response_format="b64_json",
                                )
                                b64_json_image = listicle_image_response.data[0].b64_json
                                if b64_json_image is None or b64_json_image == "":
                                    gr.Warning(f"No image generated for {name}.")
                                    continue
                                img_bytes = base64.b64decode(b64_json_image)
                                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                                listicle_images += [img]

                        return listicle_content, listicle_json, listicle_images

                    def save_artifacts(listicle_images, image_type, json_data):
                        if not listicle_images or len(listicle_images.root) == 0:
                            gr.Warning("No images to save.")
                            return
                        if not json_data or len(json_data) == 0:
                            gr.Warning("No JSON data to save.")
                            return

                        artifact_id = uuid.uuid4()

                        date = datetime.now().strftime("%m%d%Y")
                        # Saving as an 'input' with a unique id so users can easily find the images they generated for
                        # processing.
                        dir = f"images/input/{date}/{artifact_id}"
                        if not os.path.exists(dir):
                            os.makedirs(dir)

                        for index, image in enumerate(listicle_images.root):
                            image = image.image
                            if image.path is None:
                                gr.Warning(f"Image at index {index} has no path... this shouldn't happen.")
                                continue

                            # Get the name of the image from the JSON data
                            filename = f"{index}.{image_type}"
                            filepath = os.path.join(dir, filename)

                            img = cv2.imread(image.path, cv2.IMREAD_UNCHANGED)
                            cv2.imwrite(filepath, img)

                        # Save the JSON data
                        json_filepath = os.path.join(dir, "data.json")
                        with open(json_filepath, "w") as file:
                            json_data = json.loads(json_data)
                            json.dump(json_data, file, indent=4)

                        gr.Info(f"Saved generated artifacts to {dir}.")

                with gr.Column():
                    listicle_output = gr.TextArea(label="Listicle", show_label=False,
                                                  placeholder="Your generated Listicle will appear here.", lines=15,
                                                  max_lines=15,
                                                  interactive=False)
                    listicle_json_output = gr.Code("{}", language="json", label="JSON", lines=10, interactive=False)
                    listicle_image_output = gr.Gallery(label="Generated Images")
                    with gr.Column():
                        with gr.Group():
                            image_type = gr.Dropdown(["png", "jpg", "webp"], label="Image Type", value="png", interactive=True)
                            download_artifacts_button = gr.Button("Download Artifacts", variant="primary")
                        with gr.Row():
                            send_artifacts_to_single = gr.Button("Send Artifacts to Single Processing", variant="secondary")
                            send_artifacts_to_batch = gr.Button("Send Artifacts to Batch Processing", variant="secondary")

                generate_listicle_button.click(generate_listicle,
                                               inputs=[api_key, api_text_model, api_image_model, num_items, topic,
                                                       association, rating_type, details, generate_artifacts],
                                               outputs=[listicle_output, listicle_json_output, listicle_image_output])
                download_artifacts_button.click(
                    save_artifacts,
                    inputs=[listicle_image_output, image_type, listicle_json_output],
                    outputs=[]
                )
        with gr.Tab("Manual"):
            gr.Markdown("Create images one-by-one.")
            gr.Markdown("TODO")
        with gr.Tab("Batch"):
            with gr.Column():
                gr.Markdown("# Input")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        input_images = gr.File(file_types=["image"], file_count="multiple", label="Upload Image(s)")
                    with gr.Column():
                        input_json = gr.File(file_types=[".json"], file_count="single", label="Upload JSON",
                                             interactive=True)
                        validate_json_button = gr.Button("Validate JSON", variant="secondary")
                with gr.Accordion("Important Notes", open=False):
                    gr.Markdown(
                        "When using the automatic JSON parser, make sure that the number of images and the number of "
                        "items in the JSON match.")
                    gr.Markdown("""JSON **data** should be in the following format
                                ```json
                                {
                                    "month": <string>,
                                    "name": <string>,
                                    "description": <string>,
                                    "rating": <int>,
                                    "image": <string>, // <- The name of the image file this refers to.
                                }
                                ```
                                """)
                with gr.Row():
                    reset_parameters_button = gr.Button("Reset Parameters to Default", variant="secondary")
                    process_button = gr.Button("Process", variant="primary")

            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("# Parameters")
                    with gr.Row(equal_height=False):
                        _, _, (nff, nfs, nfc, nfo), (nse, nsc, nso, nsr), (nbe, nbc, nbo) = image_editor_parameters(
                            "Name",
                            default_font_size=117)
                        _, _, (dff, dfs, dfc, dfo), (dse, dsc, dso, dsr), (dbe, dbc, dbo) = image_editor_parameters(
                            "Description",
                            default_font_size=42)
                    with gr.Row(equal_height=False):
                        _, _, (mff, mfs, mfc, mfo), (mse, msc, mso, msr), (mbe, mbc, mbo) = image_editor_parameters(
                            "Month",
                            default_font_size=145)


                        def rating_text_input_fn():
                            return gr.Dropdown(["Comfortability", "Survivability"],
                                               label="Rating Text", value="Comfortability", interactive=True,
                                               allow_custom_value=True)


                        rating_text, _, (rff, rfs, rfc, rfo), (rse, rsc, rso, rsr), (
                            rbe, rbc, rbo) = image_editor_parameters("Rating",
                                                                     default_font_size=55,
                                                                     pre_opt_render_fn=rating_text_input_fn)

                with gr.Column(scale=1):
                    gr.Markdown("# Output")
                    output_preview = gr.Gallery(label="Previews")
                    with gr.Group():
                        image_type = gr.Dropdown(["png", "jpg", "webp"], label="Image Type", value="png",
                                                 interactive=True)
                        save_button = gr.Button("Save to Disk", variant="primary")

    process_button.click(process, inputs=[input_images, input_json,
                                          nff, nfs, nfc, nfo, nse, nsc, nso, nsr, nbe, nbc, nbo,
                                          dff, dfs, dfc, dfo, dse, dsc, dso, dsr, dbe, dbc, dbo,
                                          mff, mfs, mfc, mfo, mse, msc, mso, msr, mbe, mbc, mbo,
                                          rating_text, rff, rfs, rfc, rfo, rse, rsc, rso, rsr, rbe, rbc, rbo
                                          ], outputs=[output_preview])
    validate_json_button.click(validate_json, inputs=[input_json], outputs=[])
    save_button.click(save_to_disk, inputs=[output_preview, image_type], outputs=[])
    reset_parameters_button.click(reset_parameters, inputs=[], outputs=[])

if __name__ == "__main__":
    demo.launch()
