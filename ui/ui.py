import json
import gradio as gr
import os
import glob
import utils.image as image_utils
import processing.image as image_processing
import api.chatgpt as chatgpt_api
import inflect
from fontTools.ttLib import TTFont

p = inflect.engine()

font_files = []

fonts_dirs = [
    # MacOS
    "/Library/Fonts", "~/Library/Fonts", "System/Library/Fonts",
    # Linux
    "/usr/share/fonts", "~/.fonts",
    # Windows
    "C:\\Windows\\Fonts"
]
for fonts_dir in fonts_dirs:
    fonts_dir = os.path.expanduser(fonts_dir)
    if not os.path.exists(fonts_dir):
        continue
    font_files += glob.glob(os.path.join(fonts_dir, "**/*.ttf"), recursive=True)
    font_files += glob.glob(os.path.join(fonts_dir, "**/*.otf"), recursive=True)

font_families = {}
for font_file in font_files:
    font = TTFont(font_file)
    name = font['name']
    family_name = ""
    style_name = ""
    for record in name.names:
        if record.nameID == 1 and b'\000' in record.string:
            family_name = record.string.decode('utf-16-be').rstrip('\0')
        elif record.nameID == 2 and b'\000' in record.string:
            style_name = record.string.decode('utf-16-be').rstrip('\0')
    if family_name and style_name:
        if family_name not in font_families:
            font_families[family_name] = {}
        font_families[family_name][style_name] = font_file

DEBUG = False


def render_color_opacity_picker():
    with gr.Group():
        with gr.Row():
            color = gr.ColorPicker(label="Font Color", scale=1)
            opacity = gr.Slider(0, 100, value=100, label="Opacity", scale=2)

    return color, opacity


def bind_checkbox_to_visibility(checkbox, group):
    checkbox.change(
        lambda state: gr.Group(visible=state),
        inputs=checkbox,
        outputs=group
    )


def render_image_editor_parameters(name, default_font_size=55):
    with gr.Accordion(label=name):
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    font_families_list = list(font_families.keys())
                    initial_font_family = font_families_list[0]
                    font_family = gr.Dropdown(font_families_list, value=initial_font_family, label="Font Family")
                    font_styles_list = list(font_families[initial_font_family].keys())
                    font_style = gr.Dropdown(font_styles_list, value=font_styles_list[0], label="Font Style")

                def update_font_styles(selected_font_family):
                    if selected_font_family is None or selected_font_family == "":
                        return []
                    font_syles = list(font_families[selected_font_family].keys())
                    return gr.Dropdown(font_syles, value=font_syles[0], label="Font Style")

                font_family.change(update_font_styles, inputs=[font_family], outputs=[font_style])
            with gr.Group():
                font_color, font_opacity = render_color_opacity_picker()
                font_size = gr.Number(default_font_size, label="Font Size")
            with gr.Group():
                drop_shadow_checkbox = gr.Checkbox(False, label="Enable Drop Shadow")
                with gr.Group(visible=drop_shadow_checkbox.value) as additional_options:
                    drop_shadow_color, drop_shadow_opacity = render_color_opacity_picker()
                    drop_shadow_radius = gr.Number(0, label="Shadow Radius")
                    bind_checkbox_to_visibility(drop_shadow_checkbox, additional_options)
            with gr.Group():
                background_checkbox = gr.Checkbox(False, label="Enable Background")
                with gr.Group(visible=background_checkbox.value) as additional_options:
                    background_color, background_opacity = render_color_opacity_picker()
                    bind_checkbox_to_visibility(background_checkbox, additional_options)

    return ((font_family, font_style, font_size, font_color, font_opacity),
            (drop_shadow_checkbox, drop_shadow_color, drop_shadow_opacity, drop_shadow_radius),
            (background_checkbox, background_color, background_opacity))


def print_parameters(name, ff, fs, fc, fo, se, sc, so, sr, be, bc, bo):
    return f"""- {name}
> Font: Type: {ff}, Size: {fs}, Color: {fc}, Opacity: {fo}
> Drop Shadow: Enabled: {se}, Color: {sc}, Opacity: {so}, Radius: {sr}
> Background: Enabled: {be}, Color: {bc}, Opacity: {bo}
    """


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

    nff = font_families[nf_family][nf_style]
    dff = font_families[df_family][df_style]
    mff = font_families[mf_family][mf_style]
    rff = font_families[rf_family][rf_style]

    if DEBUG:
        print(f"""Beginning processing with the following parameters...
        {print_parameters("Name", nff, nfs, nfc, nfo, nse, nsc, nso, nsr, nbe, nbc, nbo)}
        {print_parameters("Description", dff, dfs, dfc, dfo, dse, dsc, dso, dsr, dbe, dbc, dbo)}
        {print_parameters("Association", mff, mfs, mfc, mfo, mse, msc, mso, msr, mbe, mbc, mbo)}
        {print_parameters("Rating", rff, rfs, rfc, rfo, rse, rsc, rso, rsr, rbe, rbc, rbo)}
        """)

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

        img = image_processing.read_image_from_disk(image_file.name)
        item = json_dict[img_name]

        # Calculate positions for the text
        top_center = int(img.shape[0] * 0.13)
        bottom_center = int(img.shape[0] * 0.70)

        # Add association and rating at the top center, one above the other
        img, (_, association_height) = image_processing.add_text(img, item["association"], top_center, mff,
                                                                 font_size=mfs,
                                                                 font_color=image_utils.get_rgba(mfc, mfo),
                                                                 show_shadow=mse,
                                                                 shadow_radius=msr,
                                                                 shadow_color=image_utils.get_rgba(msc, mso),
                                                                 show_background=mbe,
                                                                 background_color=image_utils.get_rgba(mbc, mbo))

        img, (_, _) = image_processing.add_text(img, f'{json_data["rating_type"]}: {item["rating"]}%',
                                                top_center + association_height + rating_offset,
                                                rff, font_size=rfs, font_color=image_utils.get_rgba(rfc, rfo),
                                                show_shadow=rse, shadow_radius=rsr,
                                                shadow_color=image_utils.get_rgba(rsc, rso),
                                                show_background=rbe, background_color=image_utils.get_rgba(rbc, rbo))

        # Add name and description at the bottom center, one above the other
        img, (_, name_height) = image_processing.add_text(img, item["name"], bottom_center, nff, font_size=nfs,
                                                          font_color=image_utils.get_rgba(nfc, nfo),
                                                          max_width=15,
                                                          show_shadow=nse, shadow_radius=nsr,
                                                          shadow_color=image_utils.get_rgba(nsc, nso),
                                                          show_background=nbe,
                                                          background_color=image_utils.get_rgba(nbc, nbo))
        img, (_, _) = image_processing.add_text(img, f'"{item["description"]}"',
                                                bottom_center + name_height + text_offset, dff,
                                                font_size=dfs, font_color=image_utils.get_rgba(dfc, dfo),
                                                show_shadow=dse, shadow_radius=dsr,
                                                shadow_color=image_utils.get_rgba(dsc, dso),
                                                show_background=dbe, background_color=image_utils.get_rgba(dbc, dbo),
                                                max_width=43)  # Adjust for wrapped text

        images += [img]

    return images


# Read the styles.css file and add it to the page.
css_file = os.path.join(os.path.dirname(__file__), "styles.css")
with open(css_file, "r") as file:
    css = file.read()

# gr.themes.Soft() vs gr.themes.Monochrome()? 🤔
with gr.Blocks(theme=gr.themes.Soft(), css=css) as WebApp:
    # Add css to center the items
    with gr.Column(elem_id="header"):
        gr.Image("static/logo-v2.png", label="Logo", show_label=False, image_mode="RGBA", container=False,
                 show_share_button=False, show_download_button=False, width=50, elem_id="header-logo")
        gr.Markdown("# TrendGenie", elem_id="header-title")
        gr.Markdown("## Your content creation assistant.", elem_id="header-subtitle")

    with gr.Tab("Listicle Template"):
        gr.Markdown("Create images in the style of those 'Your birth month is your ___' TikToks.")
        with gr.Tab("Generate"):
            gr.Markdown("Generate the listicle, JSON file, and images to use here using Chat-GPT.")
            with gr.Row():
                api_key = gr.Textbox(label="OpenAI API Key",
                                     placeholder="Leave empty to use the OPENAI_API_KEY environment variable.",
                                     lines=1, interactive=True)
                api_text_model = gr.Dropdown(["gpt-3.5-turbo", "gpt-4"], label="API Model", value="gpt-3.5-turbo",
                                             interactive=True)
                api_image_model = gr.Dropdown(["dall-e-2", "dall-e-3"], label="API Image Model", value="dall-e-2",
                                              interactive=True)
            with gr.Row(equal_height=False):
                with gr.Group():
                    with gr.Group():
                        with gr.Row():
                            topic = gr.Dropdown(["scary rooms", "fantasy environments"], label="Topic",
                                                value="scary rooms", interactive=True, allow_custom_value=True,
                                                info="The topic of the listicle. (noun)")
                            association = gr.Dropdown(["birth month", "astrological sign"], label="Association",
                                                      value="birth month", info="What to associate each item with.",
                                                      allow_custom_value=True)
                            rating_type = gr.Dropdown(["survivability", "comfortability"], label="Rating",
                                                      info="What the rating given represents.", value="comfortability",
                                                      interactive=True, allow_custom_value=True)
                            num_items = gr.Number(12, label="Number of list items", minimum=1, maximum=25, step=1,
                                                  interactive=True)
                        details = gr.TextArea(label="Additional Details",
                                              placeholder="Additional details about the listicle.",
                                              lines=3)
                        generate_artifacts = gr.Checkbox(False, label="Generate Artifacts", interactive=True,
                                                         info="Generate JSON and images for the listicle.")

                    generate_listicle_button = gr.Button("Generate Listicle", variant="primary")


                    def generate_listicle(api_key, api_text_model, api_image_model, number_of_items, topic, association,
                                          rating_type, details="", generate_artifacts=False):
                        openai = chatgpt_api.get_openai_client(api_key)
                        if openai is None:
                            gr.Warning("No OpenAI client. Cannot generate listicle.")
                            return None, None, None

                        listicle_images = []
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
                            message = f"Format the listicle into JSON. For the items, store as a list named 'items' with the content format: {json_format}."
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
                                image_url = chatgpt_api.get_image_response(openai, api_image_model, prompt)
                                if image_url is None or image_url == "":
                                    continue

                                item["image"] = chatgpt_api.url_to_gradio_image_name(image_url)
                                listicle_images.append(image_url)

                        # Because the json data was updated above, we need to re-serialize it.
                        listicle_json = json.dumps(listicle_json_data, indent=4)

                        return listicle_content, listicle_json, listicle_images


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

                with gr.Column():
                    listicle_output = gr.TextArea(label="Listicle", show_label=False,
                                                  placeholder="Your generated Listicle will appear here.", lines=15,
                                                  max_lines=15, interactive=False)
                    listicle_json_output = gr.Code("{}", language="json", label="JSON", lines=10, interactive=False)
                    listicle_image_output = gr.Gallery(label="Generated Images")
                    with gr.Column():
                        with gr.Group():
                            image_type = gr.Dropdown(["png", "jpg", "webp"], label="Image Type", value="png",
                                                     interactive=True)
                            download_artifacts_button = gr.Button("Download Artifacts", variant="primary")
                        with gr.Group():
                            with gr.Row():
                                send_artifacts_to_single_button = gr.Button("Send Artifacts to Single Processing",
                                                                            variant="secondary")
                                send_artifacts_to_batch_button = gr.Button("Send Artifacts to Batch Processing",
                                                                           variant="secondary")
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
            gr.Markdown("NotImplemented")
        with gr.Tab("Batch"):
            with gr.Column():
                gr.Markdown("# Input")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        input_batch_images = gr.File(file_types=["image"], file_count="multiple",
                                                     label="Upload Image(s)")
                    with gr.Column():
                        input_batch_json = gr.Code("{}", language="json", label="Configuration (JSON)", lines=10)
                        with gr.Group():
                            with gr.Row():
                                upload_json = gr.File(label="Upload JSON", file_types=[".json"])
                                set_json_button = gr.Button("Set JSON", variant="secondary")


                        def set_json(json_file):
                            if not json_file:
                                gr.Warning("No JSON file uploaded. Reverse to default.")
                                return input_batch_json.value
                            with open(json_file.name, "r") as file:
                                json_data = json.load(file)
                                json_data = json.dumps(json_data, indent=4)

                            return json_data


                        set_json_button.click(set_json, inputs=[upload_json], outputs=[input_batch_json])
                        with gr.Row():
                            validate_json_button = gr.Button("Validate JSON", variant="secondary")
                with gr.Accordion("Important Notes", open=False):
                    gr.Markdown(
                        "When using the automatic JSON parser, make sure that the number of images and the number of "
                        "items in the JSON match.")
                    gr.Markdown("""JSON **data** should be in the following format
                                ```json
                                {
                                    "rating_type": <string>,
                                    {
                                        "association": <string>,
                                        "name": <string>,
                                        "description": <string>,
                                        "rating": <int>,
                                        "image": <string>, // <- The name of the image file this refers to.
                                    }
                                }
                                ```
                                """)
                with gr.Row():
                    process_button = gr.Button("Process", variant="primary")

            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("# Parameters")
                    with gr.Row(equal_height=False):
                        (nf_family, nf_style, nfs, nfc, nfo), (nse, nsc, nso, nsr), (nbe, nbc, nbo) = render_image_editor_parameters(
                            "Name", default_font_size=117)
                        (df_family, df_style, dfs, dfc, dfo), (dse, dsc, dso, dsr), (dbe, dbc, dbo) = render_image_editor_parameters(
                            "Description", default_font_size=42)
                    with gr.Row(equal_height=False):
                        (mf_family, mf_style, mfs, mfc, mfo), (mse, msc, mso, msr), (mbe, mbc, mbo) = render_image_editor_parameters(
                            "Association", default_font_size=145)
                        (rf_family, rf_style, rfs, rfc, rfo), (rse, rsc, rso, rsr), (rbe, rbc, rbo) = render_image_editor_parameters(
                            "Rating", default_font_size=55)

                with gr.Column(scale=1):
                    gr.Markdown("# Output")
                    output_preview = gr.Gallery(label="Previews")
                    with gr.Group():
                        image_type = gr.Dropdown(["png", "jpg", "webp"], label="Image Type", value="png",
                                                 interactive=True)
                        save_button = gr.Button("Save to Disk", variant="primary")

        send_artifacts_to_batch_button.click(
            send_artifacts_to_batch,
            inputs=[listicle_image_output, listicle_json_output],
            outputs=[input_batch_images, input_batch_json]
        )

    # Add a footer
    with gr.Group(elem_id="footer"):
        gr.Image("static/hero-face.svg", label="Logo", show_label=False,
                 image_mode="RGBA", container=False, width=50, elem_id="footer-logo",
                 show_download_button=False,show_share_button=False)
        gr.Markdown("**Made by [inf0](https://github.com/infocus7).**", elem_id="footer-text")

    process_button.click(process, inputs=[input_batch_images, input_batch_json,
                                          nf_family, nf_style, nfs, nfc, nfo, nse, nsc, nso, nsr, nbe, nbc, nbo,
                                          df_family, df_style, dfs, dfc, dfo, dse, dsc, dso, dsr, dbe, dbc, dbo,
                                          mf_family, mf_style, mfs, mfc, mfo, mse, msc, mso, msr, mbe, mbc, mbo,
                                          rf_family, rf_style, rfs, rfc, rfo, rse, rsc, rso, rsr, rbe, rbc, rbo
                                          ], outputs=[output_preview])
    validate_json_button.click(validate_json, inputs=[input_batch_json], outputs=[])
    save_button.click(image_processing.save_images_to_disk, inputs=[output_preview, image_type], outputs=[])

# if __name__ == "__main__":
#     WebApp.launch()
