import utils.gradio as gru
import gradio as gr
import processing.image as image_processing
import json
import ui.listicles.utils as listicle_utils


def render_image_editor_parameters(name):
    with gr.Accordion(label=name):
        with gr.Column():
            font_family, font_style, font_color, font_opacity, font_size = gru.render_font_picker()
            with gr.Group():
                drop_shadow_checkbox = gr.Checkbox(False, label="Enable Drop Shadow")
                with gr.Group(visible=drop_shadow_checkbox.value) as additional_options:
                    drop_shadow_color, drop_shadow_opacity = gru.render_color_opacity_picker()
                    drop_shadow_radius = gr.Number(0, label="Shadow Radius")
                    gru.bind_checkbox_to_visibility(drop_shadow_checkbox, additional_options)
            with gr.Group():
                background_checkbox = gr.Checkbox(False, label="Enable Background")
                with gr.Group(visible=background_checkbox.value) as additional_options:
                    background_color, background_opacity = gru.render_color_opacity_picker()
                    gru.bind_checkbox_to_visibility(background_checkbox, additional_options)

    return ((font_family, font_style, font_size, font_color, font_opacity),
            (drop_shadow_checkbox, drop_shadow_color, drop_shadow_opacity, drop_shadow_radius),
            (background_checkbox, background_color, background_opacity))


def render_listicles_section():
    gr.Markdown("Create images in the style of those 'Your birth month is your ___' TikToks.")
    with gr.Tab("Generate Images"):
        _, send_artifacts_to_batch_button ,listicle_image_output, listicle_json_output  = render_generate_section()
    with gr.Tab("Single Image Processing"):
        render_single_section()
    with gr.Tab("Batch Image Processing"):
        input_batch_images, input_batch_json = render_batch_section()

    send_artifacts_to_batch_button.click(
                listicle_utils.send_artifacts_to_batch,
                inputs=[listicle_image_output, listicle_json_output],
                outputs=[input_batch_images, input_batch_json]
            )


def render_single_section():
    gr.Markdown("Create images one-by-one.")
    gr.Markdown("NotImplemented")


def render_batch_section():
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
                (nf_family, nf_style, nfs, nfc, nfo), (nse, nsc, nso, nsr), (
                    nbe, nbc, nbo) = render_image_editor_parameters("Name")
                (df_family, df_style, dfs, dfc, dfo), (dse, dsc, dso, dsr), (
                    dbe, dbc, dbo) = render_image_editor_parameters("Description")
            with gr.Row(equal_height=False):
                (mf_family, mf_style, mfs, mfc, mfo), (mse, msc, mso, msr), (
                    mbe, mbc, mbo) = render_image_editor_parameters("Association")
                (rf_family, rf_style, rfs, rfc, rfo), (rse, rsc, rso, rsr), (
                    rbe, rbc, rbo) = render_image_editor_parameters("Rating")

        with gr.Column(scale=1):
            gr.Markdown("# Output")
            output_preview = gr.Gallery(label="Previews")
            with gr.Group():
                image_type = gr.Dropdown(["png", "jpg", "webp"], label="Image Type", value="png",
                                         interactive=True)
                save_button = gr.Button("Save to Disk", variant="primary")

    validate_json_button.click(listicle_utils.validate_json, inputs=[input_batch_json], outputs=[])
    save_button.click(image_processing.save_images_to_disk, inputs=[output_preview, image_type],
                                    outputs=[])
    process_button.click(listicle_utils.process, inputs=[input_batch_images, input_batch_json,
                                                               nf_family, nf_style, nfs, nfc, nfo, nse, nsc, nso, nsr,
                                                               nbe,
                                                               nbc, nbo,
                                                               df_family, df_style, dfs, dfc, dfo, dse, dsc, dso, dsr,
                                                               dbe,
                                                               dbc, dbo,
                                                               mf_family, mf_style, mfs, mfc, mfo, mse, msc, mso, msr,
                                                               mbe,
                                                               mbc, mbo,
                                                               rf_family, rf_style, rfs, rfc, rfo, rse, rsc, rso, rsr,
                                                               rbe,
                                                               rbc, rbo
                                                               ], outputs=[output_preview])

    return input_batch_images, input_batch_json


def render_generate_section():
    gr.Markdown("Generate the listicle, JSON file, and images to use here using Chat-GPT.")
    with gr.Row():
        api_key = gr.Textbox(label="OpenAI API Key",
                             placeholder="Leave empty to use the OPENAI_API_KEY environment variable.",
                             lines=1, interactive=True, type="password")
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
                    num_items = gr.Number(12, label="Number of items", minimum=1, maximum=25, step=1,
                                          interactive=True)
                details = gr.TextArea(label="Additional Details",
                                      placeholder="Additional details about the listicle.",
                                      lines=3)
                generate_artifacts = gr.Checkbox(False, label="Generate Artifacts", interactive=True,
                                                 info="Generate JSON and images for the listicle.")

            generate_listicle_button = gr.Button("Generate Listicle", variant="primary")

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
        generate_listicle_button.click(listicle_utils.generate_listicle,
                                       inputs=[api_key, api_text_model, api_image_model, num_items, topic,
                                               association, rating_type, details, generate_artifacts],
                                       outputs=[listicle_output, listicle_json_output, listicle_image_output])
        download_artifacts_button.click(
            listicle_utils.save_artifacts,
            inputs=[listicle_image_output, image_type, listicle_json_output],
            outputs=[]
        )

    return send_artifacts_to_single_button, send_artifacts_to_batch_button, listicle_image_output, listicle_json_output