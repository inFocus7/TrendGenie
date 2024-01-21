"""
Tbe interface for the music section of the UI. This is the main piece where we define the Gradio interface components.
"""
import gradio as gr
import utils.gradio as gru
from ui.music.utils import generate_cover_image, process, create_music_video, create_music_video_preview
import processing.video as video_processing
import processing.image as image_processing
import ui.components.openai as openai_components
from utils import dataclasses


def render_music_section() -> None:
    """
    Renders the music cover video section of the UI.
    :return: None
    """
    gru.render_tool_description("Create a cover and a simple video for your music!")
    with gr.Tab("Generate Cover"):
        send_cover_to_process_button, send_cover_to_video_button, generated_image_output_path = render_generate_cover()
    with gr.Tab("Add Text To Image"):
        send_processed_cover_to_video_button, processed_image_input, processed_image_output_path = \
            render_process_cover()
    with gr.Tab("Create Music Video"):
        music_video_cover_image = render_music_video_creation()

    send_processed_cover_to_video_button.click(lambda image: image,
                                               inputs=[processed_image_output_path],
                                               outputs=[music_video_cover_image])
    send_cover_to_process_button.click(lambda image: image,
                                       inputs=[generated_image_output_path],
                                       outputs=[processed_image_input])
    send_cover_to_video_button.click(lambda image: image,
                                     inputs=[generated_image_output_path],
                                     outputs=[music_video_cover_image])


def render_generate_cover() -> (gr.Button, gr.Button, gr.Image):
    """
    Renders the cover generation interface component for the music cover creation section.
    :return: A tuple containing the following Gradio UI components: A button for generating a cover image, a button for
        sending the generated cover image to the "Add Text to Image" section, and an image display component for
        displaying the generated cover image.
    """
    open_ai_components = openai_components.render_openai_setup(show_text_model=False)
    with gr.Row(equal_height=False):
        with gr.Group():
            image_prompt = gr.Textbox(label="Image Prompt", lines=6, max_lines=10)
            generate_image_button = gr.Button(value="Generate Image", variant="primary")
        with gr.Column():
            with gr.Group():
                image_output, image_name, image_suffix, save_image_button = image_processing.render_image_output()
            with gr.Group():
                with gr.Row():
                    send_to_process_button = gr.Button("Send Image to 'Add Text to Image'", variant="secondary")
                    send_to_create_video_button = gr.Button("Send Image to 'Create Music Video'", variant="secondary")

    generate_image_button.click(generate_cover_image, inputs=[open_ai_components.api_key,
                                                              open_ai_components.api_image_model, image_prompt],
                                outputs=[image_output])
    save_image_button.click(image_processing.save_image_to_disk, inputs=[image_output, image_name, image_suffix],
                            outputs=[])

    return send_to_process_button, send_to_create_video_button, image_output


def render_process_cover() -> (gr.Button, gr.Image, gr.Image):
    """
    Renders the cover processing interface component for the music cover creation section. This is where we add text to
    the cover image.
    :return: A tuple containing the following Gradio UI components: A button for processing a cover image, an image
        display component for displaying the cover image before processing, and an image display component for
        displaying the cover image after processing.
    """
    with gr.Column():
        gr.Markdown("## Input")
        with gr.Group():
            input_image = gr.Image(sources=["upload"], label="Cover Image (png)", type="filepath",
                                   show_download_button=False, scale=2, elem_classes=["single-image-input"],
                                   image_mode="RGBA")

        with gr.Row(equal_height=False):
            with gr.Group():
                artist_name = gr.Textbox(label="Artist Name", lines=1, max_lines=1, scale=1)
                artist_font_display = image_processing.render_text_editor_parameters("Artist Text Parameters")

            with gr.Group():
                song_name = gr.Textbox(label="Song Title", lines=1, max_lines=1, scale=2)
                song_font_display = image_processing.render_text_editor_parameters("Song Text Parameters")

        process_button = gr.Button("Process", variant="primary")

        gr.Markdown("## Output")
        with gr.Column():
            with gr.Group():
                image_output, image_name, image_suffix, save_image_button = image_processing.render_image_output()
                send_to_create_video_button = gr.Button("Send Image to 'Create Music Video'", variant="secondary")

    process_button.click(process, inputs=[input_image, artist_name, song_name,
                                          artist_font_display.font.family, artist_font_display.font.style,
                                          artist_font_display.font.size, artist_font_display.font.color,
                                          artist_font_display.font.opacity, artist_font_display.drop_shadow.enabled,
                                          artist_font_display.drop_shadow.color,
                                          artist_font_display.drop_shadow.opacity,
                                          artist_font_display.drop_shadow.radius,
                                          artist_font_display.background.enabled,
                                          artist_font_display.background.color, artist_font_display.background.opacity,
                                          song_font_display.font.family, song_font_display.font.style,
                                          song_font_display.font.size, song_font_display.font.color,
                                          song_font_display.font.opacity, song_font_display.drop_shadow.enabled,
                                          song_font_display.drop_shadow.color, song_font_display.drop_shadow.opacity,
                                          song_font_display.drop_shadow.radius, song_font_display.background.enabled,
                                          song_font_display.background.color, song_font_display.background.opacity],
                         outputs=[image_output])
    save_image_button.click(image_processing.save_image_to_disk,
                            inputs=[image_output, image_name, image_suffix], outputs=[])

    return send_to_create_video_button, input_image, image_output


def render_music_video_creation() -> gr.Image:
    """
    Renders the music video creation interface component for the music cover creation section.
    :return: An image display component for displaying the cover image.
    """
    gr.Markdown("## Input")
    with gr.Row(equal_height=False):
        cover_image = gr.Image(label="Cover Image (png)", type="filepath", sources=["upload"],
                               show_share_button=False, show_download_button=False, scale=2, image_mode="RGBA")
        audio_filepath = gr.File(label="Audio", file_types=["audio"], scale=1, height=100)
    with gr.Column():
        background_color_opacity = gru.render_color_opacity_picker(default_name_label="Background")
        with gr.Group():
            artist_name = gr.Textbox(label="Artist Name", lines=1, max_lines=1, scale=1)
            artist_font_display = image_processing.render_text_editor_parameters("Text Parameters")
        with gr.Group():
            song_title = gr.Textbox(label="Song Title", lines=1, max_lines=1, scale=2)
            song_font_display = image_processing.render_text_editor_parameters("Text Parameters")
        with gr.Column():
            # Defaulting to 1. It's a still image, but may expand by adding some effects (grain, and not sure what else)
            fps = gr.Number(value=1, label="FPS", minimum=1, maximum=144)

            with gr.Group():
                generate_audio_visualizer_button = gr.Checkbox(value=False, label="Generate Audio Visualizer",
                                                               interactive=True)
                with gr.Group() as audio_visualizer_group:
                    audio_visualizer_color_opacity = gru.render_color_opacity_picker("Audio Visualizer")
                    with gr.Group():
                        with gr.Row():
                            audio_visualizer_amount = dataclasses.RowColGradioComponents(
                                row=gr.Number(value=90, label="Number of Rows", minimum=1,
                                              maximum=100),
                                col=gr.Number(value=65, label="Number of Columns", minimum=1,
                                              maximum=100)
                            )
                        with gr.Row():
                            audio_visualizer_dot_size = dataclasses.MinMaxGradioComponents(
                                min=gr.Number(value=1, label="Minimum Size", minimum=1, maximum=100),
                                max=gr.Number(value=7, label="Maximum Size", minimum=1, maximum=200)
                            )
                    audio_visualizer_drawing = gr.Image(label="Visualizer Drawing (png)", type="filepath",
                                                        sources=["upload"], show_share_button=False,
                                                        show_download_button=False, scale=2, height=150,
                                                        image_mode="RGBA")
                    visualizer_overlay_checkbox = gr.Checkbox(value=False, label="Overlay Visualizer on One-Another",
                                                              info="If checked, alpha-blending will be applied, which "
                                                                   "is noticeable on larger pngs where each drawing "
                                                                   "overlaps. This is only important for transparent"
                                                                   "images and is very slow. If the image is not "
                                                                   "transparent, leave this unchecked.")
            gru.bind_checkbox_to_visibility(generate_audio_visualizer_button, audio_visualizer_group)

    with gr.Group():
        with gr.Row():
            create_preview_video_button = gr.Button("Create Preview", variant="secondary")
            preview_seconds = gr.Number(value=5, label="Preview Seconds", minimum=1, maximum=10)

    create_video_button = gr.Button("Create Music Video", variant="primary")

    gr.Markdown("## Output")
    with gr.Group():
        video_data = video_processing.render_video_output()

    create_preview_video_button.click(create_music_video_preview, inputs=[cover_image, audio_filepath, fps,
                                                                          preview_seconds, artist_name,
                                                                          artist_font_display.font.family,
                                                                          artist_font_display.font.style,
                                                                          artist_font_display.font.size,
                                                                          artist_font_display.font.color,
                                                                          artist_font_display.font.opacity,
                                                                          artist_font_display.drop_shadow.enabled,
                                                                          artist_font_display.drop_shadow.color,
                                                                          artist_font_display.drop_shadow.opacity,
                                                                          artist_font_display.drop_shadow.radius,
                                                                          artist_font_display.background.enabled,
                                                                          artist_font_display.background.color,
                                                                          artist_font_display.background.opacity,
                                                                          song_title, song_font_display.font.family,
                                                                          song_font_display.font.style,
                                                                          song_font_display.font.size,
                                                                          song_font_display.font.color,
                                                                          song_font_display.font.opacity,
                                                                          song_font_display.drop_shadow.enabled,
                                                                          song_font_display.drop_shadow.color,
                                                                          song_font_display.drop_shadow.opacity,
                                                                          song_font_display.drop_shadow.radius,
                                                                          song_font_display.background.enabled,
                                                                          song_font_display.background.color,
                                                                          song_font_display.background.opacity,
                                                                          background_color_opacity.color,
                                                                          background_color_opacity.opacity,
                                                                          generate_audio_visualizer_button,
                                                                          audio_visualizer_color_opacity.color,
                                                                          audio_visualizer_color_opacity.opacity,
                                                                          audio_visualizer_drawing,
                                                                          visualizer_overlay_checkbox,
                                                                          audio_visualizer_amount.row,
                                                                          audio_visualizer_amount.col,
                                                                          audio_visualizer_dot_size.min,
                                                                          audio_visualizer_dot_size.max],
                                      outputs=[video_data.video])
    create_video_button.click(create_music_video, inputs=[cover_image, audio_filepath, fps, artist_name,
                                                          artist_font_display.font.family,
                                                          artist_font_display.font.style, artist_font_display.font.size,
                                                          artist_font_display.font.color,
                                                          artist_font_display.font.opacity,
                                                          artist_font_display.drop_shadow.enabled,
                                                          artist_font_display.drop_shadow.color,
                                                          artist_font_display.drop_shadow.opacity,
                                                          artist_font_display.drop_shadow.radius,
                                                          artist_font_display.background.enabled,
                                                          artist_font_display.background.color,
                                                          artist_font_display.background.opacity,
                                                          song_title, song_font_display.font.family,
                                                          song_font_display.font.style, song_font_display.font.size,
                                                          song_font_display.font.color, song_font_display.font.opacity,
                                                          song_font_display.drop_shadow.enabled,
                                                          song_font_display.drop_shadow.color,
                                                          song_font_display.drop_shadow.opacity,
                                                          song_font_display.drop_shadow.radius,
                                                          song_font_display.background.enabled,
                                                          song_font_display.background.color,
                                                          song_font_display.background.opacity,
                                                          background_color_opacity.color,
                                                          background_color_opacity.opacity,
                                                          generate_audio_visualizer_button,
                                                          audio_visualizer_color_opacity.color,
                                                          audio_visualizer_color_opacity.opacity,
                                                          audio_visualizer_drawing, visualizer_overlay_checkbox,
                                                          audio_visualizer_amount.row, audio_visualizer_amount.col,
                                                          audio_visualizer_dot_size.min, audio_visualizer_dot_size.max],
                              outputs=[video_data.video])
    video_data.save.click(video_processing.save_video_to_disk, inputs=[video_data.video, video_data.name,
                                                                       video_data.suffix], outputs=[])

    return cover_image
