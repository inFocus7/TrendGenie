import gradio as gr
import utils.gradio as gru
from ui.music.utils import *
import processing.video as video_processing
import processing.image as image_processing
import ui.components.openai as openai_components


def render_music_section():
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


def render_generate_cover():
    api_key, _, api_image_model = openai_components.render_openai_setup(show_text_model=False)
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

    generate_image_button.click(generate_cover_image, inputs=[api_key, api_image_model, image_prompt],
                                outputs=[image_output])
    save_image_button.click(image_processing.save_image_to_disk, inputs=[image_output, image_name, image_suffix],
                            outputs=[])

    return send_to_process_button, send_to_create_video_button, image_output


def render_process_cover():
    with gr.Column():
        gr.Markdown("## Input")
        with gr.Group():
            # Sadly we can't use RGBA here due to JPEG images not supporting alpha and breaking
            input_image = gr.Image(sources=["upload"], label="Cover Image", type="filepath", show_download_button=False,
                                   scale=2, elem_classes=["single-image-input"], image_mode="RGB")

        with gr.Row(equal_height=False):
            with gr.Group():
                artist_name = gr.Textbox(label="Artist Name", lines=1, max_lines=1, scale=1)
                (af_family, af_style, afs, afc, afo), (ase, asc, aso, asr), (
                    abe, abc, abo) = image_processing.render_text_editor_parameters("Artist Text Parameters")

            with gr.Group():
                song_name = gr.Textbox(label="Song Title", lines=1, max_lines=1, scale=2)
                (sf_family, sf_style, sfs, sfc, sfo), (sse, ssc, sso, ssr), (
                    sbe, sbc, sbo) = image_processing.render_text_editor_parameters("Song Text Parameters")

        process_button = gr.Button("Process", variant="primary")

        gr.Markdown("## Output")
        with gr.Column():
            with gr.Group():
                image_output, image_name, image_suffix, save_image_button = image_processing.render_image_output()
                send_to_create_video_button = gr.Button("Send Image to 'Create Music Video'", variant="secondary")

    process_button.click(process, inputs=[input_image, artist_name, song_name,
                                          af_family, af_style, afs, afc, afo, ase, asc, aso, asr, abe, abc, abo,
                                          sf_family, sf_style, sfs, sfc, sfo, sse, ssc, sso, ssr, sbe, sbc, sbo],
                         outputs=[image_output])
    save_image_button.click(image_processing.save_image_to_disk,
                            inputs=[image_output, image_name, image_suffix], outputs=[])

    return send_to_create_video_button, input_image, image_output


def render_music_video_creation():
    gr.Markdown("## Input")
    with gr.Row(equal_height=False):
        # Sadly we can't use RGBA here due to JPEG images not supporting alpha and breaking. It would be nice if Gradio
        # supported a way to convert to PNG before processing, but it doesn't seem to.
        cover_image = gr.Image(label="Cover Image", type="filepath", sources=["upload"],
                               show_share_button=False, show_download_button=False, scale=2, image_mode="RGB")
        audio_filepath = gr.File(label="Audio", file_types=["audio"], scale=1, height=100)
    with gr.Column():
        background_color, background_opacity = gru.render_color_opacity_picker(default_name_label="Background")
        with gr.Group():
            artist_name = gr.Textbox(label="Artist Name", lines=1, max_lines=1, scale=1)
            (artist_ffamily, artist_fstyle, artist_fsize, artist_fcolor, artist_fopacity), (ase, asc, aso, asr), (
                abe, abc, abo) = image_processing.render_text_editor_parameters("Text Parameters")
        with gr.Group():
            song_title = gr.Textbox(label="Song Title", lines=1, max_lines=1, scale=2)
            (song_ffamily, song_fstyle, song_fsize, song_fcolor, song_fopacity), (sse, ssc, sso, ssr), (
                sbe, sbc, sbo) = image_processing.render_text_editor_parameters("Text Parameters")
        with gr.Column():
            # Defaulting to 1. It's a still image, but may expand by adding some effects (grain, and not sure what else)
            fps = gr.Number(value=1, label="FPS", minimum=1, maximum=144)

            with gr.Group():
                generate_audio_visualizer_button = gr.Checkbox(value=False, label="Generate Audio Visualizer",
                                                               interactive=True)
                with gr.Group() as audio_visualizer_group:
                    audio_visualizer_color, audio_visualizer_opacity = \
                        gru.render_color_opacity_picker("Audio Visualizer")
                    with gr.Group():
                        with gr.Row():
                            audio_visualizer_num_rows = gr.Number(value=90, label="Number of Rows",
                                                                  minimum=1, maximum=100)
                            audio_visualizer_num_columns = gr.Number(value=65, label="Number of Columns",
                                                                     minimum=1, maximum=100)
                        with gr.Row():
                            audio_visualizer_min_size = gr.Number(value=1, label="Minimum Size", minimum=1, maximum=100)
                            audio_visualizer_max_size = gr.Number(value=7, label="Maximum Size", minimum=1, maximum=200)
                    # Must be a PNG file to support transparency. The idea for this is more-so to have shapes that can
                    # be rendered for the visualizer, and ideally they have transparent backgrounds, so using RGBA.
                    audio_visualizer_drawing = gr.Image(label="Visualizer Drawing (png)", type="filepath",
                                                        sources=["upload"], show_share_button=False,
                                                        show_download_button=False, scale=2, height=150,
                                                        image_mode="RGBA")
            gru.bind_checkbox_to_visibility(generate_audio_visualizer_button, audio_visualizer_group)

    create_video_button = gr.Button("Create Music Video", variant="primary")

    gr.Markdown("## Output")
    with gr.Group():
        video_output, video_name, video_suffix, save_video_button = video_processing.render_video_output()

    create_video_button.click(create_music_video, inputs=[cover_image, audio_filepath, fps,
                                                          artist_name, artist_ffamily, artist_fstyle, artist_fsize,
                                                          artist_fcolor, artist_fopacity, ase, asc, aso, asr, abe, abc,
                                                          abo, song_title, song_ffamily, song_fstyle, song_fsize,
                                                          song_fcolor, song_fopacity, sse, ssc, sso, ssr, sbe, sbc, sbo,
                                                          background_color, background_opacity,
                                                          generate_audio_visualizer_button, audio_visualizer_color,
                                                          audio_visualizer_opacity, audio_visualizer_drawing,
                                                          audio_visualizer_num_rows, audio_visualizer_num_columns,
                                                          audio_visualizer_min_size, audio_visualizer_max_size],
                              outputs=[video_output])
    save_video_button.click(video_processing.save_video_to_disk,
                            inputs=[video_output, video_name, video_suffix], outputs=[])

    return cover_image
