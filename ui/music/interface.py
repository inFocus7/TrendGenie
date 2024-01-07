import gradio as gr
import utils.gradio as gru
from ui.music.utils import *
import processing.video as video_processing


def render_music_section():
    gr.Markdown("Create a cover and a simple video for your music!")
    with gr.Tab("Generate Cover"):
        render_generate_cover()
    with gr.Tab("Process Cover Image"):
        render_process_cover()
    with gr.Tab("Create Music Video"):
        render_music_video_creation()


def render_generate_cover():
    gr.Markdown("TODO")


def render_process_cover():
    gr.Markdown("TODO")


def render_music_video_creation():
    gr.Markdown("## Input")
    with gr.Row(equal_height=False):
        cover_image = gr.Image(label="Cover Image", type="filepath", sources=["upload"],
                               show_share_button=False, show_download_button=False, scale=2)
        audio_filepath = gr.File(label="Audio", file_types=["audio"], scale=1, height=100)
    with gr.Column():
        # Defaulting to 1. It's a still image, but may expand by adding some effects (grain)
        fps = gr.Number(value=1, label="FPS", minimum=1, maximum=144)
    with gr.Column():
        with gr.Group():
            artist_name = gr.Textbox(label="Artist Name", lines=1, max_lines=1, scale=1)
            artist_ffamily, artist_fstyle, artist_fcolor, artist_fopacity, artist_fsize = gru.render_font_picker()
        with gr.Group():
            song_title = gr.Textbox(label="Song Title", lines=1, max_lines=1, scale=2)
            song_ffamily, song_fstyle, song_fcolor, song_fopacity, song_fsize = gru.render_font_picker()

    create_video_button = gr.Button("Create Music Video", variant="primary")

    gr.Markdown("## Output")
    with gr.Group():
        video_output = gr.Video(elem_classes=["video-output"], label="Video Output", interactive=False)
        with gr.Row():
            video_name = gr.Textbox(label="Name", lines=1, max_lines=1, scale=2)
            video_suffix = gr.Dropdown([".mp4", ".mov"], value=".mp4", label="File Type", allow_custom_value=False)
        save_video_button = gr.Button("Save To Disk", variant="primary")

    create_video_button.click(create_music_video, inputs=[cover_image, audio_filepath, fps,
                                                          artist_name, artist_ffamily, artist_fstyle, artist_fsize,
                                                          artist_fcolor, artist_fopacity, song_title, song_ffamily,
                                                          song_fstyle, song_fsize, song_fcolor, song_fopacity],
                              outputs=[video_output])
    save_video_button.click(video_processing.save_video_to_disk,
                            inputs=[video_output, video_name, video_suffix], outputs=[])