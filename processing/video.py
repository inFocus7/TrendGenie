import uuid
from datetime import datetime
import gradio as gr
from pathlib import Path
import os
from moviepy.editor import VideoFileClip
import utils.path_handler as path_handler

video_folder = "videos"
default_path = os.path.join(path_handler.get_default_path(), video_folder)


def render_video_output():
    video_output = gr.Video(elem_classes=["video-output"], label="Video Output", interactive=False)
    with gr.Row():
        video_name = gr.Textbox(label="Name", lines=1, max_lines=1, scale=2)
        video_suffix = gr.Dropdown([".mp4", ".mov"], value=".mp4", label="File Type", allow_custom_value=False)
    save_video_button = gr.Button("Save To Disk", variant="primary")

    return video_output, video_name, video_suffix, save_video_button


def save_video_to_disk(video, name, video_suffix=".mp4", dir=default_path):
    if not video:
        gr.Warning("No video to save.")
        return

    base_dir = Path(dir) if Path(dir).is_absolute() else Path("/").joinpath(dir)
    date = datetime.now().strftime("%m%d%Y")
    dir = f"{base_dir}/{date}"

    if name is None or name == "":
        unique_id = uuid.uuid4()
        name = f"{unique_id}{video_suffix}"
    else:
        # Remove suffix if it exists
        name = Path(name).stem
        name = f"{name}{video_suffix}"

    video_clip = VideoFileClip(video)

    if not os.path.exists(dir):
        os.makedirs(dir)

    video_fqn = os.path.join(dir, name)
    video_clip.write_videofile(video_fqn, codec="libx264", fps=video_clip.fps)

    gr.Info(f"Saved video to {video_fqn}.")
