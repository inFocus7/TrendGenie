"""
Module for handling video-related operations in a Gradio interface.
"""
import uuid
from datetime import datetime
from pathlib import Path
import os
from typing import Optional, Literal
import gradio as gr
from moviepy.editor import VideoFileClip
from utils import path_handler

VIDEO_FOLDER = "videos"
default_path = os.path.join(path_handler.get_default_path(), VIDEO_FOLDER)


def render_video_output() -> (gr.Video, gr.Textbox, gr.Dropdown, gr.Button):
    """
    Creates and returns a set of Gradio interface components for video output.

    This function sets up a video display component along with associated controls for naming the video file,
    selecting its file type, and a button for saving the video to disk. It leverages Gradio's UI components to
    create an interactive and user-friendly interface for video handling.

    :returns: A tuple containing the following Gradio UI components: A video display component for showing video output,
        a textbox for inputting the name of the video file, a dropdown menu for selecting the video file type, and a
        button that triggers the action to save the video to disk.
    """
    video_output = gr.Video(elem_classes=["video-output"], label="Video Output", interactive=False)
    with gr.Row():
        video_name = gr.Textbox(label="Name", lines=1, max_lines=1, scale=2)
        video_suffix = gr.Dropdown([".mp4", ".mov"], value=".mp4", label="File Type", allow_custom_value=False)
    save_video_button = gr.Button("Save To Disk", variant="primary")

    return video_output, video_name, video_suffix, save_video_button


def save_video_to_disk(video_path: str, name: Optional[str] = None, video_suffix: Literal[".mp4", ".mov"] = ".mp4",
                       save_dir: str = default_path) -> None:
    """
    Saves a video file to the specified directory with a given name and file suffix.

    This function handles saving a video file to disk. It constructs a file path using the provided directory,
    current date, and a unique name or the specified name. It supports saving in either .mp4 or .mov format.
    If no name is provided, it generates a unique identifier for the file name. The function creates the necessary
    directory structure if it does not exist and then saves the video using moviepy.

    :param video_path: The path to the video file to be saved.
    :param name: The desired name for the saved video file. If not provided, a unique name is generated.
    :param video_suffix: The file extension for the video. Defaults to ".mp4".
    :param save_dir: The directory where the video will be saved. Defaults to the default path defined globally.
    """
    if not video_path or video_path == "":
        gr.Warning("No video to save.")
        return

    base_dir = Path(save_dir) if Path(save_dir).is_absolute() else Path("/").joinpath(save_dir)
    date = datetime.now().strftime("%m%d%Y")
    save_dir = f"{base_dir}/{date}"

    if name is None or name == "":
        unique_id = uuid.uuid4()
        name = f"{unique_id}{video_suffix}"
    else:
        # Remove suffix if it exists
        name = Path(name).stem
        name = f"{name}{video_suffix}"

    video_clip = VideoFileClip(video_path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    video_fqn = os.path.join(save_dir, name)
    video_clip.write_videofile(video_fqn, codec="libx264", fps=video_clip.fps)

    gr.Info(f"Saved video to {video_fqn}.")
