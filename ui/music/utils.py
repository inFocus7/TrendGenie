"""
This file contains the functions and utilities used to generate the music video and cover image.
"""
import os
import subprocess
import re
import time
import tempfile
from typing import List, Dict, Optional
from dataclasses import dataclass
import cv2
from moviepy.editor import AudioFileClip
import numpy as np
import librosa
from api import chatgpt as chatgpt_api
from processing import image as image_processing
from utils import progress, visualizer, font_manager, image as image_utils, dataclasses as local_dataclasses


def analyze_audio(audio_path: str, target_fps: int) -> (List[Dict[float, float]], np.ndarray):
    """
    Analyzes the audio file at the given path and returns the frequency loudness and times relating to the frequency
    loudness.
    :param audio_path: The path to the audio file to analyze.
    :param target_fps: The target frames per second for the audio visualizer. This is used to downsample the audio so
      that it aligns with the video.
    :return: A tuple containing the frequency loudness and times relating to the frequency loudness.
    """
    y, sr = librosa.load(audio_path, sr=None)
    d = librosa.stft(y)
    d_db = librosa.amplitude_to_db(np.abs(d), ref=np.max)

    frequencies = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(d_db.shape[1]), sr=sr)

    audio_clip = AudioFileClip(audio_path)
    audio_frames_per_video_frame = len(times) / (target_fps * audio_clip.duration)

    sample_indices = np.arange(0, len(times), audio_frames_per_video_frame)
    sample_indices = np.unique(sample_indices.astype(int))
    sample_indices = sample_indices[sample_indices < len(times)]

    downsampled_times = times[sample_indices]
    downsampled_frequency_loudness = [dict(zip(frequencies, d_db[:, i])) for i in sample_indices]

    return downsampled_frequency_loudness, downsampled_times


@dataclass
class AudioVisualizerDotData:
    """
    A dataclass representing the data for the audio visualizer's dots.
    """
    size: local_dataclasses.MinMax
    count: local_dataclasses.RowCol
    color: local_dataclasses.RGBColor
    opacity: int
    visualizer_drawing: Optional[str] = None
    visualizer_drawing_overlap: bool = False


def _audio_visualizer_generator(frame_size: local_dataclasses.Size, audio_path: str, audio_length: int, fps: int,
                                dot_data: AudioVisualizerDotData) -> str:
    print("Generating audio visualizer...")
    custom_drawing = None
    if dot_data.visualizer_drawing is not None and dot_data.visualizer_drawing != "":
        custom_drawing = cv2.imread(dot_data.visualizer_drawing, cv2.IMREAD_UNCHANGED)
        if custom_drawing.shape[2] == 3:
            custom_drawing = cv2.cvtColor(custom_drawing, cv2.COLOR_BGR2RGBA)
        else:
            custom_drawing = cv2.cvtColor(custom_drawing, cv2.COLOR_BGRA2RGBA)

    frequency_loudness, times = analyze_audio(audio_path, fps)
    frame_cache = np.zeros((frame_size.height, frame_size.width, 4), dtype=np.uint8)

    total_iterations = len(times)
    start_time = time.time()
    vis = visualizer.Visualizer(size=local_dataclasses.Size(frame_size.width, frame_size.height),
                                dot_size=dot_data.size, color=image_utils.get_rgba(dot_data.color, dot_data.opacity),
                                dot_count=local_dataclasses.RowCol(dot_data.count.row, dot_data.count.col))
    vis.initialize_static_values()
    temp_visualizer_images_dir = tempfile.mkdtemp()
    os.makedirs(temp_visualizer_images_dir, exist_ok=True)
    for i, time_point in enumerate(times):
        if time_point > audio_length:
            break
        frame = frame_cache.copy()
        vis.draw_visualizer(frame, frequency_loudness[i], custom_drawing=custom_drawing,
                            custom_drawing_overlap=dot_data.visualizer_drawing_overlap)
        frame_np = np.array(frame)
        frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGRA)
        frame_filename = f'{temp_visualizer_images_dir}/frame_{i:05d}.png'
        cv2.imwrite(frame_filename, frame_np)

        progress.print_progress_bar(i, total_iterations, start_time=start_time)
    progress.print_progress_bar(total_iterations, total_iterations, end='\n', start_time=start_time)

    return temp_visualizer_images_dir


def _get_video_background(image_path: str, frame_size: local_dataclasses.Size,
                          background_overlay_color_opacity: local_dataclasses.RGBOpacity) -> np.ndarray:
    """
    Gets the background for the video, which is a gaussian blurred version of the cover image stretched with a color
    overlay.
    :param image_path: The path to the image to use background.
    :param frame_size: The size of the frame to use for the background.
    :param background_overlay_color_opacity: The color and opacity to use for the background overlay.
    :return:
    """
    background = cv2.imread(image_path)
    background = cv2.resize(background, (frame_size.width, frame_size.height))
    background = cv2.GaussianBlur(background, (49, 49), 0)
    if background.shape[2] == 3:
        background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
    overlay = np.full((frame_size.height, frame_size.width, 4),
                      image_utils.get_bgra(background_overlay_color_opacity.color,
                                           background_overlay_color_opacity.opacity),
                      dtype=np.uint8)
    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_background = background[:, :, 3] / 255.0
    for c in range(0, 3):
        background[:, :, c] = (alpha_overlay * overlay[:, :, c] +
                               alpha_background * (1 - alpha_overlay) * background[:, :, c])
    background[:, :, 3] = (alpha_overlay + alpha_background * (1 - alpha_overlay)) * 255
    return background


def _generate_final_video(background_image_path: str, visualizer_frames_dir: Optional[str], cover_image_path: str,
                          audio_path: str, fps: int) -> str:
    """
    Generates the final video using the given parameters with ffmpeg.
    :param background_image_path: The path to the background image to use for the video.
    :param visualizer_frames_dir: The path to the directory containing the audio visualizer frames.
    :param cover_image_path: The path to the cover image to use for the video.
    :param audio_path: The path to the audio file to use for the video.
    :param fps: The frames per second to use for the video.
    :return:
    """
    temp_final_video_path = tempfile.mktemp(suffix=".mp4")

    audio_clip = AudioFileClip(audio_path)
    ffmpeg_commands = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", background_image_path,
    ]

    if visualizer_frames_dir is not None:
        ffmpeg_commands.extend([
            "-framerate", str(fps),
            "-i", f'{visualizer_frames_dir}/frame_%05d.png',
        ])
        filter_complex = "[0][1]overlay=format=auto[bg];[bg][2]overlay=format=auto"
    else:
        filter_complex = "[0][1]overlay=format=auto"

    ffmpeg_commands.extend([
        "-framerate", str(fps),
        "-i", cover_image_path,
        "-i", audio_path,
        "-filter_complex", filter_complex,
        "-map", "3:a" if visualizer_frames_dir is not None else "2:a",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-strict", "experimental",
        "-t", str(audio_clip.duration),
        "-hide_banner",
        "-framerate", str(fps),
        '-pix_fmt', 'yuv420p',
        temp_final_video_path
    ])
    print("Generating final video...")

    duration_regex = re.compile(r"Duration: (\d\d):(\d\d):(\d\d)\.\d\d")
    time_regex = re.compile(r"time=(\d\d):(\d\d):(\d\d)\.\d\d")

    ffmpeg_start_time = time.time()
    with subprocess.Popen(ffmpeg_commands, stderr=subprocess.PIPE, text=True) as ffmpeg_process:
        for line in ffmpeg_process.stderr:
            # Extract total duration of the video
            duration_match = duration_regex.search(line)
            if duration_match:
                duration_match_groups = duration_match.groups()
                curr_duration = local_dataclasses.Time(
                    hours=int(duration_match_groups[0]),
                    minutes=int(duration_match_groups[1]),
                    seconds=int(duration_match_groups[2])
                )

            # Extract current time of encoding
            time_match = time_regex.search(line)
            if time_match and int(curr_duration) > 0:
                time_match_groups = time_match.groups()
                curr_time = local_dataclasses.Time(
                    hours=int(time_match_groups[0]),
                    minutes=int(time_match_groups[1]),
                    seconds=int(time_match_groups[2])
                )
                progress.print_progress_bar(int(curr_time), int(curr_duration), start_time=ffmpeg_start_time)

    progress.print_progress_bar(100, 100, end='\n', start_time=ffmpeg_start_time)

    return temp_final_video_path


def create_music_video(  # pylint: disable=too-many-locals
        image_path: str, audio_path: str, fps: int,
        artist: str, artist_font_type: str, artist_font_style: str, artist_font_size: int,
        artist_font_color: local_dataclasses.RGBColor, artist_font_opacity: int, artist_shadow_enabled: bool,
        artist_shadow_color: local_dataclasses.RGBColor, artist_shadow_opacity: int, artist_shadow_radius: int,
        artist_background_enabled: bool, artist_background_color: local_dataclasses.RGBColor,
        artist_background_opacity: int, song: str, song_font_type: str, song_font_style: str, song_font_size: int,
        song_font_color: local_dataclasses.RGBColor, song_font_opacity: int, song_shadow_enabled: bool,
        song_shadow_color: local_dataclasses.RGBColor, song_shadow_opacity: int, song_shadow_radius: int,
        song_background_enabled: bool, song_background_color: local_dataclasses.RGBColor, song_background_opacity: int,
        background_color: local_dataclasses.RGBColor = (0, 0, 0), background_opacity: int = 66,
        generate_audio_visualizer: bool = False, audio_visualizer_color: local_dataclasses.RGBColor = (255, 255, 255),
        audio_visualizer_opacity: int = 100, visualizer_drawing: Optional[str] = None,
        visualizer_drawing_overlap: bool = False, audio_visualizer_num_rows: int = 90,
        audio_visualizer_num_columns: int = 65, audio_visualizer_min_size: int = 1,
        audio_visualizer_max_size: int = 7) -> Optional[str]:
    """
    Creates a music video using the given parameters.
    :param visualizer_drawing_overlap: Whether to overlap the visualizer drawings onto one-another with alpha-blending.
      This is only noticeable on images with transparency and is a slow process, so if your visualizer drawings are
      not transparent, it is recommended to set this to False.
    :param image_path: The path to the image to use as the cover + background for the video.
    :param audio_path: The path to the audio file to use for the video.
    :param fps: The frames per second to use for the video.
    :param artist: The artist name to add to the video.
    :param artist_font_type: The font family to use for the artist name.
    :param artist_font_style: The font style to use for the artist name.
    :param artist_font_size: The font size to use for the artist name.
    :param artist_font_color: The font color to use for the artist name.
    :param artist_font_opacity: The font opacity to use for the artist name.
    :param artist_shadow_enabled: Whether to show a shadow for the artist name.
    :param artist_shadow_color: The shadow color to use for the artist name.
    :param artist_shadow_opacity: The shadow opacity to use for the artist name.
    :param artist_shadow_radius: The shadow radius to use for the artist name.
    :param artist_background_enabled: Whether to show a background for the artist name.
    :param artist_background_color: The background color to use for the artist name.
    :param artist_background_opacity: The background opacity to use for the artist name.
    :param song: The song name to add to the video.
    :param song_font_type: The font family to use for the song name.
    :param song_font_style: The font style to use for the song name.
    :param song_font_size: The font size to use for the song name.
    :param song_font_color: The font color to use for the song name.
    :param song_font_opacity: The font opacity to use for the song name.
    :param song_shadow_enabled: Whether to show a shadow for the song name.
    :param song_shadow_color: The shadow color to use for the song name.
    :param song_shadow_opacity: The shadow opacity to use for the song name.
    :param song_shadow_radius: The shadow radius to use for the song name.
    :param song_background_enabled: Whether to show a background for the song name.
    :param song_background_color: The background color to use for the song name.
    :param song_background_opacity: The background opacity to use for the song name.
    :param background_color: The background color to use for the video.
    :param background_opacity: The background opacity to use for the video.
    :param generate_audio_visualizer: Whether to generate an audio visualizer for the video.
    :param audio_visualizer_color: The color to use for the audio visualizer.
    :param audio_visualizer_opacity: The opacity to use for the audio visualizer.
    :param visualizer_drawing: The path to the image to use for the audio visualizer. If None, uses a circle.
    :param audio_visualizer_num_rows: The number of rows to use for the audio visualizer's drawings.
    :param audio_visualizer_num_columns: The number of columns to use for the audio visualizer's drawings.
    :param audio_visualizer_min_size: The minimum size to use for the audio visualizer's drawings (silence).
    :param audio_visualizer_max_size: The maximum size to use for the audio visualizer's drawings (peak loudness).
    :return: The path to the generated video, or None if there was an error.
    """
    if image_path is None or audio_path is None:
        print("No cover image and/or audio for the video.")
        return None

    # Could probably expand to 4k, but unnecessary for this type of music video
    # Maybe in a future iteration it could be worth it
    frame_size = local_dataclasses.Size(1920, 1080)

    # Set up cover
    cover = image_utils.open_image_as_rgba(image_path)

    # Create canvas with 4 channels (RGBA)
    canvas = np.zeros((frame_size.height, frame_size.width, 4), dtype=np.uint8)

    # Calculate dimensions for resizing the cover to fit within the canvas while maintaining its aspect ratio
    cover_size = local_dataclasses.Size(cover.shape[1], cover.shape[0])
    resize_factor = min(frame_size.width / cover_size.width, frame_size.height / cover_size.height)
    resize_factor *= (7 / 10)
    cover_size.width = int(cover_size.width * resize_factor)
    cover_size.height = int(cover_size.height * resize_factor)

    # Calculate cover position to center it on the canvas
    cover_pos = ((frame_size.width - cover_size.width) // 2, (frame_size.height - cover_size.height) // 2)
    cover = cv2.resize(cover, (cover_size.width, cover_size.height))

    canvas[cover_pos[1]:cover_pos[1] + cover_size.height, cover_pos[0]:cover_pos[0] + cover_size.width] = cover

    # Load song / audio
    audio_clip = AudioFileClip(audio_path)

    # Add video background
    background = _get_video_background(image_path, frame_size,
                                       local_dataclasses.RGBOpacity(background_color, background_opacity))
    background_bgr = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)
    tmp_background_image_path = tempfile.mktemp(suffix=".png")
    cv2.imwrite(tmp_background_image_path, background_bgr)

    temp_visualizer_images_dir = None
    if generate_audio_visualizer:
        temp_visualizer_images_dir = _audio_visualizer_generator(frame_size, audio_path, audio_clip.duration, fps,
                                                                 AudioVisualizerDotData(
                                                                     size=local_dataclasses.MinMax(
                                                                         audio_visualizer_min_size,
                                                                         audio_visualizer_max_size),
                                                                     color=audio_visualizer_color,
                                                                     opacity=audio_visualizer_opacity,
                                                                     count=local_dataclasses.RowCol(
                                                                         audio_visualizer_num_rows,
                                                                         audio_visualizer_num_columns),
                                                                     visualizer_drawing=visualizer_drawing,
                                                                     visualizer_drawing_overlap=\
                                                                         visualizer_drawing_overlap)
                                                                 )

    # Add text
    font_families = font_manager.get_fonts()
    text_canvas = np.zeros((frame_size.height, frame_size.width, 4), dtype=np.uint8)

    song_pos = (20, int(frame_size.height * 0.925))
    text_canvas, (_, song_height) = image_processing.add_text(text_canvas, song, song_pos,
                                                              font_families[song_font_type][song_font_style],
                                                              font_size=song_font_size,
                                                              font_color=image_utils.get_rgba(song_font_color,
                                                                                              song_font_opacity),
                                                              show_shadow=song_shadow_enabled,
                                                              shadow_radius=song_shadow_radius,
                                                              shadow_color=image_utils.get_rgba(song_shadow_color,
                                                                                                song_shadow_opacity),
                                                              show_background=song_background_enabled,
                                                              background_color=image_utils.get_rgba(
                                                                  song_background_color,
                                                                  song_background_opacity))
    artist_pos = (song_pos[0], song_pos[1] - song_height - 5)
    text_canvas, (_, _) = image_processing.add_text(text_canvas, artist, artist_pos,
                                                    font_families[artist_font_type][artist_font_style],
                                                    font_size=artist_font_size,
                                                    font_color=image_utils.get_rgba(artist_font_color,
                                                                                    artist_font_opacity),
                                                    show_shadow=artist_shadow_enabled,
                                                    shadow_radius=artist_shadow_radius,
                                                    shadow_color=image_utils.get_rgba(artist_shadow_color,
                                                                                      artist_shadow_opacity),
                                                    show_background=artist_background_enabled,
                                                    background_color=image_utils.get_rgba(
                                                        artist_background_color, artist_background_opacity))

    canvas_final = image_utils.blend_alphas(np.array(text_canvas), np.array(canvas))
    temp_canvas_image_path = tempfile.mktemp(suffix=".png")
    # Convert to BGR for OpenCV
    canvas_final = cv2.cvtColor(canvas_final, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(temp_canvas_image_path, canvas_final)

    temp_final_video_path = _generate_final_video(tmp_background_image_path, temp_visualizer_images_dir,
                                                  temp_canvas_image_path, audio_path, fps)

    # clean up the original frames
    if temp_visualizer_images_dir is not None:
        for file in os.listdir(temp_visualizer_images_dir):
            os.remove(os.path.join(temp_visualizer_images_dir, file))
        os.rmdir(temp_visualizer_images_dir)

    return temp_final_video_path


def generate_cover_image(api_key: str, api_model: str, prompt: str) -> Optional[str]:
    """
    Generates a cover image using the OpenAI API based on a given prompt and specified parameters.
    :param api_key: The API key to use for the OpenAI API.
    :param api_model: The model to use for image generation (e.g., 'dall-e-3').
    :param prompt: The text prompt based on which the image is generated.
    :return: The URL of the generated image, or None if no image was generated or if there was an error.
    """
    client = chatgpt_api.get_openai_client(api_key)
    image_url = chatgpt_api.get_image_response(client, api_model, prompt, portrait=False)
    if image_url is None or image_url == "":
        return None

    return chatgpt_api.url_to_gradio_image_name(image_url)


# pylint: disable=too-many-locals
def process(image_path: str, artist: str, song: str,
            af_family: str, af_style: str, afs: int, afc: local_dataclasses.RGBColor, afo: int, ase: bool,
            asc: local_dataclasses.RGBColor, aso: int, asr: Optional[int], abe: bool, abc: local_dataclasses.RGBColor,
            abo: int, sf_family: str, sf_style: str, sfs: int, sfc: local_dataclasses.RGBColor, sfo: int, sse: bool,
            ssc: local_dataclasses.RGBColor, sso: int, ssr: Optional[int], sbe: bool, sbc: local_dataclasses.RGBColor,
            sbo: int) \
        -> Optional[np.ndarray]:
    """
    Processes the image at the given path (by adding the requested text) and returns the processed image.
    :param image_path: The path to the image to process.
    :param artist: The artist name to add to the image.
    :param song: The song name to add to the image.
    :param af_family: The font family to use for the artist name.
    :param af_style: The font style to use for the artist name.
    :param afs: The font size to use for the artist name.
    :param afc: The font color to use for the artist name.
    :param afo: The font opacity to use for the artist name.
    :param ase: Whether to show a shadow for the artist name.
    :param asc: The shadow color to use for the artist name.
    :param aso: The shadow opacity to use for the artist name.
    :param asr: The shadow radius to use for the artist name.
    :param abe: Whether to show a background for the artist name.
    :param abc: The background color to use for the artist name.
    :param abo: The background opacity to use for the artist name.
    :param sf_family: The font family to use for the song name.
    :param sf_style: The font style to use for the song name.
    :param sfs: The font size to use for the song name.
    :param sfc: The font color to use for the song name.
    :param sfo: The font opacity to use for the song name.
    :param sse: Whether to show a shadow for the song name.
    :param ssc: The shadow color to use for the song name.
    :param sso: The shadow opacity to use for the song name.
    :param ssr: The shadow radius to use for the song name.
    :param sbe: Whether to show a background for the song name.
    :param sbc: The background color to use for the song name.
    :param sbo: The background opacity to use for the song name.
    :return: The processed image as a numpy array. If there was no image to process, returns None.
    """
    if image_path is None:
        print("No image to modify.")
        return None

    font_families = font_manager.get_fonts()

    img = image_processing.read_image_from_disk(image_path)

    # Calculate positions for the text
    top_center = (0, int(img.shape[0] * 0.13))
    bottom_center = (0, int(img.shape[0] * 0.87))

    img, (_, _) = image_processing.add_text(img, artist, top_center, font_families[af_family][af_style],
                                            font_size=afs,
                                            font_color=image_utils.get_rgba(afc, afo),
                                            show_shadow=ase,
                                            shadow_radius=asr,
                                            shadow_color=image_utils.get_rgba(asc, aso),
                                            show_background=abe,
                                            background_color=image_utils.get_rgba(abc, abo),
                                            x_center=True)

    img, (_, _) = image_processing.add_text(img, song, bottom_center, font_families[sf_family][sf_style], font_size=sfs,
                                            font_color=image_utils.get_rgba(sfc, sfo),
                                            max_width=15,
                                            show_shadow=sse, shadow_radius=ssr,
                                            shadow_color=image_utils.get_rgba(ssc, sso),
                                            show_background=sbe,
                                            background_color=image_utils.get_rgba(sbc, sbo),
                                            x_center=True)

    return img
