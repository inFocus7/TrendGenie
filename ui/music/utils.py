import os
import subprocess
import re
import time
import cv2
from moviepy.editor import AudioFileClip
import utils.font_manager as font_manager
import utils.image as image_utils
import numpy as np
import tempfile
import api.chatgpt as chatgpt_api
import processing.image as image_processing
import librosa
from utils import progress, visualizer
import cProfile


def analyze_audio(audio, target_fps):
    y, sr = librosa.load(audio, sr=None)
    D = librosa.stft(y)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    frequencies = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(D_db.shape[1]), sr=sr)

    audio_clip = AudioFileClip(audio)
    audio_frames_per_video_frame = len(times) / (target_fps * audio_clip.duration)

    sample_indices = np.arange(0, len(times), audio_frames_per_video_frame)
    sample_indices = np.unique(sample_indices.astype(int))
    sample_indices = sample_indices[sample_indices < len(times)]

    downsampled_times = times[sample_indices]
    downsampled_frequency_loudness = [dict(zip(frequencies, D_db[:, i])) for i in sample_indices]

    return downsampled_frequency_loudness, downsampled_times


def create_music_video(
        image, audio, fps,
        artist, artist_font_type, artist_font_style, artist_font_size, artist_font_color, artist_font_opacity,
        artist_shadow_enabled, artist_shadow_color, artist_shadow_opacity, artist_shadow_radius,
        artist_background_enabled, artist_background_color, artist_background_opacity,
        song, song_font_type, song_font_style, song_font_size, song_font_color, song_font_opacity, song_shadow_enabled,
        song_shadow_color, song_shadow_opacity, song_shadow_radius, song_background_enabled, song_background_color,
        song_background_opacity,
        background_color=(0, 0, 0), background_opacity=66, generate_audio_visualizer=False,
        audio_visualizer_color=(255, 255, 255), audio_visualizer_opacity=100, visualizer_drawing=None,
        audio_visualizer_num_rows=90, audio_visualizer_num_columns=65, audio_visualizer_min_size=1,
        audio_visualizer_max_size=7):
    if image is None:
        print("No cover image for the video.")
        return
    if audio is None:
        print("No audio to add to the video.")
        return

    # Could probably expand to 4k, but unnecessary for this type of music video
    # Maybe in a future iteration it could be worth it
    width, height = 1920, 1080

    # Set up cover
    cover = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    if cover.shape[2] == 3:
        cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGBA)
    else:
        cover = cv2.cvtColor(cover, cv2.COLOR_BGRA2RGBA)

    # Create canvas with 4 channels (RGBA)
    canvas = np.zeros((height, width, 4), dtype=np.uint8)

    # Calculate dimensions for resizing the cover to fit within the canvas while maintaining its aspect ratio
    cover_width, cover_height = cover.shape[1], cover.shape[0]
    canvas_width, canvas_height = width, height
    resize_factor = min(canvas_width / cover_width, canvas_height / cover_height)
    resize_factor *= (7 / 10)
    new_width = int(cover_width * resize_factor)
    new_height = int(cover_height * resize_factor)

    # Calculate cover position to center it on the canvas
    cover_pos = ((canvas_width - new_width) // 2, (canvas_height - new_height) // 2)
    cover = cv2.resize(cover, (new_width, new_height))

    canvas[cover_pos[1]:cover_pos[1] + new_height, cover_pos[0]:cover_pos[0] + new_width] = cover

    # Load song / audio
    audio_clip = AudioFileClip(audio)

    # Add video background
    background = cv2.imread(image)
    background = cv2.resize(background, (width, height))
    background = cv2.GaussianBlur(background, (49, 49), 0)
    if background.shape[2] == 3:
        background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
    background_color_overlay = image_utils.get_bgra(background_color, background_opacity)
    overlay = np.full((height, width, 4), background_color_overlay, dtype=np.uint8)
    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_background = background[:, :, 3] / 255.0
    for c in range(0, 3):
        background[:, :, c] = (alpha_overlay * overlay[:, :, c] +
                               alpha_background * (1 - alpha_overlay) * background[:, :, c])
    background[:, :, 3] = (alpha_overlay + alpha_background * (1 - alpha_overlay)) * 255
    background_bgr = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)
    tmp_background_image_path = tempfile.mktemp(suffix=".png")
    cv2.imwrite(tmp_background_image_path, background_bgr)

    audio_visualizer_color_and_opacity = image_utils.get_rgba(audio_visualizer_color, audio_visualizer_opacity)

    # Add audio visualizer
    custom_drawing = None
    if visualizer_drawing is not None and visualizer_drawing != "":
        custom_drawing = cv2.imread(visualizer_drawing, cv2.IMREAD_UNCHANGED)
        if custom_drawing.shape[2] == 3:
            custom_drawing = cv2.cvtColor(custom_drawing, cv2.COLOR_BGR2RGBA)
        else:
            custom_drawing = cv2.cvtColor(custom_drawing, cv2.COLOR_BGRA2RGBA)

    if generate_audio_visualizer:
        print("Generating audio visualizer...")
        frequency_loudness, times = analyze_audio(audio, fps)
        frame_cache = np.zeros((height, width, 4), dtype=np.uint8)

        total_iterations = len(times)
        start_time = time.time()
        vis = visualizer.Visualizer(width=width, height=height, base_size=audio_visualizer_min_size,
                                    max_size=audio_visualizer_max_size, color=audio_visualizer_color_and_opacity,
                                    dot_count=(audio_visualizer_num_rows, audio_visualizer_num_columns))
        vis.initialize_static_values()
        temp_visualizer_images_dir = tempfile.mkdtemp()
        os.makedirs(temp_visualizer_images_dir, exist_ok=True)
        for i, time_point in enumerate(times):
            if time_point > audio_clip.duration:
                break
            frame = frame_cache.copy()
            vis.draw_visualizer(frame, frequency_loudness[i], custom_drawing=custom_drawing)
            frame_np = np.array(frame)
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGRA)
            frame_filename = f'{temp_visualizer_images_dir}/frame_{i:05d}.png'
            cv2.imwrite(frame_filename, frame_np)

            progress.print_progress_bar(i, total_iterations, start_time=start_time)
        progress.print_progress_bar(total_iterations, total_iterations, end='\n', start_time=start_time)

    # Add text
    font_families = font_manager.get_fonts()
    text_canvas = np.zeros((height, width, 4), dtype=np.uint8)

    song_pos = (20, int(height * 0.925))
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

    text_np = np.array(text_canvas)
    np_canvas = np.array(canvas)
    # Normalize the alpha channels
    alpha_text = text_np[:, :, 3] / 255.0
    alpha_canvas = np_canvas[:, :, 3] / 255.0
    alpha_final = alpha_text + alpha_canvas * (1 - alpha_text)

    canvas_final = np.zeros_like(np_canvas)
    # alpha blend
    for c in range(3): # Loop over color (non-alpha) channels
        canvas_final[:, :, c] = (alpha_text * text_np[:, :, c] + alpha_canvas * (1 - alpha_text) *
                                 np_canvas[:, :, c]) / alpha_final
    canvas_final[:, :, 3] = alpha_final * 255
    canvas_final[:, :, :3][alpha_final == 0] = 0

    temp_canvas_image_path = tempfile.mktemp(suffix=".png")
    # Convert to BGR for OpenCV
    canvas_final = cv2.cvtColor(canvas_final, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(temp_canvas_image_path, canvas_final)

    temp_final_video_path = tempfile.mktemp(suffix=".mp4")

    # set up the background video commands
    ffmpeg_commands = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", tmp_background_image_path,
    ]

    if generate_audio_visualizer:
        ffmpeg_commands.extend([
            "-framerate", str(fps),
            "-i", f'{temp_visualizer_images_dir}/frame_%05d.png',
        ])
        filter_complex = "[0][1]overlay=format=auto[bg];[bg][2]overlay=format=auto"
        audio_input_map = "3:a"
    else:
        filter_complex = "[0][1]overlay=format=auto"
        audio_input_map = "2:a"

    ffmpeg_commands.extend([
        "-framerate", str(fps),
        "-i", temp_canvas_image_path,
        "-i", audio,
        "-filter_complex", filter_complex,
        "-map", audio_input_map,
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
    ffmpeg_process = subprocess.Popen(ffmpeg_commands, stderr=subprocess.PIPE, text=True)

    duration_regex = re.compile(r"Duration: (\d\d):(\d\d):(\d\d)\.\d\d")
    time_regex = re.compile(r"time=(\d\d):(\d\d):(\d\d)\.\d\d")
    total_duration_in_seconds = 0

    ffmpeg_start_time = time.time()
    while True:
        line = ffmpeg_process.stderr.readline()
        if not line:
            break

        # Extract total duration of the video
        duration_match = duration_regex.search(line)
        if duration_match:
            hours, minutes, seconds = map(int, duration_match.groups())
            total_duration_in_seconds = hours * 3600 + minutes * 60 + seconds

        # Extract current time of encoding
        time_match = time_regex.search(line)
        if time_match and total_duration_in_seconds > 0:
            hours, minutes, seconds = map(int, time_match.groups())
            current_time = hours * 3600 + minutes * 60 + seconds
            progress.print_progress_bar(current_time, total_duration_in_seconds, start_time=ffmpeg_start_time)

    ffmpeg_process.wait()
    if ffmpeg_process.returncode != 0:
        raise subprocess.CalledProcessError(ffmpeg_process.returncode, ffmpeg_commands)
    progress.print_progress_bar(100, 100, end='\n', start_time=ffmpeg_start_time)
    print("Done generating final video!\n")
    # clean up the original frames
    if generate_audio_visualizer:
        for file in os.listdir(temp_visualizer_images_dir):
            os.remove(os.path.join(temp_visualizer_images_dir, file))
        os.rmdir(temp_visualizer_images_dir)

    return temp_final_video_path


def generate_cover_image(api_key, api_model, prompt):
    client = chatgpt_api.get_openai_client(api_key)
    image_url = chatgpt_api.get_image_response(client, api_model, prompt, portrait=False)
    if image_url is None or image_url == "":
        return None

    return chatgpt_api.url_to_gradio_image_name(image_url)


def process(image_path, artist, song,
            af_family, af_style, afs, afc, afo, ase, asc, aso, asr, abe, abc, abo,
            sf_family, sf_style, sfs, sfc, sfo, sse, ssc, sso, ssr, sbe, sbc, sbo):
    if image_path is None:
        print("No image to modify.")
        return

    font_families = font_manager.get_fonts()
    aff = font_families[af_family][af_style]
    sff = font_families[sf_family][sf_style]

    img = image_processing.read_image_from_disk(image_path)

    # Calculate positions for the text
    top_center = (0, int(img.shape[0] * 0.13))
    bottom_center = (0, int(img.shape[0] * 0.87))

    img, (_, _) = image_processing.add_text(img, artist, top_center, aff,
                                            font_size=afs,
                                            font_color=image_utils.get_rgba(afc, afo),
                                            show_shadow=ase,
                                            shadow_radius=asr,
                                            shadow_color=image_utils.get_rgba(asc, aso),
                                            show_background=abe,
                                            background_color=image_utils.get_rgba(abc, abo),
                                            x_center=True)

    img, (_, _) = image_processing.add_text(img, song, bottom_center, sff, font_size=sfs,
                                            font_color=image_utils.get_rgba(sfc, sfo),
                                            max_width=15,
                                            show_shadow=sse, shadow_radius=ssr,
                                            shadow_color=image_utils.get_rgba(ssc, sso),
                                            show_background=sbe,
                                            background_color=image_utils.get_rgba(sbc, sbo),
                                            x_center=True)

    return img
