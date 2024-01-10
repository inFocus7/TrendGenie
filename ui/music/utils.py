import math
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from moviepy.editor import AudioFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips
import utils.font_manager as font_manager
import utils.image as image_utils
import numpy as np
import tempfile
import api.chatgpt as chatgpt_api
import processing.image as image_processing
import librosa
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


cached_visualizer_dot_positions = None
cached_visualizer_background = None


# TODO - look into why the ellipsis resizing is not ranging over ALL of x axis, but seemingly repeated.
# TODO - allow for custom 'dot' shape/image.
def draw_visualizer(canvas, frequency_data, base_size=1, max_size=7, color=(255, 255, 255, 255), dot_count=(90, 65), # the more dots, the more drawings, meaning slower.
                    alias_scale=1):
    global cached_visualizer_dot_positions, cached_visualizer_background
    width, height = canvas.size[0] * alias_scale, canvas.size[1] * alias_scale

    if cached_visualizer_background is None:
        cached_visualizer_background = Image.new("RGBA", (width, height))
    large_canvas = cached_visualizer_background.copy()
    large_draw = ImageDraw.Draw(large_canvas)

    if cached_visualizer_dot_positions is None:
        # Calculate and store dot positions
        x_positions = (width / dot_count[0]) * np.arange(dot_count[0]) + (width / dot_count[0] / 2)
        y_positions = (height / dot_count[1]) * np.arange(dot_count[1]) + (height / dot_count[1] / 2)
        grid_x, grid_y = np.meshgrid(x_positions, y_positions)
        cached_visualizer_dot_positions = [(grid_x[y, x], grid_y[y, x]) for x in range(dot_count[0]) for y in
                                           range(dot_count[1])]

    # Precompute log frequencies
    freq_keys = np.array(list(frequency_data.keys()))
    start_freq = freq_keys[freq_keys > 0][0] if freq_keys[freq_keys > 0].size > 0 else 1.0
    end_freq = freq_keys[-1]
    log_freqs = np.logspace(np.log10(start_freq), np.log10(end_freq), dot_count[0])

    # Find the maximum and minimum loudness values, ignoring -80 dB
    freq_bands = np.array([frequency_data[key] for key in freq_keys if key > 0])  # Ignore 0 Hz
    max_loudness = np.max(freq_bands)
    filtered_loudness = freq_bands[freq_bands > -80]
    min_loudness = np.min(filtered_loudness) if filtered_loudness.size > 0 else -80

    # Precompute loudness values
    loudness_values = {}
    for x in range(dot_count[0]):
        lower_bound = log_freqs[x]
        upper_bound = log_freqs[x + 1] if x < dot_count[0] - 1 else end_freq + 1
        band_freqs = [freq for freq in freq_keys if lower_bound <= freq < upper_bound]
        if not band_freqs:
            closest_freq = min(freq_keys, key=lambda f: abs(f - lower_bound))
            band_freqs = [closest_freq]

        band_loudness = [frequency_data[freq] for freq in band_freqs]
        avg_loudness = np.mean(band_loudness) if band_loudness else -80
        loudness_values[x] = avg_loudness

    cached_dot_sizes = {}
    for i, (pos_x, pos_y) in enumerate(cached_visualizer_dot_positions):
        column = i // dot_count[1]  # Ensure the correct column is computed

        if column not in cached_dot_sizes:
            avg_loudness = loudness_values[column]

            # Scale the loudness to the dot size
            scaled_loudness = (avg_loudness - min_loudness) / (max_loudness - min_loudness) if max_loudness != min_loudness else 0
            dot_size = base_size + scaled_loudness * (max_size - base_size)
            dot_size = min(max(dot_size, base_size), max_size) * alias_scale

            cached_dot_sizes[column] = dot_size
        else:
            dot_size = cached_dot_sizes[column]

        large_draw.ellipse([(pos_x - dot_size / 2, pos_y - dot_size / 2), (pos_x + dot_size / 2, pos_y + dot_size / 2)], fill=color, outline=color)

    canvas.paste(large_canvas.resize(canvas.size, Image.LANCZOS))


def create_music_video(
        image, audio, fps,
        artist, artist_font_type, artist_font_style, artist_font_size, artist_font_color, artist_font_opacity,
        song, song_font_type, song_font_style, song_font_size, song_font_color, song_font_opacity,
        background_color=(0, 0, 0), background_opacity=66, generate_audio_visualizer=False,
        audio_visualizer_color=(255, 255, 255), audio_visualizer_opacity=100):
    if image is None:
        print("No cover image for the video.")
        return
    if audio is None:
        print("No audio to add to the video.")
        return

    # Could probably expand to 4k, but unnecessary for this type of music video
    # Maybe in a future iteration it could be worth it
    width, height = 1920, 1080
    canvas = Image.new("RGBA", (width, height))

    # Set up cover
    cover = Image.open(image)
    if cover.mode != 'RGBA':
        cover = cover.convert('RGBA')
    cover.thumbnail((math.floor(width * (2 / 3)), math.floor(height * (2 / 3))))
    cover_width, cover_height = cover.size
    cover_pos = ((width - cover_width) // 2, (height - cover_height) // 2)
    canvas.paste(cover, cover_pos, cover)

    # Load Audio Clip
    audio_clip = AudioFileClip(audio)

    # Add video background
    background = Image.open(image).resize((width, height)).filter(ImageFilter.GaussianBlur(15))
    background_color_overlay = image_utils.get_rgba(background_color, background_opacity)
    overlay = Image.new("RGBA", (width, height), background_color_overlay)
    background.paste(overlay, (0, 0), overlay)
    background_np = np.array(background)
    background_clip = ImageClip(background_np).set_duration(audio_clip.duration)

    audio_visualizer_color_opacity = image_utils.get_rgba(audio_visualizer_color, audio_visualizer_opacity)

    # Add audio visualizer
    visualizer_clip = []
    if generate_audio_visualizer:
        frequency_loudness, times = analyze_audio(audio, fps)
        audio_frame_duration = 1.0 / fps
        frame_cache = Image.new("RGBA", (width, height))
        for i, time_point in enumerate(times):
            if time_point > audio_clip.duration:
                break
            frame = frame_cache.copy()
            cProfile.runctx("draw_visualizer(frame, frequency_loudness[i], color=audio_visualizer_color_opacity)",
                            locals=locals(), globals=globals())
            draw_visualizer(frame, frequency_loudness[i], color=audio_visualizer_color_opacity)
            frame_np = np.array(frame)
            frame_clip = ImageClip(frame_np).set_duration(audio_frame_duration)
            # If I want to add some blending effect, i'll have to do some mask/function here and blend this frame with
            # the equivalent background frame.
            visualizer_clip.append(frame_clip)

    current_clip = background_clip
    if len(visualizer_clip) > 0:
        visualizer_clip = concatenate_videoclips(visualizer_clip, method="compose")
        visualizer_clip.set_opacity(0.01)
        current_clip = CompositeVideoClip([background_clip, visualizer_clip])

    # Place the cover on top of the background
    np_canvas = np.array(canvas)
    canvas_clip = ImageClip(np_canvas).set_duration(audio_clip.duration)
    current_clip = CompositeVideoClip([current_clip, canvas_clip])

    # Add text
    font_families = font_manager.get_fonts()
    text_canvas = Image.new("RGBA", (width, height))
    text_draw = ImageDraw.Draw(text_canvas)

    artist_font = ImageFont.truetype(font_families[artist_font_type][artist_font_style], artist_font_size)
    artist_font_fill = image_utils.get_rgba(artist_font_color, artist_font_opacity)
    song_font = ImageFont.truetype(font_families[song_font_type][song_font_style], song_font_size)
    song_font_fill = image_utils.get_rgba(song_font_color, song_font_opacity)

    # TODO place one on top of the other like in previous generation
    text_draw.text((50, height - 150), artist, fill=artist_font_fill, font=artist_font)
    text_draw.text((50, height - 100), song, fill=song_font_fill, font=song_font)
    text_np = np.array(text_canvas)
    text_clip = ImageClip(text_np).set_duration(audio_clip.duration)
    current_clip = CompositeVideoClip([current_clip, text_clip])

    current_clip = current_clip.set_audio(audio_clip)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_video_file:
        temp_video_path = temp_video_file.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_path = temp_audio_file.name

    current_clip.write_videofile(temp_video_path, codec="libx264", fps=fps, temp_audiofile=temp_audio_path)

    return temp_video_path


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
    top_center = int(img.shape[0] * 0.13)
    bottom_center = int(img.shape[0] * 0.70)

    img, (_, _) = image_processing.add_text(img, artist, top_center, aff,
                                            font_size=afs,
                                            font_color=image_utils.get_rgba(afc, afo),
                                            show_shadow=ase,
                                            shadow_radius=asr,
                                            shadow_color=image_utils.get_rgba(asc, aso),
                                            show_background=abe,
                                            background_color=image_utils.get_rgba(abc, abo))

    img, (_, _) = image_processing.add_text(img, song, bottom_center, sff, font_size=sfs,
                                            font_color=image_utils.get_rgba(sfc, sfo),
                                            max_width=15,
                                            show_shadow=sse, shadow_radius=ssr,
                                            shadow_color=image_utils.get_rgba(ssc, sso),
                                            show_background=sbe,
                                            background_color=image_utils.get_rgba(sbc, sbo))

    return img
