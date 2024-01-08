import math
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from moviepy.editor import AudioFileClip, ImageClip
import utils.font_manager as font_manager
import utils.image as image_utils
import numpy as np
import tempfile
from pathlib import Path
import api.chatgpt as chatgpt_api
import processing.image as image_processing


def create_music_video(
        image, audio, fps,
        artist, artist_font_type, artist_font_style, artist_font_size, artist_font_color, artist_font_opacity,
        song, song_font_type, song_font_style, song_font_size, song_font_color, song_font_opacity,
        background_color=(0, 0, 0), background_opacity=66):
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
    cover.thumbnail((math.floor(width * (2 / 3)), math.floor(height * (2 / 3))))
    cover_width, cover_height = cover.size
    canvas.paste(cover, ((width - cover_width) // 2, (height - cover_height) // 2))

    # Add video background
    background = Image.open(image).resize((width, height)).filter(ImageFilter.GaussianBlur(15))
    background_color_overlay = image_utils.get_rgba(background_color, background_opacity)
    overlay = Image.new("RGBA", (width, height), background_color_overlay)
    background.paste(overlay, (0, 0), overlay)
    background = Image.alpha_composite(background.convert('RGBA'), canvas.convert('RGBA'))
    draw = ImageDraw.Draw(background)

    # Add text
    font_families = font_manager.get_fonts()
    artist_font = ImageFont.truetype(font_families[artist_font_type][artist_font_style], artist_font_size)
    artist_font_fill = image_utils.get_rgba(artist_font_color, artist_font_opacity)
    song_font = ImageFont.truetype(font_families[song_font_type][song_font_style], song_font_size)
    song_font_fill = image_utils.get_rgba(song_font_color, song_font_opacity)

    # TODO place one on top of the other like in previous generation
    draw.text((50, height - 100), artist, fill=artist_font_fill, font=artist_font)
    draw.text((50, height - 150), song, fill=song_font_fill, font=song_font)

    # Create Video
    audio_clip = AudioFileClip(audio)
    background_np = np.array(background)
    video_clip = ImageClip(background_np).set_duration(audio_clip.duration).set_audio(audio_clip)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_video_file:
        temp_video_path = temp_video_file.name

    # get suffix of audio clip
    audio_suffix = Path(audio_clip.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=audio_suffix, delete=True) as temp_audio_file:
        temp_audio_path = temp_audio_file.name

    video_clip.write_videofile(temp_video_path, codec="libx264", fps=fps, temp_audiofile=temp_audio_path)

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
