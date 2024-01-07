from PIL import ImageFont, ImageDraw, Image, ImageFilter
import numpy as np
import textwrap
import gradio as gr
import uuid
from datetime import datetime
import os
import cv2
from pathlib import Path


def add_background(image_pil, draw, position, text, font, padding=(15, 5), fill_color=(0, 0, 0, 255), border_radius=0):
    # Calculate width and height of text with padding
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x1 = position[0] - padding[0]  # left
    y1 = position[1] - padding[1]  # top
    x2 = x1 + text_width + 2 * padding[0]  # right
    y2 = y1 + text_height + 2 * padding[1]  # bottom

    rect_img = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
    rect_draw = ImageDraw.Draw(rect_img)
    rect_draw.rounded_rectangle([x1, y1, x2, y2], fill=fill_color, radius=border_radius)
    image_pil.paste(rect_img, (0, 0), rect_img)

    return (x1 + padding[0], y1 + padding[1]), (x2 - x1, y2 - y1)


def add_blurred_shadow(image_pil, text, position, font, shadow_color=(0, 0, 0), shadow_offset=(0, 0),
                       blur_radius=1):
    # Create an image for the shadow
    shadow_image = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_image)

    # Draw the shadow text
    shadow_draw.text((position[0] + shadow_offset[0], position[1] + shadow_offset[1]), text, font=font,
                     fill=shadow_color)

    # Apply a Gaussian blur to the shadow image
    blurred_shadow = shadow_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Composite the blurred shadow with the original image
    image_pil.paste(blurred_shadow, (0, 0), blurred_shadow)


def read_image_from_disk(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # Convert to RGBA for PIL usage
    return cv2.resize(img, (1080, 1920), interpolation=cv2.INTER_AREA)


def save_images_to_disk(images, image_type, dir="trendgenie/images"):
    if not images or len(images.root) == 0:
        gr.Warning("No images to save.")
        return

    base_dir = Path(dir) if Path(dir).is_absolute() else Path("/").joinpath(dir)

    date = datetime.now().strftime("%m%d%Y")
    unique_id = uuid.uuid4()
    dir = f"{base_dir}/{date}/{unique_id}"

    if not os.path.exists(dir):
        os.makedirs(dir)

    for index, image_container in enumerate(images.root):
        image = image_container.image
        if image.path is None:
            gr.Warning(f"Image at index {index} has no path... this shouldn't happen.")
            continue

        filename = f"{index}.{image_type}"
        filepath = os.path.join(dir, filename)

        img = cv2.imread(image.path, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(filepath, img)

    gr.Info(f"Saved generated images to {dir}.")
    return dir


# Function to add text to an image with custom font, size, and wrapping
def add_text(image, text, position, font_path, font_size, font_color=(255, 255, 255, 255), shadow_color=(255, 255, 255),
             shadow_radius=None, max_width=None, show_background=False, show_shadow=False,
             background_color=(0, 0, 0, 255)):
    # Convert OpenCV image to PIL image
    image_pil = Image.fromarray(image).convert("RGBA")

    txt_layer = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(txt_layer)

    img_width, img_height = image.shape[1], image.shape[0]

    if max_width:  # Prepare for text wrapping if max_width is provided
        wrapped_text = textwrap.fill(text, width=max_width)
    else:
        wrapped_text = text

    lines = wrapped_text.split('\n')

    y_offset = 0
    max_line_width = 0  # Keep track of the widest line
    total_height = 0  # Accumulate total height of text block
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        max_line_width = max(max_line_width, line_width)
        total_height += line_height

        text_x = (img_width - line_width) / 2  # Adjusted to use numpy width
        line_y = position + y_offset
        y_offset += (line_height + 6)

        if show_background:
            (text_x, line_y), _ = add_background(image_pil, draw, (text_x, line_y), line, font,
                                                 fill_color=background_color, border_radius=10)

        if show_shadow:
            shadow_position = (text_x, line_y)
            add_blurred_shadow(image_pil, line, shadow_position, font, shadow_color=shadow_color,
                               blur_radius=shadow_radius)

        draw.text((text_x, line_y), line, font=font, fill=font_color)

    image_pil = Image.alpha_composite(image_pil, txt_layer)
    return np.array(image_pil), (max_line_width, total_height)
