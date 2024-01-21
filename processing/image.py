"""
This module contains functions for processing images.
"""
import textwrap
import uuid
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal, Union, Tuple
from PIL import ImageFont, ImageDraw, Image, ImageFilter
import numpy as np
import gradio as gr
import cv2
from utils import gradio as gru, path_handler, dataclasses

IMAGE_FOLDER = "images"

default_path = os.path.join(path_handler.get_default_path(), IMAGE_FOLDER)


def render_image_output() -> (gr.Image, gr.Textbox, gr.Dropdown, gr.Button):
    """
    Renders the image output components.
    :return: A tuple containing the image output, image name, image suffix, and save image button components.
    """
    image_output = gr.Image(elem_classes=["single-image-output"],
                            label="Image Output", interactive=False,
                            show_download_button=False, type="filepath")
    with gr.Row():
        image_name = gr.Textbox(label="Name", lines=1, max_lines=1, scale=2, interactive=True)
        image_suffix = gr.Dropdown([".png", ".jpg", ".webp"], value=".png", label="File Type",
                                   allow_custom_value=False, interactive=True)
    save_image_button = gr.Button("Save To Disk", variant="primary", interactive=True)

    return image_output, image_name, image_suffix, save_image_button


def render_text_editor_parameters(name: str) -> dataclasses.FontDisplayGradioComponents:
    """
    Renders the text editor parameters.
    :param name: The name of the text editor parameters. This is used as the label for the accordion.
    :return: Classes containing the font, drop shadow, and background components.
    """
    with gr.Accordion(label=name):
        with gr.Column():
            font_data = gru.render_font_picker()
            with gr.Group():
                drop_shadow_enabled = gr.Checkbox(False, label="Enable Drop Shadow", interactive=True)
                with gr.Group(visible=drop_shadow_enabled.value) as additional_options:
                    drop_shadow_color_opacity = gru.render_color_opacity_picker()
                    drop_shadow_radius = gr.Number(0, label="Shadow Radius")
                    gru.bind_checkbox_to_visibility(drop_shadow_enabled, additional_options)
            with gr.Group():
                background_enabled = gr.Checkbox(False, label="Enable Background", interactive=True)
                with gr.Group(visible=background_enabled.value) as additional_options:
                    background_color_opacity = gru.render_color_opacity_picker()
                    gru.bind_checkbox_to_visibility(background_enabled, additional_options)

    drop_shadow_data = dataclasses.FontDropShadowGradioComponents(drop_shadow_enabled, drop_shadow_color_opacity.color,
                                                                  drop_shadow_color_opacity.opacity, drop_shadow_radius)
    background_data = dataclasses.FontBackgroundGradioComponents(background_enabled, background_color_opacity.color,
                                                                 background_color_opacity.opacity)

    return dataclasses.FontDisplayGradioComponents(font_data, drop_shadow_data, background_data)


def add_background(image_pil: Image, draw: ImageDraw, position: tuple[int, int], text: str, font: ImageFont,
                   padding: tuple[int, int] = (15, 5), fill_color: tuple[int, int, int, int] = (0, 0, 0, 255),
                   border_radius: int = 0) -> (tuple[int, int], tuple[int, int]):
    """
    Adds a background to the text.
    :param image_pil: The PIL image to add the background to.
    :param draw: The PIL draw object to use.
    :param position: The position of the text on the image.
    :param text: The text to add the background to.
    :param font: The font to use.
    :param padding: The padding between the font and background.
    :param fill_color: The color of the background.
    :param border_radius: The border radius of the background.
    :return: A tuple containing the position of the text and the size of the background.
    """
    # Calculate width and height of text with padding
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    rect_pos = dataclasses.FourEdges(left=position[0] - padding[0],
                                     top=position[1] - padding[1],
                                     right=(position[0] - padding[0]) + text_width + 2 * padding[0],
                                     bottom=(position[1] - padding[1]) + text_height + 2 * padding[1])

    rect_img = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
    rect_draw = ImageDraw.Draw(rect_img)
    rect_draw.rounded_rectangle([rect_pos.left, rect_pos.top, rect_pos.right, rect_pos.bottom], fill=fill_color,
                                radius=border_radius)
    image_pil.paste(rect_img, (0, 0), rect_img)

    return ((rect_pos.left + padding[0], rect_pos.top + padding[1]),
            (rect_pos.right - rect_pos.left, rect_pos.bottom - rect_pos.top))


def add_blurred_shadow(image_pil: Image, text: str, position: tuple[float, float], font: ImageFont,
                       shadow_color: tuple[int, int, int, int] = (0, 0, 0, 0), shadow_offset: tuple[int, int] = (0, 0),
                       blur_radius: int = 1) -> None:
    """
    Adds a blurred shadow (or highlight) to the text.
    :param image_pil: The PIL image to add the shadow to.
    :param text: The text to add the shadow to.
    :param position: The position of the text on the image.
    :param font: The font to use.
    :param shadow_color: The color of the shadow.
    :param shadow_offset: The offset of the shadow.
    :param blur_radius: The blur radius of the shadow.
    """
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


def read_image_from_disk(filepath: str, size: Optional[cv2.typing.Size] = None) -> np.ndarray:
    """
    Reads and returns an image from disk using CV2.
    :param filepath: The path to the image.
    :param size: The size to resize the image to.
    :return: The image as a NumPy array.
    """
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # Convert to RGBA for PIL usage
    if size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def save_images_to_disk(images: gr.data_classes.RootModel, image_type: gr.Dropdown, save_dir: str = default_path) -> \
        Optional[str]:
    """
    Saves a list of images to disk.
    :param images: The list of images to save. Imported from a gradio.Gallery component.
    :param image_type: The type of image to save.
    :param save_dir: The directory to save the images to.
    :return: The directory the images were saved to. None if there was an error.
    """
    if not images or len(images.root) == 0:
        gr.Warning("No images to save.")
        return None

    base_dir = Path(save_dir) if Path(save_dir).is_absolute() else Path("/").joinpath(save_dir)

    date = datetime.now().strftime("%m%d%Y")
    unique_id = uuid.uuid4()
    save_dir = f"{base_dir}/{date}/{unique_id}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for index, image_container in enumerate(images.root):
        image = image_container.image
        if image.path is None:
            gr.Warning(f"Image at index {index} has no path... this shouldn't happen.")
            continue

        filename = f"{index}.{image_type}"
        filepath = os.path.join(save_dir, filename)

        img = cv2.imread(image.path, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(filepath, img)

    gr.Info(f"Saved generated images to {save_dir}.")
    return save_dir


def save_image_to_disk(image_path: str, name: Optional[str] = None, image_suffix: Literal[".png", ".jpg", ".webp"] = \
                       ".png", save_dir: str = default_path) -> Optional[str]:
    """
    Saves an image to disk.
    :param image_path: The path to the temporary image.
    :param name: The name to give the saved image.
    :param save_dir: The directory to save the image to.
    :param image_suffix: The suffix to give the saved image.
    :return: The directory the image was saved to. None if there was an error.
    """
    if image_path is None:
        gr.Warning("No image to save.")
        return None

    base_dir = Path(save_dir) if Path(save_dir).is_absolute() else Path("/").joinpath(save_dir)

    date = datetime.now().strftime("%m%d%Y")
    unique_id = uuid.uuid4()
    save_dir = f"{base_dir}/{date}/{unique_id}"

    if name is None or name == "":
        unique_id = uuid.uuid4()
        name = f"{unique_id}{image_suffix}"
    else:
        # Remove suffix if it exists
        name = Path(name).stem
        name = f"{name}{image_suffix}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filepath = os.path.join(save_dir, name)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(filepath, img)

    gr.Info(f"Saved generated image to {save_dir}.")
    return save_dir


def _get_lines(text: str, max_width: Optional[int] = None) -> list[str]:
    """
    Gets the lines of text from a string.
    :param text: The text to get the lines from.
    :param max_width: The maximum width of the text before wrapping.
    :return: A list of lines.
    """
    if max_width:  # Prepare for text wrapping if max_width is provided
        wrapped_text = textwrap.fill(text, width=max_width)
    else:
        wrapped_text = text

    return wrapped_text.split('\n')


# A lot of the reported variables come from the parameters
# pylint: disable=too-many-locals
def add_text(image: Union[Image.Image, np.ndarray], text: str, position: Tuple[int, int], font_path: str,
             font_size: int, font_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
             shadow_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
             shadow_radius: Optional[int] = None, max_width: Optional[int] = None, show_background: bool = False,
             show_shadow: bool = False, background_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
             x_center: bool = False) -> (np.ndarray, Tuple[int, int]):
    """
    Adds text to an image with custom font, size, and wrapping.
    :param image: The image to add text to.
    :param text: The text to add to the image.
    :param position: The (x, y) position of the text on the image.
    :param font_path: The path to the font to use.
    :param font_size: The size of the font.
    :param font_color: The color of the font.
    :param shadow_color: The color of the shadow.
    :param shadow_radius: The radius of the shadow.
    :param max_width: The maximum width of the text before wrapping.
    :param show_background: Whether to show a background behind the text.
    :param show_shadow: Whether to show a shadow behind the text.
    :param background_color: The color of the background.
    :param x_center: Whether to center the text on the x-axis. This ignores the positional x parameter.
    :return: A tuple containing the image with text added and the size of the text block.
    """
    if not isinstance(position, tuple):
        raise TypeError("Position must be a 2-tuple.", type(position))

    # Check if the image is a NumPy array (OpenCV image)
    if isinstance(image, np.ndarray):
        # Convert OpenCV image (BGR) to PIL image (RGB)
        image_pil = Image.fromarray(image).convert("RGBA")
    elif isinstance(image, Image.Image):
        image_pil = image.convert("RGBA")
    else:
        raise TypeError("Unsupported image type.", type(image))

    txt_layer = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(txt_layer)

    lines = _get_lines(text, max_width)

    y_offset = 0
    # max_line_width = 0  # Keep track of the widest line
    # total_height = 0  # Accumulate total height of text block
    text_container = dataclasses.Size(width=0, height=0)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_size = dataclasses.Size(width=bbox[2] - bbox[0], height=bbox[3] - bbox[1])
        text_container.width = max(text_container.width, line_size.width)
        text_container.height += line_size.height

        pos = dataclasses.Position
        pos.x = position[0]
        if x_center:
            pos.x = (image_pil.width - line_size.width) / 2
        pos.y = position[1] + y_offset
        y_offset += (line_size.height + 6)

        if show_background:
            (pos.x, pos.y), _ = add_background(image_pil, draw, (pos.x, pos.y), line, font,
                                               fill_color=background_color, border_radius=10)

        if show_shadow:
            shadow_position = (pos.x, pos.y)
            add_blurred_shadow(image_pil, line, shadow_position, font, shadow_color=shadow_color,
                               blur_radius=shadow_radius)

        draw.text((pos.x, pos.y), line, font=font, fill=font_color)

    image_pil = Image.alpha_composite(image_pil, txt_layer)
    return np.array(image_pil), (text_container.width, text_container.height)
