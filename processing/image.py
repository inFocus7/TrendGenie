"""
Module for handling image-related operations in a Gradio interface.
"""
import textwrap
import uuid
from datetime import datetime
import os
from pathlib import Path
from typing import Tuple, Optional, Union, Any, Literal
import PIL
from PIL import ImageFont, ImageDraw, Image, ImageFilter
import numpy as np
import gradio as gr
import cv2
from utils import path_handler
import utils.gradio as gru

IMAGE_FOLDER = "images"
default_path = os.path.join(path_handler.get_default_path(), IMAGE_FOLDER)


def render_image_output() -> (gr.Image, gr.Textbox, gr.Dropdown, gr.Button):
    """
    Creates and returns a set of Gradio interface components for image output.

    This function sets up an image display component along with associated controls for naming the image file,
    selecting its file type, and a button for saving the image to disk. It leverages Gradio's UI components to
    create an interactive and user-friendly interface for image handling.

    Returns:
    - Tuple[gr.Image, gr.Textbox, gr.Dropdown, gr.Button]: A tuple containing Gradio UI components:
        - gr.Image: An image display component for showing image output.
        - gr.Textbox: A textbox for inputting the name of the image file.
        - gr.Dropdown: A dropdown menu for selecting the image file type.
        - gr.Button: A button that triggers the action to save the image to disk.
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


def render_text_editor_parameters(name: str) -> ((gr.Dropdown, gr.Dropdown, gr.Number, gr.ColorPicker, gr.Slider),
                                                 (gr.Checkbox, gr.ColorPicker, gr.Slider, gr.Number),
                                                 (gr.Checkbox, gr.ColorPicker, gr.Slider)):
    """
    Creates and returns a set of Gradio interface components for text editor parameters.

    This function sets up a set of Gradio UI components for configuring the text editor parameters. It includes
    controls for font family, font style, font size, font color, font opacity, drop shadow, drop shadow color,
    drop shadow opacity, drop shadow radius, background, background color, and background opacity.

    :param name: The name of the text editor parameters section.
    :return: A tuple of tuples containing Gradio UI components: A tuple containing Gradio UI
            components for configuring the font family, font style, font size, font color, and font opacity. A tuple
            containing Gradio UI components for configuring the drop shadow, drop shadow color, drop shadow opacity,
            and drop shadow radius. A tuple containing Gradio UI components for configuring the background, background
            color, and background opacity.
    """
    with gr.Accordion(label=name):
        with gr.Column():
            font_family, font_style, font_color, font_opacity, font_size = gru.render_font_picker()
            with gr.Group():
                drop_shadow_checkbox = gr.Checkbox(False, label="Enable Drop Shadow", interactive=True)
                with gr.Group(visible=drop_shadow_checkbox.value) as additional_options:
                    drop_shadow_color, drop_shadow_opacity = gru.render_color_opacity_picker()
                    drop_shadow_radius = gr.Number(0, label="Shadow Radius")
                    gru.bind_checkbox_to_visibility(drop_shadow_checkbox, additional_options)
            with gr.Group():
                background_checkbox = gr.Checkbox(False, label="Enable Background", interactive=True)
                with gr.Group(visible=background_checkbox.value) as additional_options:
                    background_color, background_opacity = gru.render_color_opacity_picker()
                    gru.bind_checkbox_to_visibility(background_checkbox, additional_options)

    return ((font_family, font_style, font_size, font_color, font_opacity),
            (drop_shadow_checkbox, drop_shadow_color, drop_shadow_opacity, drop_shadow_radius),
            (background_checkbox, background_color, background_opacity))


def add_background(image_pil: PIL.Image, draw: PIL.ImageDraw, position: Tuple[int, int], text: str, font: PIL.ImageFont,
                   padding: Tuple[int, int] = (15, 5), fill_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
                   border_radius: int = 0) -> (Tuple[int, int], Tuple[int, int]):
    """
    Adds a background to text on an image.

    :param image_pil: The image to get the size of for text placement.
    :param draw: The image draw object to use for drawing the background.
    :param position: The position of the text on the image.
    :param text: The text to add a background to.
    :param font: The font used for the text.
    :param padding: The padding to add between the text and the background.
    :param fill_color: The RGBA color to fill the background with.
    :param border_radius: The radius of the border.

    :return: A tuple containing the position of the text and the size of the background.
    """
    # Calculate width and height of text with padding
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    left = position[0] - padding[0]
    top = position[1] - padding[1]  # top
    right = left + text_width + 2 * padding[0]
    bottom = top + text_height + 2 * padding[1]

    rect_img = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
    rect_draw = ImageDraw.Draw(rect_img)
    rect_draw.rounded_rectangle([left, top, right, bottom], fill=fill_color, radius=border_radius)
    image_pil.paste(rect_img, (0, 0), rect_img)

    return (left + padding[0], top + padding[1]), (right - left, bottom - top)


def add_blurred_shadow(image_pil: PIL.Image, text: str, position: Tuple[int, int], font: PIL.ImageFont,
                       shadow_color: Tuple[int, int, int, int] = (0, 0, 0), shadow_offset: Tuple[int, int] = (0, 0),
                       blur_radius: int = 1):
    """
    Adds a blurred shadow or highlight to text on an image.
    :param image_pil: The image to place the shadow on.
    :param text: The text to add a shadow to.
    :param position: The position of the text on the image.
    :param font: The font used for the text.
    :param shadow_color: The RGBA color of the shadow.
    :param shadow_offset: The offset of the shadow.
    :param blur_radius: The radius of the blur.
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


def read_image_from_disk(filepath: str, size: Optional[Tuple[int, int]] = None) \
        -> Union[cv2.Mat, np.ndarray[Any, np.dtype[np.generic]], np.ndarray]:
    """
    Reads an image from disk and returns it as a NumPy array for use with PIL.
    :param filepath: The path to the image file.
    :param size: The size to resize the image to.

    :return: A NumPy array containing the image.
    """
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # Convert to RGBA for PIL usage
    if size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def save_images_to_disk(images: gr.data_classes.GradioRootModel, image_type: Literal["png", "jpg", "webp"],
                        save_dir: str = default_path) -> Optional[str]:
    """
    Saves a list of images to disk.
    :param images: The list of images to save from Gradio's Gallery.
    :param image_type: The type of image to save.
    :param save_dir: The directory to save the images to.
    :return: The directory the images were saved to.
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


def save_image_to_disk(image_path: str, name: Optional[str] = None,
                       image_suffix: Literal[".png", ".jpg", ".webp"] = ".png", save_dir: str = default_path) \
        -> Optional[str]:
    """
    Saves an image to disk.
    :param image_path: The path to the image to save. (from a temporary directory from Gradio)
    :param name: The name of the image file. If not provided, a generated name will be used.
    :param image_suffix: The suffix of the image file denoting its type.
    :param save_dir: The directory to save the image to.
    :return: The directory the image was saved to.
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


# Function to add text to an image with custom font, size, and wrapping
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

    img_width, _ = image_pil.size

    if max_width:  # Prepare for text wrapping if max_width is provided
        wrapped_text = textwrap.fill(text, width=max_width)
    else:
        wrapped_text = text

    lines = wrapped_text.split('\n')

    x_pos, y_pos = position
    y_offset = 0
    max_line_width = 0  # Keep track of the widest line
    total_height = 0  # Accumulate total height of text block
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        max_line_width = max(max_line_width, line_width)
        total_height += line_height

        text_x = x_pos  # Adjusted to use numpy width
        if x_center:
            text_x = (img_width - line_width) / 2
        line_y = y_pos + y_offset
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
