import cv2
import json
from PIL import ImageFont, ImageDraw, Image, ImageFilter
import numpy as np
import textwrap

font_path = './fonts/EBGaramondSC08-Regular.otf'

# Load JSON
with open('list.json') as file:
    data = json.load(file)


def add_background(image_pil, draw, position, text, font, padding=(15, 5), fill_color=(0, 0, 0, 255), border_radius=0):
    # Calculate width and height of text with padding
    text_width, text_height = draw.textsize(text, font=font)
    x1 = position[0] - padding[0]  # left
    y1 = position[1] - padding[1]  # top
    x2 = x1 + text_width + 2 * padding[0]  # right
    y2 = y1 + text_height + 2 * padding[1]  # bottom

    rect_img = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
    rect_draw = ImageDraw.Draw(rect_img)
    rect_draw.rounded_rectangle([x1, y1, x2, y2], fill=fill_color, radius=border_radius)
    image_pil.paste(rect_img, (0,0), rect_img)

    return (x1 + padding[0], y1 + padding[1]), (x2 - x1, y2 - y1)


def add_blurred_shadow(draw, image_pil, text, position, font, shadow_color=(0, 0, 0), shadow_offset=(0, 0),
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


# Function to add text to an image with custom font, size, and wrapping
def add_text(image, text, position, font_path, font_size, font_color=(255, 255, 255), shadow_color=(255, 255, 255),
             shadow_radius=None, max_width=None, background=False, background_color=(0, 0, 0, 255)):
    # Convert OpenCV image to PIL image
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, font_size)

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
        line_width, line_height = draw.textsize(line, font=font)
        max_line_width = max(max_line_width, line_width)
        total_height += line_height

        text_x = (img_width - line_width) / 2  # Adjusted to use numpy width
        line_y = position + y_offset
        y_offset += (line_height + 3)

        if background:
            (text_x, line_y), _ = add_background(image_pil, draw, (text_x, line_y), line, font, fill_color=background_color, border_radius=10)

        if shadow_radius:
            shadow_position = (text_x, line_y)
            add_blurred_shadow(draw, image_pil, line, shadow_position, font, shadow_color=shadow_color,
                               blur_radius=shadow_radius)

        draw.text((text_x, line_y), line, font=font, fill=font_color)
        # y_offset += (line_height + 3)  # Increment or decrement y_offset for next line depending on alignment

    return np.array(image_pil), (max_line_width, total_height)


# Iterate through each item in the JSON
for item in data:
    # Read the image
    img_path = item["image"]
    img = cv2.imread("images/input/" + img_path, cv2.IMREAD_UNCHANGED)  # Ensure to use correct path to images
    rating_offset = 15
    text_offset = 30
    if img is not None:
        # Resize image to 1080x1920
        img = cv2.resize(img, (1080, 1920), interpolation=cv2.INTER_AREA)

        # Calculate positions for the text
        top_center = int(img.shape[0] * 0.13)
        bottom_center = int(img.shape[0] * 0.70)

        # Add month and rating at the top center, one above the other
        img, (_, month_height) = add_text(img, item["month"], top_center, font_path, 142, shadow_radius=4)
        img, (_, _) = add_text(img, f'Comfortability: {item["rating"]}%', top_center + month_height + rating_offset, font_path, 54,
                               font_color=(216, 203, 170), shadow_radius=5, shadow_color=(0, 0, 0), background=True, background_color=(0,0,0,80))

        # Add name and description at the bottom center, one above the other
        img, (_, name_height) = add_text(img, item["name"], bottom_center, font_path, 112,
                                         max_width=15, shadow_radius=4)
        img, (_, _) = add_text(img, f'"{item["description"]}"', bottom_center + name_height + text_offset, font_path, 42,
                               max_width=43, shadow_radius=4)  # Adjust for wrapped text

        # Convert back to OpenCV image and save
        cv2.imwrite("images/output/" + f'{item["month"]}.png', img)
    else:
        print(f"Image {img_path} not found.")
