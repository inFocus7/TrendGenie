"""
This file contains functions for image processing.
"""
from typing import Tuple
import cv2
import numpy as np
from utils import dataclasses


def get_alpha_from_opacity(opacity: int) -> int:
    """
    Converts an opacity value from 0-100 to 0-255.
    :param opacity: The opacity value from 0-100.
    :return: The opacity value from 0-255.
    """
    # Opacity should be 0 -> 0, 100 -> 255
    return int(opacity * 255 / 100)


def get_rgba(color: dataclasses.RGBColor, opacity: int) -> Tuple[int, int, int, int]:
    """
     Gets the RGBA value for a given color and opacity.
     :param color: The color to use. Either a hex string or a tuple of RGB values.
     :param opacity: The opacity to use, from 0 to 100.
     :return: The RGBA value.
     """
    # if color is hex, convert to rgb
    if not isinstance(color, tuple) and color.startswith("#"):
        color = color.lstrip("#")
        color = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    return color[0], color[1], color[2], get_alpha_from_opacity(opacity)


def get_bgra(color: dataclasses.RGBColor, opacity: int) -> Tuple[int, int, int, int]:
    """
     Gets the BGRA value for a given color and opacity.
     :param color: The color to use. Either a hex string or a tuple of BGR values.
     :param opacity: The opacity to use, from 0 to 100.
     :return: The BGRA value.
     """
    # if color is hex, convert to rgb
    if not isinstance(color, tuple) and color.startswith("#"):
        color = color.lstrip("#")
        color = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    return color[2], color[1], color[0], get_alpha_from_opacity(opacity)


def open_image_as_rgba(image_path: str) -> np.ndarray:
    """
    Opens an image as RGBA.
    :param image_path: The path to the image.
    :return: The image as RGBA.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    return img


def blend_alphas(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Blends two images together using alpha blending.
    :param a: The first image.
    :param b: The second image.
    :return: The blended image.
    """
    if a.shape != b.shape:
        raise ValueError("both images must have the same shape to blend alphas")
    if a.shape[2] != 4 or b.shape[2] != 4:
        raise ValueError("both images must have 4 channels to blend alphas")

    alpha_text = a[:, :, 3] / 255.0
    alpha_canvas = b[:, :, 3] / 255.0
    alpha_final = alpha_text + alpha_canvas * (1 - alpha_text)

    final = np.zeros_like(b)
    # alpha blend
    for c in range(3):  # Loop over color (non-alpha) channels
        final[:, :, c] = (alpha_text * a[:, :, c] + alpha_canvas * (1 - alpha_text) *
                                 b[:, :, c]) / alpha_final
    final[:, :, 3] = alpha_final * 255
    final[:, :, :3][alpha_final == 0] = 0

    return final
