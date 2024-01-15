"""
This file contains functions for image processing.
"""
from typing import Tuple, Union


def get_alpha_from_opacity(opacity: int) -> int:
    """
    Converts an opacity value from 0-100 to 0-255.
    :param opacity: The opacity value from 0-100.
    :return: The opacity value from 0-255.
    """
    # Opacity should be 0 -> 0, 100 -> 255
    return int(opacity * 255 / 100)


def get_rgba(color: Union[str, Tuple[int, int, int]], opacity: int) -> Tuple[int, int, int, int]:
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


def get_bgra(color: Union[str, Tuple[int, int, int]], opacity: int) -> Tuple[int, int, int, int]:
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
