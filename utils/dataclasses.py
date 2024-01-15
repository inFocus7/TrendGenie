"""
This module contains dataclasses and type aliases used in the project.
"""
from typing import Union, Optional
from dataclasses import dataclass
import gradio as gr


@dataclass
class FourEdges:
    """
    A dataclass representing the four edges of a rectangle.
    """
    top: int
    bottom: int
    left: int
    right: int


@dataclass
class Position:
    """
    A dataclass representing a position on a 2d plane.
    """
    x: int
    y: int


@dataclass
class Size:
    """
    A dataclass representing a size on a 2d plane.
    """
    width: int
    height: int


@dataclass
class OpenAIGradioComponents:
    """
    A dataclass representing the components of the OpenAI API.
    """
    api_key: gr.Textbox
    api_text_model: Optional[gr.Dropdown]
    api_image_model: Optional[gr.Dropdown]


@dataclass
class RGBOpacity:
    """
    A dataclass representing an RGB color with an opacity value.
    """
    color: tuple[int, int, int]
    opacity: int


@dataclass
class MinMax:
    """
    A dataclass representing a minimum and maximum value.
    """
    min: int
    max: int


@dataclass
class RowCol:
    """
    A dataclass representing a row and column.
    """
    row: int
    col: int


@dataclass
class FontGradioComponents:
    """
    A dataclass representing the components of the font editor.
    """
    family: gr.Dropdown
    style: gr.Dropdown
    color: gr.ColorPicker
    opacity: gr.Slider
    size: gr.Number


@dataclass
class FontDropShadowGradioComponents:
    """
    A dataclass representing the components of the drop shadow editor.
    """
    enabled: gr.Checkbox
    color: gr.ColorPicker
    opacity: gr.Slider
    radius: gr.Number


@dataclass
class FontBackgroundGradioComponents:
    """
    A dataclass representing the components of the background editor.
    """
    enabled: gr.Checkbox
    color: gr.ColorPicker
    opacity: gr.Slider


RGBColor = Union[str, tuple[int, int, int]]
