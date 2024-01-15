"""
This module contains dataclasses and type aliases used in the project.
"""
from typing import Union
from dataclasses import dataclass


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


RGBColor = Union[str, tuple[int, int, int]]
