"""
This module manages the fonts and the inflect engine.
"""
import glob
import os
from typing import Optional, Dict
from fontTools.ttLib import TTFont
import inflect
from utils import path_handler

NestedDict = Dict[str, Dict[str, str]]

FONT_FAMILIES: Optional[NestedDict] = None
P: Optional[inflect.engine] = None
FONTS_DIRS = [
    # MacOS
    "/Library/Fonts", "~/Library/Fonts", "System/Library/Fonts",
    # Linux
    "/usr/share/fonts", "~/.fonts",
    # Windows
    "C:\\Windows\\Fonts"
]


def initialize_inflect() -> inflect.engine:
    """
    Initializes the inflect engine.
    :return: The inflect engine.
    """
    global P  # pylint: disable=global-statement
    if P is None:
        P = inflect.engine()

    return P


def initialize_fonts() -> NestedDict:
    """
    Initializes the font families from the global FONTS_DIRS.
    :return: The font families and their paths. They are called by map[font_family][font_style].
    """
    global FONT_FAMILIES  # pylint: disable=global-statement

    font_files = []
    # Add TrendGenie fonts
    trendgenie_fonts_dir = os.path.join(path_handler.get_default_path(), "fonts")
    FONTS_DIRS.append(trendgenie_fonts_dir)
    for fonts_dir in FONTS_DIRS:
        fonts_dir = os.path.expanduser(fonts_dir)
        if not os.path.exists(fonts_dir):
            continue
        font_files += glob.glob(os.path.join(fonts_dir, "**/*.ttf"), recursive=True)
        font_files += glob.glob(os.path.join(fonts_dir, "**/*.otf"), recursive=True)

    FONT_FAMILIES = {}
    for font_file in font_files:
        font = TTFont(font_file)
        name = font['name']
        family_name = ""
        style_name = ""
        for record in name.names:
            if record.nameID == 1 and b'\000' in record.string:
                family_name = record.string.decode('utf-16-be').rstrip('\0')
            elif record.nameID == 2 and b'\000' in record.string:
                style_name = record.string.decode('utf-16-be').rstrip('\0')
        if family_name and style_name:
            if family_name not in FONT_FAMILIES:
                FONT_FAMILIES[family_name] = {}
            FONT_FAMILIES[family_name][style_name] = font_file

    return FONT_FAMILIES


def get_fonts() -> NestedDict:
    """
    Gets the font families. If they are not initialized, it initializes them.
    :return: The font families and their paths. They are called by map[font_family][font_style].
    """
    global FONT_FAMILIES  # pylint: disable=global-statement
    if FONT_FAMILIES is None:
        FONT_FAMILIES = initialize_fonts()

    return FONT_FAMILIES


def get_inflect() -> inflect.engine:
    """
    Gets the inflect engine. If it is not initialized, it initializes it.
    :return: The inflect engine.
    """
    global P  # pylint: disable=global-statement
    if P is None:
        P = initialize_inflect()

    return P
