import glob
import os
from fontTools.ttLib import TTFont
import inflect
import utils.path_handler as path_handler

font_families = None
p = None
fonts_dirs = [
    # MacOS
    "/Library/Fonts", "~/Library/Fonts", "System/Library/Fonts",
    # Linux
    "/usr/share/fonts", "~/.fonts",
    # Windows
    "C:\\Windows\\Fonts"
]


def initialize_inflect():
    global p
    if p is None:
        p = inflect.engine()

    return p


def initialize_fonts():
    global font_families
    if font_families is None:
        font_families = font_families

    font_files = []
    # Add TrendGenie fonts
    trendgenie_fonts_dir = os.path.join(path_handler.get_default_path(), "fonts")
    fonts_dirs.append(trendgenie_fonts_dir)
    for fonts_dir in fonts_dirs:
        print("Searching for fonts in (if it exists)", fonts_dir)
        fonts_dir = os.path.expanduser(fonts_dir)
        if not os.path.exists(fonts_dir):
            continue
        font_files += glob.glob(os.path.join(fonts_dir, "**/*.ttf"), recursive=True)
        font_files += glob.glob(os.path.join(fonts_dir, "**/*.otf"), recursive=True)

    font_families = {}
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
            if family_name not in font_families:
                font_families[family_name] = {}
            font_families[family_name][style_name] = font_file

    return font_families

def get_fonts():
    global font_families
    if font_families is None:
        font_families = initialize_fonts()

    return font_families

def get_inflect():
    global p
    if p is None:
        p = initialize_inflect()

    return p