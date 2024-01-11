"""
This module contains functions for handling paths.
"""
import os
from pathlib import Path

default_path = None


def get_default_path() -> str:
    """
    Gets the default path for saving files, which is the user's home directory under a folder called "trendgenie".
    :return:
    """
    global default_path
    if default_path is None:
        homepath = Path.home()
        default_path = os.path.join(homepath, "trendgenie")

    return default_path
