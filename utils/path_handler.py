"""
This module contains functions for handling paths.
"""
import os
from pathlib import Path

DEFAULT_PATH = None


def get_default_path() -> str:
    """
    Gets the default path for saving files, which is the user's home directory under a folder called "trendgenie".
    :return: The default path.
    """
    global DEFAULT_PATH  # pylint: disable=global-statement
    if DEFAULT_PATH is None:
        homepath = Path.home()
        DEFAULT_PATH = os.path.join(homepath, "trendgenie")

    return DEFAULT_PATH
