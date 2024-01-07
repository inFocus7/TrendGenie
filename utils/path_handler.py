import os
from pathlib import Path

default_path = None

def get_default_path():
    global default_path
    if default_path is None:
        homepath = Path.home()
        default_path = os.path.join(homepath, "trendgenie")

    return default_path

