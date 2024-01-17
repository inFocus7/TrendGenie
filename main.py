"""
This is the main file for the web app. It launches the web app and initializes the font manager and inflect engine.
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*

from ui import ui
from utils import font_manager

if __name__ == '__main__':
    # Initialize fonts, and svg file grabber at start
    font_manager.initialize_fonts()
    font_manager.initialize_inflect()
    ui.WebApp.launch(share=False, server_name='0.0.0.0')
