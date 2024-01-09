#!/usr/bin/env python3
# -*- coding: utf-8 -*

import ui.ui as ui
import utils.font_manager as font_manager

if __name__ == '__main__':
    # Initialize font manager / fonts at start
    font_manager.initialize_fonts()
    font_manager.initialize_inflect()
    ui.WebApp.launch(share=False, server_name='0.0.0.0')
