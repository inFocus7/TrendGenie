"""
This file contains the main UI code that runs the TrendGenie web app.
"""
import os
import gradio as gr
import ui.listicles.interface as listicle_interface
import ui.music.interface as music_interface

# Read the styles.css file and add it to the page.
css_file = os.path.join(os.path.dirname(__file__), "styles.css")
with open(css_file, "r", encoding="utf-8") as file:
    css = file.read()

with gr.Blocks(theme=gr.themes.Soft(), css=css) as WebApp:
    # Header
    with gr.Column(elem_id="header"):
        gr.Image("static/logo-v2.png", label="Logo", show_label=False, image_mode="RGBA", container=False,
                 show_share_button=False, show_download_button=False, width=50, elem_id="header-logo")
        gr.Markdown("# TrendGenie", elem_id="header-title")
        gr.Markdown("## Your content creation assistant.", elem_id="header-subtitle")
    # Content
    with gr.Tab("Listicles"):
        listicle_interface.render_listicles_section()
    with gr.Tab("Music Cover Videos"):
        music_interface.render_music_section()
    # Footer
    with gr.Group(elem_id="footer"):
        gr.Image("static/hero-face.svg", label="Logo", show_label=False,
                 image_mode="RGBA", container=False, width=50, elem_id="footer-logo",
                 show_download_button=False, show_share_button=False)
        gr.Markdown("**Made by [inf0](https://github.com/infocus7).**", elem_id="footer-text")
