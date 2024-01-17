"""
This module contains utility functions for rendering widely-used Gradio components.
"""
import gradio as gr
from utils import font_manager, dataclasses


def render_color_opacity_picker(default_name_label: str = "Font") -> dataclasses.ColorOpacityGradioComponents:
    """
    Renders a color picker with the appropriate styling.
    :param default_name_label: The default name label to use.
    :return: A class containing the color and opacity components.
    """
    with gr.Group():
        with gr.Row():
            color = gr.ColorPicker(label=f"{default_name_label} Color", scale=1, interactive=True)
            opacity = gr.Slider(0, 100, value=100, label="Opacity", scale=2, interactive=True)

    return dataclasses.ColorOpacityGradioComponents(color, opacity)


def bind_checkbox_to_visibility(checkbox: gr.Checkbox, group: gr.Group):
    """
    Binds a checkbox to the visibility of a group. When the checkbox is checked, the group is visible, and when the
    checkbox is unchecked, the group is hidden.
    :param checkbox: The Checkbox component to bind.
    :param group: The Group component to bind.
    """
    checkbox.change(
        lambda state: gr.Group(visible=state),
        inputs=checkbox,
        outputs=group
    )


def render_font_picker(default_font_size: int = 55) -> dataclasses.FontGradioComponents:
    """
    Renders a font picker with the appropriate styling.
    :param default_font_size: The default font size to use.
    :return: A tuple containing the font family, font style, font color, font opacity, and font size components.
    """
    font_families = font_manager.get_fonts()
    with gr.Group():
        with gr.Row():
            font_families_list = list(font_families.keys())
            initial_font_family = font_families_list[0] if len(font_families_list) > 0 else ""
            font_family = gr.Dropdown(font_families_list, value=initial_font_family, label="Font Family",
                                      interactive=True)
            font_styles_list = list(font_families[initial_font_family].keys() if initial_font_family else [])
            initial_font_style = font_styles_list[0] if len(font_styles_list) > 0 else ""
            font_style = gr.Dropdown(font_styles_list, value=initial_font_style, label="Font Style", interactive=True)

        def update_font_styles(selected_font_family):
            if selected_font_family is None or selected_font_family == "":
                return []
            font_styles = list(font_families[selected_font_family].keys())
            return gr.Dropdown(font_styles, value=font_styles[0], label="Font Style")

        font_family.change(update_font_styles, inputs=[font_family], outputs=[font_style])
    with gr.Group():
        font_color_opacity = render_color_opacity_picker()
        font_size = gr.Number(default_font_size, label="Font Size", interactive=True)

    return dataclasses.FontGradioComponents(font_family, font_style, font_color_opacity.color,
                                            font_color_opacity.opacity, font_size)


def render_tool_description(description: str):
    """
    Renders a description for a tool with the appropriate styling.
    :param description: The description to render.
    """
    gr.Markdown(description, elem_classes=["tool-description"])
