import gradio as gr
import utils.font_manager as font_manager


def render_color_opacity_picker():
    with gr.Group():
        with gr.Row():
            color = gr.ColorPicker(label="Font Color", scale=1)
            opacity = gr.Slider(0, 100, value=100, label="Opacity", scale=2)

    return color, opacity


def bind_checkbox_to_visibility(checkbox, group):
    checkbox.change(
        lambda state: gr.Group(visible=state),
        inputs=checkbox,
        outputs=group
    )


def render_font_picker(default_font_size=55):
    font_families = font_manager.get_fonts()
    with gr.Group():
        with gr.Row():
            font_families_list = list(font_families.keys())
            initial_font_family = font_families_list[0]
            font_family = gr.Dropdown(font_families_list, value=initial_font_family, label="Font Family")
            font_styles_list = list(font_families[initial_font_family].keys())
            font_style = gr.Dropdown(font_styles_list, value=font_styles_list[0], label="Font Style")

        def update_font_styles(selected_font_family):
            if selected_font_family is None or selected_font_family == "":
                return []
            font_syles = list(font_families[selected_font_family].keys())
            return gr.Dropdown(font_syles, value=font_syles[0], label="Font Style")

        font_family.change(update_font_styles, inputs=[font_family], outputs=[font_style])
    with gr.Group():
        font_color, font_opacity = render_color_opacity_picker()
        font_size = gr.Number(default_font_size, label="Font Size")

    return font_family, font_style, font_color, font_opacity, font_size