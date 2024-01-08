import gradio as gr


def render_openai_setup(show_text_model=True, show_image_model=True):
    api_text_model = None
    api_image_model = None
    with gr.Row():
        api_key = gr.Textbox(label="OpenAI API Key",
                             placeholder="Leave empty to use the OPENAI_API_KEY environment variable.",
                             lines=1, interactive=True, type="password")
        if show_text_model:
            api_text_model = gr.Dropdown(["gpt-3.5-turbo", "gpt-4"], label="API Model", value="gpt-3.5-turbo",
                                         interactive=True)
        if show_image_model:
            api_image_model = gr.Dropdown(["dall-e-2", "dall-e-3"], label="API Image Model", value="dall-e-2",
                                          interactive=True)

    return api_key, api_text_model, api_image_model
