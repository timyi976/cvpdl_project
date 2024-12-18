import gradio as gr
import PIL

from gen_layout import load_layout_planner_model, generate as generate_layout
from gen_style_prompt import generate_style
from gen_image import get_args as get_image_gen_args, load_model as load_image_gen_model, generate_image

layout_planner, layout_tokenizer = load_layout_planner_model()
image_gen_args = get_image_gen_args()
_, image_gen_model = load_image_gen_model(image_gen_args)

def cut_blank_on_image(image: PIL.Image):
    # crop the top 512 * 512
    image = image.crop((0, 0, 512, 512))
    return image


# Define a function to generate image based on text input
def pipeline(prompt):
    global layout_planner, layout_tokenizer
    ocrs = generate_layout(layout_planner, layout_tokenizer, [prompt])
    print(ocrs)

    style_prompt = generate_style(prompt, ocrs[0])

    image = generate_image(image_gen_args, image_gen_model, style_prompt)
    image = cut_blank_on_image(image)

    return style_prompt, image


# generate_image('An airplane with a livery of yellow "Sun" and blue "Airlines".')

# Define the layout
with gr.Blocks() as demo:
    # Create a two-column layout
    with gr.Row():
        with gr.Column():
            # Left column: Text input box and "Generate" button
            prompt_input = gr.Textbox(label="Enter your prompt", placeholder="Describe an image...", lines=3)
            generate_button = gr.Button("Generate")
        
        with gr.Column():
            # Right column: Display generated prompt and image
            prompt_output = gr.Textbox(label="Generated Prompt", interactive=False, lines=3)
            image_output = gr.Image(label="Generated Image")
    
    # Link the button to the image generation function
    generate_button.click(pipeline, inputs=prompt_input, outputs=[prompt_output, image_output])

# Launch the demo
demo.launch(share=True)