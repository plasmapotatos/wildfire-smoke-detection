import gradio as gr
from PIL import Image

def process_batch(image_paths):
    # Example processing: return image sizes
    images = [Image.open(image_path) for image_path in image_paths]
    return images

iface = gr.Interface(
    fn=process_batch, 
    inputs=gr.File(file_types=["image"], file_count="multiple"), 
    outputs=gr.Textbox(label="Image Sizes"),
    title="Batch Image Processing",
    description="Upload multiple images to process them in batch."
)

iface.launch()
