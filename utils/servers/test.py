import gradio as gr
from PIL import Image

def process_batch(images):
    # Example processing: return image sizes
    print(images)
    results = [image.size for image in images]
    return results

iface = gr.Interface(
    fn=process_batch, 
    inputs=gr.File(file_types=["image"], file_count="multiple"), 
    outputs=gr.Textbox(label="Image Sizes"),
    title="Batch Image Processing",
    description="Upload multiple images to process them in batch."
)

iface.launch()
