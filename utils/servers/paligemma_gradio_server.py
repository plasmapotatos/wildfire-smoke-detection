import io

import gradio as gr
import torch
from PIL import Image
from transformers import (
    AutoModelForPreTraining,
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
)

# Load the pre-trained model and tokenizer
model_id = "google/paligemma-3b-mix-224"
device = "cuda:0"
dtype = torch.bfloat16

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
).eval()
processor = AutoProcessor.from_pretrained(model_id)


# Define the inference function
def generate_caption(prompt, image):
    # Preprocess the image
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(
        model.device
    )
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)

    return decoded


# Create the Gradio interface
iface = gr.Interface(
    fn=generate_caption,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Image(type="pil", label="Upload JPEG Image"),
    ],
    outputs="text",
    title="Image Captioning with PALIGEMMA-3B-MIX-448",
    description="Enter a prompt and upload a JPEG image. The model will generate a caption.",
)

# Launch the Gradio app
iface.launch()
