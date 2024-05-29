import io
import os

import gradio as gr
import kagglehub
import keras
import keras_nlp
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForPreTraining,
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
)

# Load the pre-trained model and tokenizer
os.environ["KERAS_BACKEND"] = "jax"
keras.config.set_floatx("bfloat16")
pali_gemma_lm = keras_nlp.models.PaliGemmaCausalLM.from_preset("pali_gemma_3b_mix_224")


def pil_to_np(image):
    # Remove alpha channel if neccessary.
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image


# Define the inference function
def generate_caption(prompt, image_paths):
    # Preprocess the image
    images = [Image.open(image_path) for image_path in image_paths]
    images = [pil_to_np(image) for image in images]
    print(prompt, images)
    model_inputs = {
        "images": images,
        "prompts": [prompt] * len(images),
    }
    # input_len = model_inputs["input_ids"].shape[-1]
    # with torch.no_grad():
    #     outputs = model.generate(inputs={
    #         "images": images,
    #         "prompts": [prompt] * len(images),
    #     }, max_new_tokens=500)
    outputs = pali_gemma_lm.generate(inputs=model_inputs)
    return outputs


# Create the Gradio interface
iface = gr.Interface(
    fn=generate_caption,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.File(file_types=["image"], file_count="multiple", label="Upload Images"),
    ],
    outputs="text",
    title="Image Captioning with PALIGEMMA-3B-MIX-448",
    description="Enter a prompt and upload a JPEG image. The model will generate a caption.",
)

# Launch the Gradio app
iface.launch()
