import base64
import copy
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import gradio as gr
import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

model_id = "microsoft/Phi-3-vision-128k-instruct"
device = "cuda:0"
dtype = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="eager",
)  # use _attn_implementation='eager' to disable flash attention
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def generate_caption(prompt, assistant, image):
    messages = [
        {"role": "user", "content": f"<|image_1|>\n{prompt}"},
        {
            "role": "assistant",
            "content": assistant,
        },
    ]
    processed_prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    generation_args = {
        "max_new_tokens": 500,
        "temperature": 0.0,
        "do_sample": False,
    }
    model_inputs = processor(processed_prompt, images=image, return_tensors="pt").to(
        model.device
    )
    generate_ids = model.generate(
        **model_inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args,
    )

    # remove input tokens
    generate_ids = generate_ids[:, model_inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response


# Create the Gradio interface
iface = gr.Interface(
    fn=generate_caption,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Textbox(label="Assistant"),
        gr.Image(type="pil", label="Upload JPEG Image"),
    ],
    outputs="text",
    title="Image Captioning with PALIGEMMA-3B-MIX-448",
    description="Enter a prompt and upload a JPEG image. The model will generate a caption.",
)

# Launch the Gradio app
iface.launch()
