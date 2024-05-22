import requests
import torch
from PIL import Image
from transformers import (
    AutoModelForPreTraining,
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
)

from utils.image_utils import extract_and_parse_coordinates, overlay_bbox

model_id = "google/paligemma-3b-mix-224"
device = "cuda:0"
dtype = torch.bfloat16

image = Image.open("test/test.jpg")

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
).eval()
processor = AutoProcessor.from_pretrained(model_id)

prompt = "detect en Detect the smoke in the image"
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(
    model.device
)
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    width = image.width
    height = image.height
    coords = extract_and_parse_coordinates(decoded, width, height)
    print(decoded, coords)
    if coords:
        image = overlay_bbox(image, coords)
    image.save("test/test_smoke_bbox.jpg")
