import requests
from accelerate import Accelerator
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# Initialize the accelerator
accelerator = Accelerator()

model_id = "microsoft/Phi-3-vision-128k-instruct"

# Load model and processor
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="eager",
)  # Use _attn_implementation='eager' to disable flash attention

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Prepare the model for distributed training
model = accelerator.prepare(model)
print(accelerator.device)

messages = [
    {
        "role": "user",
        "content": "<|image_1|>Is there smoke in the image?",
    },
    {
        "role": "assistant",
        "content": """You are given an image of a horizon scene. Your task is to determine if there is smoke in the image. Look for any smoke-like objects that seem to expand in size, as this could indicate the presence of smoke. Output "yes" if you see smoke, and "no" otherwise.""",
    },
]

url = "https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png"
image = Image.open("test/false_negatives/1.jpg")

prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = processor(prompt, [image], return_tensors="pt").to(accelerator.device)

generation_args = {
    "max_new_tokens": 500,
    "temperature": 0.0,
    "do_sample": False,
}

generate_ids = model.generate(
    **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
)

# Remove input tokens
generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
while True:
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(response)
