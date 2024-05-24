import base64
import json
import socket
import tempfile
from io import BytesIO

import numpy as np
import requests
from gradio_client import Client, file
from PIL import Image

from utils.image_utils import extract_and_parse_coordinates, overlay_bbox
from utils.prompts import LLAVA_PROMPT, PALIGEMMA_PROMPT


def to_base_64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_str


def llava_request(prompt, model_name, image_paths=None, images=None):
    if image_paths is None and images is None:
        raise ValueError("Either image_paths or images must be provided")
    # load images
    images = []
    if images is not None:
        for image in images:
            images.append(image)
    else:
        for image_path in image_paths:
            if image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
                image = Image.open(image_path)
                images.append(image)
    # convert to base64
    base64_images = []

    for image in images:
        base64_images.append(to_base_64(image))

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": base64_images,
        "stream": False,
    }
    while True:
        try:
            r = requests.post(
                "http://localhost:11434/api/generate", data=json.dumps(payload)
            )
            return json.loads(r.text)["response"]
        except Exception as e:
            print(e)
            continue


def prompt_llava(prompt, image_paths=None, images=None):
    if image_paths is None and images is None:
        raise ValueError("Either image_path or image must be provided")

    results = []

    if images is not None:
        processed_images = []
        for image in images:
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            img_b64 = base64.b64encode(buffer.getvalue())
            processed_images.append(img_b64)

        url = "http://localhost:8000"
        headers = {"Content-type": "application/json", "Accept": "text/plain"}
        payload = {"images": processed_images, "prompt": prompt}

        try:
            response = requests.post(url, data=payload, headers=headers)
            if response.status_code == 200:
                return response.text
            else:
                print(
                    "Failed to send POST request. Status code:",
                    response.status_code,
                )
        except requests.exceptions.RequestException as e:
            print("Error:", e)
    else:
        url = "http://localhost:8000"
        payload = {"image_paths": image_paths, "prompt": prompt}

        try:
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                return response.text
            else:
                print("Failed to send POST request. Status code:", response.status_code)
        except requests.exceptions.RequestException as e:
            print("Error:", e)


def prompt_llava_next(prompt, image_paths=None, images=None):
    if image_paths is None and images is None:
        raise ValueError("Either image_path or image must be provided")

    results = []

    if images is not None:
        processed_images = []
        for image in images:
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            img_b64 = base64.b64encode(buffer.getvalue())
            processed_images.append(img_b64)

        url = "http://localhost:8000"
        headers = {"Content-type": "application/json", "Accept": "text/plain"}
        payload = {"images": processed_images, "prompt": prompt}

        try:
            response = requests.post(url, data=payload, headers=headers)
            if response.status_code == 200:
                return response.text
            else:
                print(
                    "Failed to send POST request. Status code:",
                    response.status_code,
                )
        except requests.exceptions.RequestException as e:
            print("Error:", e)
    else:
        url = "http://localhost:8000"
        payload = {"image_paths": image_paths, "prompt": prompt}

        try:
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                return response.text
            else:
                print("Failed to send POST request. Status code:", response.status_code)
        except requests.exceptions.RequestException as e:
            print("Error:", e)


def prompt_paligemma(prompt, image_paths=None, images=None, client=None):
    if not client:
        client = Client("http://127.0.0.1:7860/")

    results = []
    if images is not None:
        for image in images:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as temp:
                # Save the image to the temporary file
                image.save(temp, format="JPEG")
                temp.flush()
                try:
                    output = client.predict(
                        prompt=prompt,
                        image=file(temp.name),
                        api_name="/predict",
                    )
                    results.append(output)
                except requests.exceptions.RequestException as e:
                    print("Error:", e)
    else:
        for image_path in image_paths:
            try:
                output = client.predict(
                    prompt=prompt,
                    image=file(image_path),
                    api_name="/predict",
                )
                results.append(output)
            except requests.exceptions.RequestException as e:
                print("Error:", e)

    return results


prompt = """You are a proficient smoke detector at a fire tower. Does the following image contain wildfire smoke? Look carefully, and distinguish between clouds and smoke. Output "yes" only if there is smoke, and "no" only if there is no smoke. Be conservative, and only output "yes" if you are sure there is smoke. Reason out your logic, and enclose it in <Reasoning> <Reasoning/>. *****Do NOT go over 50 words*****. If you find yourself repeating yourself in your reasoning, stop your reasoning immediately. Then, output one line which is either "yes" or "no", enclosing it in <Output> <Output/>."""

temp = """You are a proficient smoke detector at a fire tower. Does the following image contain wildfire smoke? Look carefully, and distinguish between clouds and smoke. Reason out your logic, and enclose it in <Reasoning> <Reasoning/>. *****Do NOT go over 50 words*****. If you find yourself repeating yourself in your reasoning, stop your reasoning immediately. Then, output one line which is either "yes" or "no", enclosing it in <Output> <Output/>.

*****Example Image with Smoke Reasoning*****
The image shows a large plume of dark smoke rising from the mountains, which is indicative of a wildfire. The smoke is distinct from the surrounding clouds and vegetation, and its presence suggests a fire is occurring. The smoke is not a natural occurrence in this context, and its presence is a clear indication of a wildfire. The smoke is not a cloud, as it is not associated with precipitation or weather patterns.
*****End of Example Image with Smoke Reasoning*****

*****Example Image without Smoke Reasoning*****
The image shows a clear blue sky with no visible signs of smoke or haze. The sky is devoid of any particles or discoloration that would indicate the presence of wildfire smoke. The absence of any visible signs of smoke or haze leads to the conclusion that there is no wildfire smoke present in the image
*****End of Example Image without Smoke Reasoning*****

Remember to keep your reasoning concise, not more than 50 words, and end with a <Output> tag enclosing your final answer, for example <Output>yes<Output/> or <Output>no<Output/>."""

test = """"You are a proficient smoke detector at a fire tower. Does the following image contain wildfire smoke? Look carefully, and distinguish between clouds and smoke. Reason out your logic. Then, output one line which is either "yes" or "no".
"""

if __name__ == "__main__":
    # image_bytes = BytesIO()
    image = Image.open("test/test.jpg")
    response = prompt_paligemma(PALIGEMMA_PROMPT, images=[image])
    print(response)
    # with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as temp:
    #     # Save the image to the temporary file
    #     image.save(temp, format="JPEG")
    #     temp.flush()
    #     response = prompt_paligemma(PALIGEMMA_PROMPT, image_paths=[temp.name])
    #     print(response)
    # image_paths = []
    # for i in range(4):
    #     for j in range(4):
    #         image_paths.append(f"tile_{i}_{j}.jpg")
    # images = [Image.open(image_path) for image_path in image_paths]
    # responses = prompt_paligemma(PALIGEMMA_PROMPT, image_paths=image_paths)
    # width = images[0].width
    # height = images[0].height
    # coords = [
    #     extract_and_parse_coordinates(response, width, height) for response in responses
    # ]
    # print(coords)
    # results = []
    # for r in range(4):
    #     for c in range(4):
    #         if coords[4 * r + c] is not None:
    #             results.append(overlay_bbox(images[4 * r + c], coords[4 * r + c]))
    #         else:
    #             results.append(images[4 * r + c])
    # for i, img in enumerate(results):
    #     img.save(f"test/output_{i}.jpg")
