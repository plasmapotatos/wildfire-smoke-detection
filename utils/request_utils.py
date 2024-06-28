import ast
import base64
import json
import socket
import tempfile
import tiktoken
from datetime import datetime
from io import BytesIO
import os

import numpy as np
import requests
from gradio_client import Client, file
from PIL import Image

from utils.image_utils import (
    extract_and_calculate_horizon,
    extract_and_parse_coordinates,
    overlay_bbox,
    stitch_image_with_bboxes,
)
from utils.prompts import (
    LLAVA_PROMPT,
    PALIGEMMA_DETECT_PROMPT,
    PALIGEMMA_SEGMENT_PROMPT,
    PHI3_ASSISTANT,
    PHI3_PROMPT,
    GPT4_BASIC_PROMPT,
    GPT4_REASONING_PROMPT,
)


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


def prompt_phi3(prompt, assistant, image_paths=None, images=None, client=None):
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
                        assistant=assistant,
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
                    assistant=assistant,
                    image=file(image_path),
                    api_name="/predict",
                )
                results.append(output)
            except requests.exceptions.RequestException as e:
                print("Error:", e)

    return results


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def prompt_gpt4(prompt, image_paths=None, images=None):
    api_key = os.environ.get("OPENAI_API_KEY")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    results = []

    if images is not None:
        for image in images:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as temp:
                # Save the image to the temporary file
                image.save(temp, format="JPEG")
                temp.flush()

                base64_image = encode_image(temp.name)
                payload["messages"][0]["content"][1]["image_url"]["url"] = f"data:image/jpeg;base64,{base64_image}"
                while True:
                    try:
                        response = requests.post(
                            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
                        )
                        print(response.json())
                        results.append(response.json()["choices"][0]["message"]["content"])
                        print(response.json()['usage']['total_tokens'])
                        break
                    except Exception as e:
                        print("Error:", e)
    else:
        for image_path in image_paths:
            base64_image = encode_image(image_path)
            payload["messages"][0]["content"][1]["image_url"]["url"] = f"data:image/jpeg;base64,{base64_image}"
            while True:
                try:
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
                    )
                    results.append(response.json()["choices"][0]["message"]["content"])
                    break
                except Exception as e:
                    print("Error:", e)

    return results


prompt = """Is there smoke in the image? How confident are you?"""

assistant = """You are given an image of a horizon scene. Your task is to determine if there is smoke in the image. Look for any smoke-like objects that seem to expand in size, as this could indicate the presence of smoke. Output "yes" if you see smoke, and "no" otherwise. Additionally, output a floating point number between 0 and 1 to indicating the chance of smoke. A value closer to 1 indicates higher chance of smoke."""

if __name__ == "__main__":
    image = Image.open("test/test_smoke.jpg")
    responses = prompt_phi3(GPT4_REASONING_PROMPT, "follow the prompt", images=[image])
    print(responses)
    # image = Image.open("test/frame2/tile_2.jpg")
    # responses = prompt_phi3(prompt, assistant, images=[image])
    # print(responses, datetime.now())
    # # Open the file in read mode
    # with open("test/tile_boxes.txt", "r") as tile_boxes_file:
    #     tile_boxes = [
    #         ast.literal_eval(line.strip().strip('"')) for line in tile_boxes_file
    #     ]
    # images = []
    # original_image = Image.open("test/test_smoke.jpg")
    # for i in range(10):
    #     image = Image.open(f"test/tile_{i}.jpg")
    #     images.append(image)
    # segment_responses = prompt_paligemma(PALIGEMMA_SEGMENT_PROMPT, images=images)
    # print(segment_responses)
    # detect_responses = prompt_paligemma(PALIGEMMA_DETECT_PROMPT, images=images)
    # horizon_xs = [
    #     extract_and_calculate_horizon(
    #         segment_response, images[0].width, images[0].height
    #     )
    #     for segment_response in segment_responses
    # ]
    # bboxs = [
    #     extract_and_parse_coordinates(
    #         detect_response, images[0].width, images[0].height
    #     )
    #     for detect_response in detect_responses
    # ]
    # print(horizon_xs)
    # print(tile_boxes)
    # print(bboxs)
    # stitched_image = stitch_image_with_bboxes(original_image, bboxs, tile_boxes)
    # stitched_image.save("test/stitched_image.jpg")
    # # image_bytes = BytesIO()
    # image = Image.open("test/test.jpg")
    # response = prompt_paligemma(PALIGEMMA_PROMPT, images=[image])
    # print(response)
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
