import base64
import copy
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import requests
import torch
from PIL import Image

from llava.eval.run_llava import eval_model
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from prompts import stitched_prompt

model_path = "liuhaotian/llava-v1.5-7b"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name
)


# Define the HTTP request handler
class RequestHandler(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()

    # Define the do_GET method to handle GET requests
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length).decode("utf-8")

        parsed_data = parse_qs(post_data)
        print(len(parsed_data["images"]))

        if "prompt" in parsed_data:
            image_paths = None
            images = None
            if "image_paths" in parsed_data:
                image_paths = parsed_data["image_paths"][0]
            elif "images" in parsed_data:
                image_b64s = parsed_data["images"]
                image_bytes = [base64.b64decode(image_b64) for image_b64 in image_b64s]
                images = [Image.open(BytesIO(image_byte)) for image_byte in image_bytes]
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing image")
                return
            prompt = parsed_data["prompt"][0]

            print(prompt, images)

            result = prompt_llava(prompt, image_paths, images)
            self._set_response()
            self.wfile.write(str(result).encode())
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing parameters")


def prompt_llava(prompt, image_paths=None, images=None):
    print("prompt:", prompt)
    if image_paths is None and images is None:
        raise ValueError("Either image_path or image must be provided.")
    if image_paths is not None:
        images = []
        for image_path in image_paths:
            image = Image.open(image_path)
            images.append(image)

    results = []

    for image in images:
        args = type(
            "Args",
            (),
            {
                "model_path": model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query": prompt,
                "conv_mode": None,
                "images": [image],
                "sep": ",",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512,
                "tokenizer": tokenizer,
                "model": model,
                "image_processor": image_processor,
                "context_len": context_len,
            },
        )()
        output = eval_model(args)
        results.append(output)
    print(results)

    return results


def run(server_class=HTTPServer, handler_class=RequestHandler, port=8000):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
