import io
import os

import kagglehub
import keras
import keras_nlp
import numpy as np
import requests
from PIL import Image

kagglehub.login()

os.environ["KERAS_BACKEND"] = "jax"
keras.config.set_floatx("bfloat16")
pali_gemma_lm = keras_nlp.models.PaliGemmaCausalLM.from_preset("pali_gemma_3b_mix_224")


def read_image(url):
    contents = io.BytesIO(requests.get(url).content)
    image = Image.open(contents)
    print(image.size)
    image = np.array(image)
    # Remove alpha channel if neccessary.
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image


image_url = "https://storage.googleapis.com/keras-cv/models/paligemma/cow_beach_1.png"
image = read_image(image_url)

prompts = [
    "answer en where is the cow standing?\n",
    "answer en what color is the cow?\n",
    "describe en\n",
    "detect cow\n",
    "segment cow\n",
]
images = [image, image, image, image, image]
outputs = pali_gemma_lm.generate(
    inputs={
        "images": images,
        "prompts": prompts,
    }
)
for output in outputs:
    print(output)
