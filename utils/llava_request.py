import requests
import json
import base64

from io import BytesIO
from PIL import Image

def to_base_64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_str

def llava_request(prompt, model_name, image_paths):
    # load images
    images = []
    for image_path in image_paths:
        if image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
            image = Image.open(image_path)
            images.append(image)
    #convert to base64
    base64_images = []

    for image in images:
        base64_images.append(to_base_64(image))
        
    payload={"model":model_name, "prompt":prompt, "images": base64_images, "stream": False}
    while(True):
        try:
            r = requests.post("http://localhost:11434/api/generate", data=json.dumps(payload))
            return json.loads(r.text)["response"]
        except Exception as e:
            print(e)
            continue

if(__name__ == "__main__"):
    response = llava_request("What is in the image", "llava:7b", "./cropped_bbox_0.jpg")
    print(response)