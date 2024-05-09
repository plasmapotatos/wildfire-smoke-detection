import requests
import json
import base64

def llava_request(prompt, model_name, images):
    payload={"model":model_name, "prompt":prompt, "images": images, "stream": False}
    while(True):
        try:
            r = requests.post("http://localhost:11434/api/generate", data=json.dumps(payload))
            return json.loads(r.text)["response"]
        except Exception as e:
            print(e)
            continue

if(__name__ == "__main__"):
    with open("./cropped_bbox_0.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        response = llava_request("What is in the image", "llava:7b", [encoded_string.decode("utf-8")])
        print(response)