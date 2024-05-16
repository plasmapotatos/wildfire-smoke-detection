import requests


def send_post_request(image_path, prompt):
    url = "http://localhost:8000"
    payload = {"image_path": image_path, "prompt": prompt}

    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            return response.text
        else:
            print("Failed to send POST request. Status code:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("Error:", e)


# Example usage:
image_path = "/home/wei/wildfire-smoke-dataset/test_no_smoke.jpg"
prompt = """Do you see smoke in this image? If so, where is it?"""
print(send_post_request(image_path, prompt))
