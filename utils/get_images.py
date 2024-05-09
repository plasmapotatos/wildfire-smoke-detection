import os
import xml.etree.ElementTree as ET
import cv2
from PIL import Image

def get_images(image_dir):
    images = []
    for image in os.listdir(image_dir):
        if image.endswith('.jpeg') or image.endswith('.jpg'):
            images.append(Image.open(os.path.join(image_dir, image)))
    return images