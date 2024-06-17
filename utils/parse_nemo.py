import json
import os
from typing import List, Tuple

from PIL import Image, ImageDraw


def read_json_files(folder_path: str) -> List[dict]:
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    data = []
    for file in json_files:
        with open(os.path.join(folder_path, file), "r") as f:
            data.append(json.load(f))
    return data


def get_bounding_boxes(objects: List[dict]) -> List[Tuple[int, int, int, int]]:
    bounding_boxes = []
    for obj in objects:
        points = obj["points"]["exterior"]
        if len(points) == 2:
            x_min, y_min = points[0]
            x_max, y_max = points[1]
            bounding_boxes.append((x_min, y_min, x_max, y_max))
    return bounding_boxes


def calculate_union_bounding_box(
    bounding_boxes: List[Tuple[int, int, int, int]]
) -> Tuple[int, int, int, int]:
    if not bounding_boxes:
        return None
    x_min = min(box[0] for box in bounding_boxes)
    y_min = min(box[1] for box in bounding_boxes)
    x_max = max(box[2] for box in bounding_boxes)
    y_max = max(box[3] for box in bounding_boxes)
    return (x_min, y_min, x_max, y_max)


def overlay_bbox(image_path: str, bbox: Tuple[int, int, int, int], output_path: str):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline="red", width=3)
    image.save(output_path)


def process_images_and_annotations(
    img_folder: str, ann_folder: str, output_folder: str
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img_files = [
        f for f in os.listdir(img_folder) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    for img_file in img_files:
        img_path = os.path.join(img_folder, img_file)
        json_file = img_file.rsplit(".", 1)[0] + ".json"
        json_path = os.path.join(ann_folder, json_file)
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                objects = data["objects"]
                bounding_boxes = get_bounding_boxes(objects)
                union_box = calculate_union_bounding_box(bounding_boxes)

                if union_box:
                    output_path = os.path.join(output_folder, img_file)
                    overlay_bbox(img_path, union_box, output_path)


# Paths to the folders
img_folder = "nemo/val/img"
ann_folder = "nemo/val/ann"
output_folder = "test/stitched"

# Process images and annotations
process_images_and_annotations(img_folder, ann_folder, output_folder)
