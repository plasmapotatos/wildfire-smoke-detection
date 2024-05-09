import os
import xml.etree.ElementTree as ET
import cv2
from PIL import Image

# Function to parse XML files and extract bounding box coordinates
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find('filename').text
    folder = root.find('folder').text
    path = root.find('path').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    boxes = []
    for obj in root.findall('object'):
        box = obj.find('bndbox')
        xmin = int(float(box.find('xmin').text))
        ymin = int(float(box.find('ymin').text))
        xmax = int(float(box.find('xmax').text))
        ymax = int(float(box.find('ymax').text))
        boxes.append((xmin, ymin, xmax, ymax))
    return path, boxes

# Function to draw bounding boxes on image
def draw_boxes(image, boxes):
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

def crop_objects(image_path, objects):
    image = Image.open(image_path)
    cropped_images = []
    for i, obj in enumerate(objects):
        cropped_image = image.crop(obj)
        cropped_images.append(cropped_image)
    return cropped_images

# Directory containing images and corresponding XML files
image_dir = './day_time_wildfire_v2/images'
xml_dir = './day_time_wildfire_v2/annotations/xmls'
output_dir = './day_time_wildfire_v2/cropped_images'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each XML file and save annotated image
for xml_file in os.listdir(xml_dir):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(xml_dir, xml_file)
        image_file = os.path.splitext(xml_file)[0] + '.jpeg'  # Assuming JPEG images
        image_path = os.path.join(image_dir, image_file)
        if os.path.exists(image_path):
            boxes = parse_xml(xml_path)[1]
            image = cv2.imread(image_path)
            draw_boxes(image, boxes)
            output_path = os.path.join(output_dir, image_file)
            cropped_images = crop_objects(image_path, boxes)
            for image in cropped_images:
                image.save(output_path)
    
# if __name__ == "__main__":
#     xml_file = "./day_time_wildfire_v2/annotations/xmls/ckagz7s5solbc0841r1aklq1g.xml"
#     image_path, objects = parse_xml(xml_file)
#     cropped_images = crop_objects(image_path, objects)
#     print(f"Cropped {len(cropped_images)} bounding box images.")

