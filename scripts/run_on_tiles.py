import os

from PIL import Image
from tqdm import tqdm

from utils.llava_request import llava_request

image_super_folder = "./images/tiled_images"
model_name = "llava:7b-v1.6-mistral-fp16"  # Edit this model name as needed
prompt = """You are a proficient smoke detector at a fire tower. Does the following image contain wildfire smoke? Look carefully, and distinguish between clouds and smoke. Output "yes" only if there is smoke, and "no" only if there is no smoke. Be conservative, so if you are not sure, output "no"."""  # Edit this prompt as needed


def add_border(image, color, border_size):
    width, height = image.size
    bordered_image = Image.new(
        "RGB", (width + border_size * 2, height + border_size * 2), color
    )
    bordered_image.paste(image, (border_size, border_size))
    return bordered_image


def get_tiled_llava(image_folder_name, prompt):
    # Load all tiled images
    tiled_images = []
    image_paths = []
    for filename in sorted(os.listdir(image_folder_name)):
        if filename.endswith(".jpg"):
            image_file_path = os.path.join(image_folder_name, filename)
            image = Image.open(image_file_path)
            tiled_images.append(image)
            image_paths.append(image_file_path)

    results = []
    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        result = llava_request(prompt, model_name, image_path)
        results.append(result)

    # Add border based on LLAVA results
    bordered_images = []
    print(results)
    for i, result in enumerate(results):
        if result.lower() == "yes" or result.lower() == " yes ":
            bordered_image = add_border(tiled_images[i], "red", 5)
        else:
            bordered_image = add_border(tiled_images[i], "white", 5)
        bordered_images.append(bordered_image)

    # Stitch images back together
    num_images_side = int(len(tiled_images) ** 0.5)
    total_width = bordered_images[0].width * num_images_side
    total_height = bordered_images[0].height * num_images_side
    stitched_image = Image.new("RGB", (total_width, total_height))

    for i, bordered_image in enumerate(bordered_images):
        x = (i % num_images_side) * bordered_image.width
        y = (i // num_images_side) * bordered_image.height
        stitched_image.paste(bordered_image, (x, y))
    return results, stitched_image


if __name__ == "__main__":
    for folder in tqdm(sorted(os.listdir(image_super_folder)), mininterval=0.1):
        if not os.path.isdir(os.path.join(image_super_folder, folder)):
            continue
        # print(f"Processing folder: {folder}")

        image_folder_name = os.path.join(image_super_folder, folder)
        save_image_path = os.path.join("images", f"{image_folder_name}_stitched.jpg")

        # Skip if stitched image already exists
        if os.path.exists(save_image_path):
            print(f"Skipping folder {folder} as stitched image already exists.")
            continue

        results, stitched_image = get_tiled_llava(image_folder_name, prompt)

        # Save stitched image
        if not os.path.exists(save_image_path):
            os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
        stitched_image.save(save_image_path)

        # Save results
        if not os.path.exists("results"):
            os.makedirs("results")

        with open(f"results/{folder}_results.txt", "w") as f:
            for result in results:
                f.write(result + "\n")
