import os
import shutil

def copy_folders(list_file, source_dir, output_dir):
    # Read the folder names from the txt file
    with open(list_file, 'r') as file:
        folder_names = file.read().splitlines()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through the folder names and copy them
    for folder_name in folder_names:
        source_path = os.path.join(source_dir, folder_name)
        destination_path = os.path.join(output_dir, folder_name)
        
        if os.path.exists(source_path) and os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path)
            print(f"Copied: {folder_name}")
        else:
            print(f"Folder not found: {folder_name}")

# Parameters
list_file = 'splits/test.txt'    # Path to your text file
source_dir = 'raw_data'   # Path to your source directory containing folders
output_dir = 'splits'   # Path to your output directory

# Run the function
copy_folders(list_file, source_dir, output_dir)
