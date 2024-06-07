import cv2
import numpy as np

def frame_difference(image1_path, image2_path, output_path):
    # Load the two images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    # Check if images were loaded successfully
    if image1 is None:
        print(f"Error: Could not load image at {image1_path}")
        return
    if image2 is None:
        print(f"Error: Could not load image at {image2_path}")
        return
    
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Compute the absolute difference between the two images
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply a binary threshold to get a binary image
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Save the frame difference image
    cv2.imwrite(output_path, diff_thresh)
    print(f"Frame difference image saved at {output_path}")
    
    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage'
path1 = "frames/1569962296_-00060.jpg"
path2 = "frames/1569962476_+00120.jpg"
output_path = "test/frame_difference.jpg"
frame_difference(path1, path2, output_path)
