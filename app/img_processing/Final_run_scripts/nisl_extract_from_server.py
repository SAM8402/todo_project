import os
from PIL import Image

def resize_images(input_folder, output_folder, target_size=(1333, 1333)):
    """
    Resizes images with '_thumbnail_original.jpg' in their name to a target size
    and saves them to an output folder.

    Args:
        input_folder (str): The path to the folder containing the original images.
        output_folder (str): The path to the folder where resized images will be saved.
        target_size (tuple): A tuple (width, height) for the desired image size.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    processed_count = 0
    print(f"Scanning for images in: {input_folder}")

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is a JPG image and contains the specific string
        if filename.lower().endswith('.jpg') and '_thumbnail_original.jpg' in filename.lower():
            input_filepath = os.path.join(input_folder, filename)
            output_filename = filename.replace('_thumbnail_original.jpg', '_resized.jpg') # Optional: change output filename
            output_filepath = os.path.join(output_folder, output_filename)

            try:
                # Open the image
                with Image.open(input_filepath) as img:
                    print(f"Processing: {filename}")

                    # Resize the image
                    # Using Image.LANCZOS for high-quality downsampling
                    resized_img = img.resize(target_size, Image.LANCZOS)

                    # Save the resized image to the output folder
                    resized_img.save(output_filepath)
                    print(f"Saved resized image to: {output_filepath}")
                    processed_count += 1

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    if processed_count > 0:
        print(f"\nFinished processing. Resized {processed_count} images.")
    else:
        print("\nNo images matching '_thumbnail_original.jpg' were found or processed.")

# --- Configuration ---
# IMPORTANT: Replace these paths with your actual folder paths
INPUT_FOLDER = '/mnt/remote/analytics/146/NISL/'
OUTPUT_FOLDER = '/home/projects/medimg/supriti/brain-registration/146/146_nissl'
TARGET_WIDTH = 1333
TARGET_HEIGHT = 1333

# Run the function
if __name__ == "__main__":
    # Ensure Pillow is installed: pip install Pillow
    resize_images(INPUT_FOLDER, OUTPUT_FOLDER, (TARGET_WIDTH, TARGET_HEIGHT))



