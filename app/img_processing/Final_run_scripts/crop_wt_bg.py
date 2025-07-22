import os
import numpy as np
from PIL import Image, ImageOps


def crop_to_content(image_path, output_path=None, background_threshold=240, edge_detection=True):
    """
    Crop an image to its content by removing background areas.
    Uses edge detection and morphological operations for better results.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the cropped image (optional)
        background_threshold: Threshold for background detection (0-255)
        edge_detection: Use edge detection for better boundary detection
    
    Returns:
        Cropped image as PIL Image object
    """
    # Load image
    try:
        img = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Could not load image: {image_path} - {e}")

    width, height = img.size
    
    # For large images, use downsampling to find bounds efficiently
    max_size = 800  # Reduced for better performance
    if max(width, height) > max_size:
        # Calculate scale factor
        scale_factor = max_size / max(width, height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Downsample for analysis
        small_img = img.resize((new_width, new_height), Image.LANCZOS)
        gray = ImageOps.grayscale(small_img)
        gray_array = np.array(gray)
        
    else:
        # For smaller images, process directly
        gray = ImageOps.grayscale(img)
        gray_array = np.array(gray)
        scale_factor = 1

    # Use adaptive threshold based on image statistics - more conservative
    mean_val = np.mean(gray_array)
    if mean_val > 200:  # Bright image
        adaptive_threshold = mean_val - 15  # Reduced from 30 to 15
    else:
        adaptive_threshold = min(background_threshold, mean_val + 25)  # Reduced from 50 to 25
    
    print(f"Using threshold: {adaptive_threshold:.1f} (mean: {mean_val:.1f})")
    
    # Create content mask
    content_mask = gray_array < adaptive_threshold
    
    # Remove small noise using morphological operations
    from scipy import ndimage
    # Remove small isolated pixels
    content_mask = ndimage.binary_opening(content_mask, structure=np.ones((3,3)))
    # Fill small holes
    content_mask = ndimage.binary_closing(content_mask, structure=np.ones((5,5)))
    
    if not np.any(content_mask):
        print(f"Warning: No content found in {image_path}")
        return img
    
    # Find content regions and get the largest connected component
    labeled_array, num_features = ndimage.label(content_mask)
    if num_features > 1:
        # Keep only the largest connected component
        sizes = ndimage.sum(content_mask, labeled_array, range(num_features + 1))
        largest_component = np.argmax(sizes[1:]) + 1  # Skip background (0)
        content_mask = labeled_array == largest_component
    
    # Find bounding box
    content_coords = np.where(content_mask)
    if len(content_coords[0]) == 0:
        print(f"Warning: No content found after filtering in {image_path}")
        return img
        
    top_small = int(np.min(content_coords[0]))
    bottom_small = int(np.max(content_coords[0]))
    left_small = int(np.min(content_coords[1]))
    right_small = int(np.max(content_coords[1]))
    
    # Scale back to original image coordinates if needed
    if scale_factor != 1:
        top = int(top_small / scale_factor)
        bottom = int(bottom_small / scale_factor)
        left = int(left_small / scale_factor)
        right = int(right_small / scale_factor)
    else:
        top, bottom, left, right = top_small, bottom_small, left_small, right_small

    # Add padding (increased to prevent over-cropping)
    padding_y = max(15, int(0.03 * height))
    padding_x = max(15, int(0.03 * width))

    # Apply padding while staying within image bounds
    left = max(0, left - padding_x)
    right = min(width, right + padding_x)
    top = max(0, top - padding_y)
    bottom = min(height, bottom + padding_y)

    print(f"Crop bounds: ({left}, {top}) to ({right}, {bottom})")
    print(f"Original size: {width}x{height}, Cropped size: {right-left}x{bottom-top}")

    # Crop the original image
    cropped = img.crop((left, top, right, bottom))

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cropped.save(output_path)
        print(f"Cropped image saved to: {output_path}")

    return cropped


def crop_images_in_directory(input_dir, output_dir=None, background_threshold=240):
    """
    Crop all images in a directory to their content.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save cropped images (optional)
        background_threshold: Threshold for background detection (0-255)
    """
    if output_dir is None:
        output_dir = input_dir + "_cropped"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Supported image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

    # Process all images in the directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                crop_to_content(input_path, output_path, background_threshold)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Example usage
if __name__ == "__main__":
    # Path to the image data directory
    image_data_dir = "/home/projects/bfi_viewer/app/backend/brainviewer/static/images_data"

    # Test with a single image first
    # test_image = os.path.join(image_data_dir, "142", "bfi-919.png")
    # test_output = os.path.join(image_data_dir, "142_cropped", "bfi-919_cropped.png")

    # print("Testing with a single image...")
    # try:
    #     cropped = crop_to_content(test_image, test_output)
    #     print(
    #         f"Successfully cropped test image. Original size: {Image.open(test_image).size}, Cropped size: {cropped.size}")
    # except Exception as e:
    #     print(f"Error: {e}")

    # Process entire directory - NOW ENABLED
    input_dir = os.path.join(image_data_dir, "142")
    output_dir = os.path.join(image_data_dir, "cropped")
    crop_images_in_directory(input_dir, output_dir)

    print("Image cropping completed for all images!")
