import os
from PIL import Image, ImageOps

def crop_to_content(image_path, output_path=None, background_threshold=240):
    """
    Crop an image to its content by removing background areas.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the cropped image (optional)
        background_threshold: Threshold for background detection (0-255)
    
    Returns:
        Cropped image as PIL Image object
    """
    # Load image
    try:
        img = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Could not load image: {image_path} - {e}")
    
    # Convert to grayscale for easier processing
    gray = ImageOps.grayscale(img)
    
    # Find bounding box by scanning pixels
    width, height = gray.size
    
    # Find the bounds of non-background content
    left = width
    right = 0
    top = height
    bottom = 0
    
    found_content = False
    
    # Scan all pixels to find content bounds
    for y in range(height):
        for x in range(width):
            pixel = gray.getpixel((x, y))
            if pixel < background_threshold:
                found_content = True
                left = min(left, x)
                right = max(right, x)
                top = min(top, y)
                bottom = max(bottom, y)
    
    if not found_content:
        print(f"Warning: No content found in {image_path}")
        return img
    
    # Add small padding (5% of image dimensions)
    padding_y = max(5, int(0.05 * height))
    padding_x = max(5, int(0.05 * width))
    
    # Apply padding while staying within image bounds
    left = max(0, left - padding_x)
    right = min(width, right + padding_x)
    top = max(0, top - padding_y)
    bottom = min(height, bottom + padding_y)
    
    # Crop the image using PIL
    cropped = img.crop((left, top, right, bottom))
    
    # Save if output path provided
    if output_path:
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
    test_image = os.path.join(image_data_dir, "142", "bfi-1.png")
    test_output = os.path.join(image_data_dir, "142", "bfi-1_cropped.png")
    
    print("Testing with a single image...")
    try:
        cropped = crop_to_content(test_image, test_output)
        print(f"Successfully cropped test image. Original size: {Image.open(test_image).size}, Cropped size: {cropped.size}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Uncomment below to process entire directory
    # input_dir = os.path.join(image_data_dir, "142")
    # output_dir = os.path.join(image_data_dir, "142_cropped")
    # crop_images_in_directory(input_dir, output_dir)
    
    print("Image cropping test completed!")