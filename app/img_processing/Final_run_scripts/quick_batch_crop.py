import os
from PIL import Image, ImageOps
import time

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
    
    # Find bounding box by scanning pixels more efficiently
    width, height = gray.size
    
    # Find the bounds of non-background content
    left = width
    right = 0
    top = height
    bottom = 0
    
    found_content = False
    
    # More efficient scanning - sample every few pixels for initial bounds
    step = max(1, min(width, height) // 100)  # Sample every 1% of image
    
    for y in range(0, height, step):
        for x in range(0, width, step):
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
    
    # Refine bounds by scanning the edges more precisely
    # Scan left edge
    for x in range(max(0, left - 20), left + 20):
        if x >= width:
            break
        for y in range(top, bottom + 1):
            if y >= height:
                break
            if gray.getpixel((x, y)) < background_threshold:
                left = min(left, x)
                break
    
    # Scan right edge
    for x in range(right - 20, min(width, right + 20)):
        if x < 0:
            continue
        for y in range(top, bottom + 1):
            if y >= height:
                break
            if gray.getpixel((x, y)) < background_threshold:
                right = max(right, x)
                break
    
    # Scan top edge
    for y in range(max(0, top - 20), top + 20):
        if y >= height:
            break
        for x in range(left, right + 1):
            if x >= width:
                break
            if gray.getpixel((x, y)) < background_threshold:
                top = min(top, y)
                break
    
    # Scan bottom edge
    for y in range(bottom - 20, min(height, bottom + 20)):
        if y < 0:
            continue
        for x in range(left, right + 1):
            if x >= width:
                break
            if gray.getpixel((x, y)) < background_threshold:
                bottom = max(bottom, y)
                break
    
    # Add padding (5% of image dimensions)
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
    
    return cropped

def crop_images_in_directory(input_dir, output_dir, max_images=None, background_threshold=240):
    """
    Crop all images in a directory to their content and save to output directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save cropped images
        max_images: Maximum number of images to process (None for all)
        background_threshold: Threshold for background detection (0-255)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    
    # Get list of all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    # Limit number of images if specified
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Processing {len(image_files)} image files")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Process all images in the directory
    processed_count = 0
    error_count = 0
    start_time = time.time()
    
    for i, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Get original size
            original_img = Image.open(input_path)
            original_size = original_img.size
            
            # Crop the image
            cropped = crop_to_content(input_path, output_path, background_threshold)
            
            # Print progress
            print(f"[{i}/{len(image_files)}] {filename} - {original_size[0]}x{original_size[1]} -> {cropped.size[0]}x{cropped.size[1]}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"[{i}/{len(image_files)}] ERROR: {filename} - {e}")
            error_count += 1
    
    # Summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("-" * 50)
    print(f"Completed in {elapsed_time:.2f} seconds")
    print(f"Successfully processed: {processed_count} images")
    print(f"Errors: {error_count} images")

if __name__ == "__main__":
    # Configuration
    image_data_dir = "/home/projects/bfi_viewer/app/backend/brainviewer/static/images_data"
    
    # Process the 142 directory - test with first 10 images
    input_dir = os.path.join(image_data_dir, "142")
    output_dir = os.path.join(image_data_dir, "142_cropped")
    
    print("=== Quick Batch Image Cropping Test ===")
    
    # Test with first 10 images
    crop_images_in_directory(input_dir, output_dir, max_images=10)