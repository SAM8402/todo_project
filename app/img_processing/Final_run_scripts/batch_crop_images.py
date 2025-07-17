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

def crop_images_in_directory(input_dir, output_dir, background_threshold=240):
    """
    Crop all images in a directory to their content and save to output directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save cropped images
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
    
    print(f"Found {len(image_files)} image files to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Background threshold: {background_threshold}")
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
            print(f"[{i}/{len(image_files)}] {filename}")
            print(f"  Original: {original_size[0]}x{original_size[1]} -> Cropped: {cropped.size[0]}x{cropped.size[1]}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"[{i}/{len(image_files)}] ERROR processing {filename}: {e}")
            error_count += 1
    
    # Summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("-" * 50)
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Successfully processed: {processed_count} images")
    print(f"Errors: {error_count} images")
    print(f"Output saved to: {output_dir}")

def main():
    """Main function to run the batch cropping process."""
    # Configuration
    image_data_dir = "/home/projects/bfi_viewer/app/backend/brainviewer/static/images_data"
    
    # Process the 142 directory
    input_dir = os.path.join(image_data_dir, "142")
    output_dir = os.path.join(image_data_dir, "142_cropped")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    print("=== Batch Image Cropping ===")
    print(f"Processing images from: {input_dir}")
    print(f"Saving cropped images to: {output_dir}")
    
    # Ask for confirmation
    response = input("Do you want to proceed? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Process images
    crop_images_in_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()