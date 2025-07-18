#!/usr/bin/env python3
"""
Batch Image Cropping Script
============================

This script crops all images in a directory to their content by removing
background areas and saves the cropped images to a separate output directory.

Usage:
    python crop_all_images.py

Features:
- Automatically detects and removes white/light backgrounds
- Preserves image content with appropriate padding
- Efficient processing with progress tracking
- Comprehensive error handling
- Configurable background threshold
"""

import os
from PIL import Image, ImageOps
import time
import argparse

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
    width, height = gray.size
    
    # Find the bounds of non-background content using efficient sampling
    left = width
    right = 0
    top = height
    bottom = 0
    found_content = False
    
    # Sample every few pixels for initial bounds detection
    step = max(1, min(width, height) // 200)  # Sample every 0.5% of image
    
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
    
    # Refine bounds by scanning edges more precisely
    edge_margin = 50  # pixels to check around rough bounds
    
    # Refine left edge
    for x in range(max(0, left - edge_margin), min(width, left + edge_margin)):
        for y in range(top, bottom + 1, step):
            if y >= height:
                break
            if gray.getpixel((x, y)) < background_threshold:
                left = min(left, x)
                break
    
    # Refine right edge
    for x in range(max(0, right - edge_margin), min(width, right + edge_margin)):
        for y in range(top, bottom + 1, step):
            if y >= height:
                break
            if gray.getpixel((x, y)) < background_threshold:
                right = max(right, x)
                break
    
    # Refine top edge
    for y in range(max(0, top - edge_margin), min(height, top + edge_margin)):
        for x in range(left, right + 1, step):
            if x >= width:
                break
            if gray.getpixel((x, y)) < background_threshold:
                top = min(top, y)
                break
    
    # Refine bottom edge
    for y in range(max(0, bottom - edge_margin), min(height, bottom + edge_margin)):
        for x in range(left, right + 1, step):
            if x >= width:
                break
            if gray.getpixel((x, y)) < background_threshold:
                bottom = max(bottom, y)
                break
    
    # Add padding (3% of image dimensions)
    padding_y = max(10, int(0.03 * height))
    padding_x = max(10, int(0.03 * width))
    
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
    image_files.sort()  # Sort for consistent processing order
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image files to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Background threshold: {background_threshold}")
    print("=" * 60)
    
    # Process all images in the directory
    processed_count = 0
    error_count = 0
    total_original_size = 0
    total_cropped_size = 0
    start_time = time.time()
    
    for i, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Get original size
            original_img = Image.open(input_path)
            original_size = original_img.size
            original_pixels = original_size[0] * original_size[1]
            
            # Crop the image
            cropped = crop_to_content(input_path, output_path, background_threshold)
            cropped_pixels = cropped.size[0] * cropped.size[1]
            
            # Calculate compression ratio
            compression_ratio = (1 - cropped_pixels / original_pixels) * 100
            
            # Print progress
            if i % 100 == 0 or i <= 10:  # Print every 100th image or first 10
                print(f"[{i:4d}/{len(image_files)}] {filename}")
                print(f"    Size: {original_size[0]}x{original_size[1]} -> {cropped.size[0]}x{cropped.size[1]} ({compression_ratio:.1f}% reduction)")
            
            processed_count += 1
            total_original_size += original_pixels
            total_cropped_size += cropped_pixels
            
        except Exception as e:
            print(f"[{i:4d}/{len(image_files)}] ERROR: {filename} - {e}")
            error_count += 1
    
    # Summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    overall_compression = (1 - total_cropped_size / total_original_size) * 100 if total_original_size > 0 else 0
    
    print("=" * 60)
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Successfully processed: {processed_count} images")
    print(f"Errors: {error_count} images")
    print(f"Overall size reduction: {overall_compression:.1f}%")
    print(f"Average time per image: {elapsed_time/len(image_files):.3f} seconds")
    print(f"Output saved to: {output_dir}")

def main():
    """Main function to run the batch cropping process."""
    parser = argparse.ArgumentParser(description='Batch crop images to content')
    parser.add_argument('--input', '-i', default=None, help='Input directory path')
    parser.add_argument('--output', '-o', default=None, help='Output directory path')
    parser.add_argument('--threshold', '-t', type=int, default=240, help='Background threshold (0-255)')
    
    args = parser.parse_args()
    
    # Default configuration
    if args.input is None:
        image_data_dir = "/home/projects/bfi_viewer/app/backend/brainviewer/static/images_data"
        input_dir = os.path.join(image_data_dir, "142")
    else:
        input_dir = args.input
    
    if args.output is None:
        output_dir = input_dir + "_cropped"
    else:
        output_dir = args.output
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    print("=== Batch Image Cropping ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Background threshold: {args.threshold}")
    
    # Process images
    crop_images_in_directory(input_dir, output_dir, args.threshold)
    
    print("\n✓ Batch cropping completed successfully!")

if __name__ == "__main__":
    main()