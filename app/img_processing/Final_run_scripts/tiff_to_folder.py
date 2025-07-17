#!/usr/bin/env python3
import argparse
import re
import os
from pathlib import Path
import tifffile
from PIL import Image

def sanitize(name: str, max_length=200) -> str:
    """
    Turn any filesystem-unsafe characters into underscores,
    but preserve letters, digits, dash, dot and underscore.
    Also limit filename length for Windows compatibility.
    """
    safe_name = re.sub(r'[^\w\-.]', '_', name.strip())
    return safe_name[:max_length] if len(safe_name) > max_length else safe_name

def extract_se_number(text: str) -> str | None:
    """
    Extracts the number following 'SE_' from a given text.
    Returns the extracted number as a string, or None if not found.
    """
    match = re.search(r'SE_(\d+)', text)
    if match:
        return match.group(1)
    return None

def split_tiff(input_path, output_dir="nissl_slices_149", out_fmt="jpg"):
    """
    Splits a multi-page TIFF into individual image files,
    naming each file primarily by the number after 'SE_' found in its label,
    or a fallback.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(input_path) as tif:
        labels = None
        if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata and 'Labels' in tif.imagej_metadata:
            labels = tif.imagej_metadata['Labels']
            print(f"Found {len(labels)} ImageJ slice labels")
        
        for i, page in enumerate(tif.pages, start=1):
            try:
                frame_data = page.asarray()
                
                # Default name if no SE_ number is found
                file_base_name = f"slice_{i:03d}"

                # Try to get the descriptive label from ImageJ metadata
                desc = None
                if labels and i <= len(labels):
                    desc = labels[i-1]
                
                # Attempt to extract SE_ number from the description/label
                if desc:
                    extracted_se_number = extract_se_number(desc)
                    if extracted_se_number:
                        file_base_name = extracted_se_number
                    else:
                        # If label exists but no SE_ number, use a sanitized version of the label
                        file_base_name = sanitize(desc)
                else:
                    # If no ImageJ label, try extracting from the input TIFF filename as a last resort
                    # (though if individual slice labels have names, they're usually preferred)
                    base_file_se_number = extract_se_number(input_path.name)
                    if base_file_se_number:
                        file_base_name = f"{base_file_se_number}_slice_{i:03d}"
                    # If still no specific name, the default 'slice_00X' will be used.

                # Construct the output filename
                name_for_file = file_base_name # This will be just the SE_ number or a descriptive name
                out_file = output_dir / f"{name_for_file}.{out_fmt}"
                
                # Avoid overwriting in case of duplicate names
                counter = 1
                while out_file.exists():
                    out_file = output_dir / f"{name_for_file}_{counter:02d}.{out_fmt}"
                    counter += 1
                
                pil_image = Image.fromarray(frame_data)
                
                if pil_image.mode not in ("L", "RGB"):
                    pil_image = pil_image.convert("RGB")
                
                pil_image.save(str(out_file))
                
                print(f"[{i:03d}] → {out_file.name}")
                
                del pil_image
                del frame_data
                
            except Exception as e:
                print(f"Error processing slice {i}: {str(e)}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split an ImageJ stack TIFF into files named by the 'SE_' number from each slice's label."
    )
    parser.add_argument(
        "tif_path",
        default="B_142_FB34_nisl_stack_output.tif",
        help="Path to your multi-page TIFF (e.g. B_142_FB34_nisl_stack_output.tif)"
    )
    parser.add_argument(
        "-o", "--outdir", 
        default="nissl_slices_142",
        help="Directory to save the individual images into."
    )
    parser.add_argument(
        "-f", "--format", 
        default="jpg", choices=["png", "jpg", "tif", "bmp"],
        help="Output image format."
    )
    args = parser.parse_args()
    
    try:
        split_tiff(args.tif_path, args.outdir, args.format)
    except Exception as e:
        print(f"Error: {str(e)}")