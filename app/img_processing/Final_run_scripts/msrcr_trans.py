import os
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
import re
from tqdm import tqdm
import logging
import traceback

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper function to extract index from diverse filenames ---
def extract_index_from_filename(filename):
    base_name = os.path.splitext(filename)[0]
    try:
        return int(base_name)
    except ValueError:
        pass
    match_original = re.search(r'_(\d+)_original$', base_name, re.IGNORECASE)
    if match_original:
        try:
            return int(match_original.group(1))
        except ValueError:
            pass
    match_b = re.search(r'_B_(\d+)', base_name, re.IGNORECASE)
    if match_b:
        try:
            return int(match_b.group(1))
        except ValueError:
            pass
    all_digits = re.findall(r'\d+', base_name)
    if all_digits:
        try:
            return int(all_digits[-1])
        except ValueError:
            pass
    return None

# --- Helper functions for image processing ---

def adjust_gamma(image, gamma=1.0):
    """
    Applies gamma correction to an image.
    gamma < 1.0 brightens, gamma > 1.0 darkens.
    """
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table).squeeze() 

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Applies unsharp mask for sharpening."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, 0)
    sharpened = np.minimum(sharpened, 255)
    sharpened = sharpened.astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

# --- Main image processing function for a single image ---
def process_single_image(args):
    """
    Processes a single image using the specified enhancement parameters.
    Expects an original image path, its corresponding mask path, an output path,
    a debug folder path, and a dictionary of configuration parameters.
    """
    input_image_path, input_mask_path, output_path, config = args
    
    try:
        # 1. Load Image and Mask
        img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logging.error(f"Failed to read input image: {input_image_path}. Skipping.")
            return False

        # Ensure input image is in BGR format for consistent processing
        if img.shape[2] == 4: # If it has an alpha channel (PNG), convert to BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif len(img.shape) == 2: # If it's grayscale, convert to BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else: # Already BGR
            img_bgr = img 
            
        #debug_filename_base = os.path.splitext(os.path.basename(input_image_path))[0]
        #current_image_debug_folder = os.path.join(debug_output_folder, debug_filename_base)
        #os.makedirs(current_image_debug_folder, exist_ok=True)

        #cv2.imwrite(os.path.join(current_image_debug_folder, f"{debug_filename_base}_step1_bgr_input.png"), img_bgr)


        # --- CRITICAL MASK HANDLING ---
        mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logging.error(f"Failed to read mask image: {input_mask_path}. Skipping.")
            return False
        
        # Log min/max values of the raw loaded mask
        min_mask_val = np.min(mask)
        max_mask_val = np.max(mask)
        logging.info(f"Mask '{os.path.basename(input_mask_path)}' raw min/max: {min_mask_val}/{max_mask_val}")

        # Always generate both binary and inverted binary masks for comprehensive debugging
        _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        _, binary_mask_inv = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY_INV)
        
        #cv2.imwrite(os.path.join(current_image_debug_folder, f"{debug_filename_base}_step1a_mask_used_binary.png"), binary_mask)
        #cv2.imwrite(os.path.join(current_image_debug_folder, f"{debug_filename_base}_step1a_mask_used_binary_inv.png"), binary_mask_inv)

        # *** IMPORTANT: Based on your mask 'B_142-ST_BFIW-SE_1012_original_Simple Segmentation.jpg',
        #     which has BLACK foreground on WHITE background, we must use binary_mask_inv. ***
        #     If your masks change (e.g., white foreground on black background), you would switch to binary_mask.
        foreground_mask = (binary_mask_inv > 0) # This should correctly select your black specimen as foreground
        
        # Add a check for an empty or full mask
        foreground_pixel_count = np.sum(foreground_mask)
        if foreground_pixel_count == 0:
            logging.error(f"Mask '{os.path.basename(input_mask_path)}' resulted in an empty foreground. Skipping image.")
            return False
        if foreground_pixel_count == foreground_mask.size:
            logging.warning(f"Mask '{os.path.basename(input_mask_path)}' resulted in a full foreground (entire image). This might indicate a mask issue if the image is mostly background.")


        # 2. Isolate Foreground for Processing
        # Create a temporary BGR image with only the foreground for processing steps
        processed_bgr_foreground_isolated = np.zeros_like(img_bgr, dtype=np.uint8)
        processed_bgr_foreground_isolated[foreground_mask] = img_bgr[foreground_mask]
        #cv2.imwrite(os.path.join(current_image_debug_folder, f"{debug_filename_base}_step2_foreground_isolated_bgr.png"), processed_bgr_foreground_isolated)


        # 3. Apply Image Adjustments to the FOREGROUND ONLY (using LAB color space)
        # Convert the isolated foreground to LAB for processing
        lab_img_fg = cv2.cvtColor(processed_bgr_foreground_isolated, cv2.COLOR_BGR2LAB)
        L_fg, A_fg, B_fg = cv2.split(lab_img_fg)

        # Create copies to apply modifications only to foreground pixels
        L_processed = L_fg.copy().astype(np.float32) 
        A_processed = A_fg.copy().astype(np.float32)
        B_processed = B_fg.copy().astype(np.float32)

        # --- Get Enhancement Parameters from the config dictionary ---
        GAMMA_VALUE = config['GAMMA_VALUE']
        CLAHE_CLIP_LIMIT = config['CLAHE_CLIP_LIMIT']
        CLAHE_TILE_GRID_SIZE = config['CLAHE_TILE_GRID_SIZE']
        TARGET_LAB_A_CV = config['TARGET_LAB_A_CV']
        TARGET_LAB_B_CV = config['TARGET_LAB_B_CV']
        COLOR_TONE_BLEND_FACTOR = config['COLOR_TONE_BLEND_FACTOR']
        SHARPEN_KERNEL_SIZE = config['SHARPEN_KERNEL_SIZE']
        SHARPEN_SIGMA = config['SHARPEN_SIGMA']
        SHARPEN_AMOUNT = config['SHARPEN_AMOUNT']

        # 3a. Overall Luminosity Adjustment (Gamma Correction and Normalization)
        # Apply gamma only to foreground pixels' L channel
        L_gamma_corrected_fg_pixels = adjust_gamma(L_fg[foreground_mask], GAMMA_VALUE) 
        
        # Normalize the gamma-corrected L channel to the full 0-255 range for maximum brightness
        L_stretched_fg_pixels = np.zeros_like(L_gamma_corrected_fg_pixels, dtype=np.uint8)
        cv2.normalize(L_gamma_corrected_fg_pixels, L_stretched_fg_pixels, 0, 255, cv2.NORM_MINMAX)
        
        # Assign back to the L_processed array
        L_processed[foreground_mask] = L_stretched_fg_pixels.ravel()
        #cv2.imwrite(os.path.join(current_image_debug_folder, f"{debug_filename_base}_step3a_L_gamma_stretched.png"), L_processed.astype(np.uint8))


        # 3b. Adaptive Contrast Enhancement (CLAHE for texture detail)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        # Apply CLAHE to the L_processed (which now contains gamma-corrected and stretched L values)
        L_clahe_applied_fg_pixels = clahe.apply(L_processed[foreground_mask].astype(np.uint8))
        
        # Assign back to the L_processed array
        L_processed[foreground_mask] = L_clahe_applied_fg_pixels.ravel()
        #cv2.imwrite(os.path.join(current_image_debug_folder, f"{debug_filename_base}_step3b_L_clahe_applied.png"), L_processed.astype(np.uint8))


        # 3c. Color Tone Adjustment (Blending A and B channels)
        A_final_fg_pixels = (A_fg[foreground_mask].astype(np.float32) * (1 - COLOR_TONE_BLEND_FACTOR)) + (TARGET_LAB_A_CV * COLOR_TONE_BLEND_FACTOR)
        B_final_fg_pixels = (B_fg[foreground_mask].astype(np.float32) * (1 - COLOR_TONE_BLEND_FACTOR)) + (TARGET_LAB_B_CV * COLOR_TONE_BLEND_FACTOR)
        
        # Clip values and assign back to the A_processed and B_processed arrays
        A_processed[foreground_mask] = np.clip(A_final_fg_pixels, 0, 255).astype(np.uint8)
        B_processed[foreground_mask] = np.clip(B_final_fg_pixels, 0, 255).astype(np.uint8)

        # Merge the processed LAB channels back into a BGR image
        processed_lab_img = cv2.merge([L_processed.astype(np.uint8), A_processed.astype(np.uint8), B_processed.astype(np.uint8)])
        processed_bgr_img = cv2.cvtColor(processed_lab_img, cv2.COLOR_LAB2BGR)
        
        #cv2.imwrite(os.path.join(current_image_debug_folder, f"{debug_filename_base}_step3c_color_luminosity_adjusted.png"), processed_bgr_img)

        # 4. Sharpening
        # Apply sharpening to the already processed BGR foreground
        final_processed_img_sharpened = unsharp_mask(processed_bgr_img, 
                                                kernel_size=SHARPEN_KERNEL_SIZE, 
                                                sigma=SHARPEN_SIGMA, 
                                                amount=SHARPEN_AMOUNT, 
                                                threshold=0) 

        #cv2.imwrite(os.path.join(current_image_debug_folder, f"{debug_filename_base}_step4_sharpened_foreground.png"), final_processed_img_sharpened)

        # 5. Prepare RGBA output for transparent background
        # Create an alpha channel: 255 for foreground, 0 for background
        alpha_channel = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        alpha_channel[foreground_mask] = 255 

        # Merge the sharpened BGR image with the alpha channel
        final_output_rgba = cv2.merge([final_processed_img_sharpened[:,:,0], 
                                        final_processed_img_sharpened[:,:,1], 
                                        final_processed_img_sharpened[:,:,2], 
                                        alpha_channel])
        
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # Save as PNG to preserve transparency
        # Ensure the output filename has a .png extension
        output_filename_png = os.path.splitext(output_path)[0] + ".png"
        cv2.imwrite(output_filename_png, final_output_rgba)
        
        # Save a debug version with transparency for verification
        #cv2.imwrite(os.path.join(current_image_debug_folder, f"{debug_filename_base}_final_transparent_bg.png"), final_output_rgba)

        logging.info(f"Successfully processed: {input_image_path} -> {output_filename_png}")
        return True
    except Exception as e:
        logging.error(f"Error processing {input_image_path}: {e}")
        traceback.print_exc()
        return False

# --- Function to collect all image and mask paths and set up tasks for multiprocessing ---
def collect_processing_tasks(image_folder, mask_folder, output_folder, config):
    tasks = []
    
    if not os.path.exists(image_folder):
        logging.error(f"Image directory '{image_folder}' does not exist.")
        return []

    if not os.path.exists(mask_folder):
        logging.error(f"Mask directory '{mask_folder}' does not exist.")
        return []

    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.jp2'))] # Added .jp2
    
    logging.info(f"Found {len(image_files)} image files in '{image_folder}'.")

    for img_filename in sorted(image_files):
        img_path = os.path.join(image_folder, img_filename)
        img_base_name_no_ext = os.path.splitext(img_filename)[0]

        mask_filename = None
        
        # Attempt 1: Mask has the exact same base name as the image (e.g., image.jpg -> image.png)
        if os.path.exists(os.path.join(mask_folder, img_base_name_no_ext + ".png")):
            mask_filename = img_base_name_no_ext + ".png"
        elif os.path.exists(os.path.join(mask_folder, img_base_name_no_ext + ".jpg")):
            mask_filename = img_base_name_no_ext + ".jpg"
        
        # Attempt 2: Mask name contains specific keywords (e.g., "image_mask.png", "image_seg.jpg")
        # Added a specific check for "Simple Segmentation" pattern.
        if not mask_filename:
            for m_file in os.listdir(mask_folder):
                m_base_name_no_ext = os.path.splitext(m_file)[0]
                # Check for direct match or pattern like 'image_base_name_Simple Segmentation.jpg'
                if (m_base_name_no_ext == img_base_name_no_ext + "_Simple Segmentation" or
                    img_base_name_no_ext in m_base_name_no_ext and 
                    ("mask" in m_base_name_no_ext.lower() or 
                     "seg" in m_base_name_no_ext.lower())):
                    if m_base_name_no_ext.startswith(img_base_name_no_ext):
                        mask_filename = m_file
                        break
        
        if mask_filename:
            mask_path = os.path.join(mask_folder, mask_filename)
            output_file_path = os.path.join(output_folder, img_filename) # Still use original filename, but it will be saved as PNG
            tasks.append((img_path, mask_path, output_file_path, config))
        else:
            logging.warning(f"No corresponding mask found for image: {img_filename}. Skipping this image.")
    
    return tasks

# --- Main execution block ---
if __name__ == "__main__":
    # --- 1. Configuration Paths (!!! YOU MUST UPDATE THESE !!!) ---
    INPUT_IMAGE_FOLDER = "/home/projects/medimg/gnikhil/bg_remove/inference_results/142/" # Example: "/path/to/your/image/folder"
    INPUT_MASK_FOLDER = "/home/projects/medimg/gnikhil/bg_remove/inference_results/142/"  # Example: "/path/to/your/mask/folder"
    OUTPUT_FOLDER = "/home/projects/medimg/supriti/brain-registration/142/142_BFI_trans/" # Updated output folder for transparent background
    #DEBUG_OUTPUT_FOLDER = "/home/projects/medimg/supriti/149_gemini_debug_v23/" # Updated debug folder for transparent background

    # --- 2. Define Enhancement Parameters ---
    
    # Target RGB for a specific BFIW-like warm beige/yellow
    # You will need to carefully select this RGB based on your target BFIW images.
    # I've picked one that should lean towards a good yellowish-brown.
    TARGET_COLOR_RGB = (220, 200, 150) # (R, G, B) - for a warm, yellowish-brown/beige tone

    # Convert the target RGB color to LAB for precise adjustment
    target_color_bgr = (TARGET_COLOR_RGB[2], TARGET_COLOR_RGB[1], TARGET_COLOR_RGB[0]) 
    dummy_img = np.array([[target_color_bgr]], dtype=np.uint8)
    TARGET_LAB_A_CV = cv2.cvtColor(dummy_img, cv2.COLOR_BGR2LAB)[0,0][1] # Get 'a' channel
    TARGET_LAB_B_CV = cv2.cvtColor(dummy_img, cv2.COLOR_BGR2LAB)[0,0][2] # Get 'b' channel

    processing_config = {
        'GAMMA_VALUE': 0.35,             # Controls overall brightness/contrast curve. Lower for brighter.
        'CLAHE_CLIP_LIMIT': 2.0,         # Controls local contrast enhancement. Lower for softer, higher for sharper.
        'CLAHE_TILE_GRID_SIZE': (8, 8),  # Size of grid for CLAHE. Smaller for finer detail, larger for broader.
        
        'TARGET_LAB_A_CV': TARGET_LAB_A_CV, # Calculated target 'a' for the desired yellow/beige tone
        'TARGET_LAB_B_CV': TARGET_LAB_B_CV, # Calculated target 'b' for the desired yellow/beige tone
        
        'COLOR_TONE_BLEND_FACTOR': 1.0, # 1.0 means full replacement of A/B channels with target color.
                                        # Reduce to blend with original color.
        
        'SHARPEN_KERNEL_SIZE': (3, 3),  # Smaller kernel for subtle detail sharpening
        'SHARPEN_SIGMA': 0.5,           # Small sigma for less aggressive blurring in unsharp mask
        'SHARPEN_AMOUNT': 1.2           # Strength of sharpening
    }
    
    # --- 3. Create Output Directories ---
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    #os.makedirs(DEBUG_OUTPUT_FOLDER, exist_ok=True) 
    
    logging.info("--- Starting Image Enhancement Process for BFIW-like texture with Transparent Background ---")
    logging.info(f"Input Images:          {INPUT_IMAGE_FOLDER}")
    logging.info(f"Input Masks:           {INPUT_MASK_FOLDER}")
    logging.info(f"Output Destination:    {OUTPUT_FOLDER}")
    #logging.info(f"Debug Outputs:         {DEBUG_OUTPUT_FOLDER}")
    logging.info(f"Enhancement Config:    {processing_config}")

    # --- 4. Collect all processing tasks ---
    logging.info("Collecting image and mask pairs...")
    tasks = collect_processing_tasks(INPUT_IMAGE_FOLDER, INPUT_MASK_FOLDER, OUTPUT_FOLDER, processing_config)
    
    if not tasks:
        logging.warning("No image-mask pairs found for processing.")
        logging.warning("Please ensure your 'INPUT_IMAGE_FOLDER' and 'INPUT_MASK_FOLDER' paths are correct,")
        logging.warning("and that mask filenames correspond to image filenames as expected by the 'collect_processing_tasks' function.")
    else:
        logging.info(f"Found {len(tasks)} image-mask pairs to process.")
        
        # --- 5. Run Multiprocessing ---
        num_processes = max(1, cpu_count() - 1) 
        logging.info(f"Using {num_processes} parallel processes for faster processing.")

        with Pool(processes=num_processes) as pool:
            list(tqdm(pool.imap_unordered(process_single_image, tasks), total=len(tasks), desc="Processing Images"))
        
        logging.info("--- Image Enhancement Process Complete ---")
        logging.info(f"Check your enhanced images in: {OUTPUT_FOLDER}")
        #logging.info(f"Detailed intermediate steps for each image are in: {DEBUG_OUTPUT_FOLDER}")
        logging.info("Review the log messages above for any skipped files or errors during processing.")