import cv2
import numpy as np
import os
import re
from matplotlib import pyplot as plt
import multiprocessing # Import multiprocessing module
from collections import deque # For a simple queue/buffer

# --- Global CLAHE instance (for multiprocessing, it needs to be created in each process) ---
# To avoid pickling issues, we'll create CLAHE inside the worker function.

def natural_sort_key(s):
    """Sorts strings containing numbers in a natural way."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def preprocess_image_worker(image_path, pad_amount, new_dims, downsample_dims_coarse):
    """
    Worker function to load, pad, preprocess (CLAHE, Gaussian), and downsample an image.
    This function will be run in separate processes.
    """
    img_color_orig = cv2.imread(image_path)
    if img_color_orig is None:
        print(f"Worker Warning: Could not load image {image_path}. Returning None.")
        return None, None, None

    # Pad the image
    img_color_padded = cv2.copyMakeBorder(img_color_orig, pad_amount, pad_amount, pad_amount, pad_amount, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_gray_padded_full = cv2.cvtColor(img_color_padded, cv2.COLOR_BGR2GRAY)

    # Initialize CLAHE and Gaussian Blur in this process
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 
    img_gray_processed_full = clahe.apply(img_gray_padded_full)
    img_gray_processed_full = cv2.GaussianBlur(img_gray_processed_full, (5,5), 0)

    # Downsample for coarse pass
    img_gray_processed_coarse = cv2.resize(img_gray_processed_full, downsample_dims_coarse, interpolation=cv2.INTER_AREA)

    return img_color_padded, img_gray_processed_full, img_gray_processed_coarse


def register_nissl_images_sequential_ecc_multires(input_folder, output_folder, 
                                            transformation_type=cv2.MOTION_EUCLIDEAN,
                                            padding_factor=0.75, 
                                            cc_threshold_coarse=0.5, cc_threshold_fine=0.7, 
                                            downsample_factor=0.25, # Coarse resolution factor
                                            max_iters_coarse=2000, eps_coarse=1e-5,
                                            max_iters_fine=5000, eps_fine=1e-6,
                                            visualize_registration=False,
                                            num_workers=4, # New parameter for multiprocessing workers
                                            buffer_size=5): # New parameter for pre-processed image buffer
    """
    Registers sequential Nissl images using a multi-resolution ECC approach
    with multiprocessing for image loading and preprocessing.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg'))], key=natural_sort_key)

    if not image_files:
        print(f"No JPG/JPEG image files found in {input_folder}")
        return

    # --- Determine padding and dimensions based on the first image ---
    ref_image_name = image_files[0]
    reference_image_path = os.path.join(input_folder, ref_image_name)

    print(f"Using the first image in sequence as absolute reference: {reference_image_path}")
    img_ref_abs_color_orig = cv2.imread(reference_image_path)
    if img_ref_abs_color_orig is None:
        print(f"Error: Could not load absolute reference image {reference_image_path}. Check path and format.")
        return

    height_orig, width_orig = img_ref_abs_color_orig.shape[:2]

    max_dim = max(height_orig, width_orig)
    pad_amount = int(max_dim * padding_factor)

    new_height_padded = height_orig + 2 * pad_amount
    new_width_padded = width_orig + 2 * pad_amount

    crop_x1 = pad_amount
    crop_y1 = pad_amount
    crop_x2 = crop_x1 + width_orig
    crop_y2 = crop_y1 + height_orig

    coarse_width = int(new_width_padded * downsample_factor)
    coarse_height = int(new_height_padded * downsample_factor)
    downsample_dims_coarse = (coarse_width, coarse_height) # Tuple for passing to worker

    print(f"Original image size: {width_orig}x{height_orig}")
    print(f"Padding each side by: {pad_amount} pixels")
    print(f"Padded image size for registration: {new_width_padded}x{new_height_padded}")
    print(f"Output will be cropped to original reference size: {width_orig}x{height_orig}")

    # --- Process the absolute reference image using the worker function ---
    print(f"Pre-processing absolute reference image: {ref_image_name}...")
    img_ref_abs_color_padded, img_ref_gray_processed_full, img_ref_gray_processed_coarse = \
        preprocess_image_worker(reference_image_path, pad_amount, (new_width_padded, new_height_padded), downsample_dims_coarse)

    if img_ref_abs_color_padded is None:
        print("Error: Pre-processing of initial reference failed. Exiting.")
        return

    # Save the *original* absolute reference
    cv2.imwrite(os.path.join(output_folder, f"registered_{ref_image_name}"), img_ref_abs_color_orig, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"Saved absolute reference image (original size): registered_{ref_image_name}")

    # Initialize current reference for pairwise registration
    current_ref_color_for_warp = img_ref_abs_color_padded 
    current_ref_gray_for_ecc_full = img_ref_gray_processed_full
    current_ref_gray_for_ecc_coarse = img_ref_gray_processed_coarse

    # Initialize the cumulative transformation matrix
    cumulative_warp_matrix = np.eye(2, 3, dtype=np.float32)
    if transformation_type == cv2.MOTION_HOMOGRAPHY:
        cumulative_warp_matrix = np.eye(3, 3, dtype=np.float32)

    # ECC criteria
    criteria_coarse = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iters_coarse, eps_coarse)
    criteria_fine = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iters_fine, eps_fine)

    print(f"\nStarting sequential multi-resolution ECC registration process for {len(image_files)} images...")
    processed_count = 0
    skipped_images = []

    # --- Multiprocessing setup ---
    pool = multiprocessing.Pool(processes=num_workers)
    # Use a deque as a buffer to store pre-processed images
    preprocessed_buffer = deque() 

    # Prefill the buffer
    print(f"Pre-filling buffer with {min(buffer_size, len(image_files) - 1)} images...")
    for j in range(1, min(buffer_size + 1, len(image_files))): # Fill buffer for the first few images
        img_name = image_files[j]
        img_path = os.path.join(input_folder, img_name)
        # Apply async for non-blocking submission
        preprocessed_buffer.append(
            pool.apply_async(preprocess_image_worker, 
                             (img_path, pad_amount, (new_width_padded, new_height_padded), downsample_dims_coarse))
        )

    # Iterate through images, starting from the second one
    for i in range(1, len(image_files)):
        img_name = image_files[i]
        
        # Get the pre-processed image from the buffer
        if not preprocessed_buffer: # Should not happen if prefill and subsequent adds work
             print(f"Error: Buffer empty for {img_name}. Skipping.")
             skipped_images.append(img_name)
             continue
        
        # Retrieve result from the async task
        img_moving_color_padded, img_moving_gray_processed_full, img_moving_gray_processed_coarse = \
            preprocessed_buffer.popleft().get() # .get() blocks until result is ready

        if img_moving_color_padded is None: # Means worker failed to load/process
            print(f"Warning: Pre-processing failed for {img_name}. Skipping.")
            skipped_images.append(img_name)
            continue

        # As soon as we take one out, add the next one to keep the buffer full
        if i + buffer_size < len(image_files):
            next_img_name = image_files[i + buffer_size]
            next_img_path = os.path.join(input_folder, next_img_name)
            preprocessed_buffer.append(
                pool.apply_async(preprocess_image_worker, 
                                 (next_img_path, pad_amount, (new_width_padded, new_height_padded), downsample_dims_coarse))
            )

        # Initialize pairwise matrix for ECC (always identity relative to current_ref)
        initial_pairwise_warp_matrix = np.eye(2, 3, dtype=np.float32)
        if transformation_type == cv2.MOTION_HOMOGRAPHY:
            initial_pairwise_warp_matrix = np.eye(3, 3, dtype=np.float32)

        # --- COARSE Registration Pass (on downsampled images) ---
        pairwise_warp_matrix_coarse = initial_pairwise_warp_matrix.copy()
        cc_coarse = -1

        try:
            (cc_coarse, pairwise_warp_matrix_coarse) = cv2.findTransformECC(
                np.float32(current_ref_gray_for_ecc_coarse), 
                np.float32(img_moving_gray_processed_coarse), 
                pairwise_warp_matrix_coarse, 
                transformation_type, 
                criteria_coarse
            )

            if cc_coarse < cc_threshold_coarse:
                print(f"Warning: Low coarse CC ({cc_coarse:.2f}) for {img_name}. Skipping and not updating cumulative transform.")
                skipped_images.append(img_name)
                continue 

            scaled_pairwise_matrix = pairwise_warp_matrix_coarse.copy()
            if transformation_type != cv2.MOTION_HOMOGRAPHY:
                scaled_pairwise_matrix[0, 2] *= (new_width_padded / coarse_width)
                scaled_pairwise_matrix[1, 2] *= (new_height_padded / coarse_height)
            else:
                # Homography scaling is more complex, typically by hand or use point transforms.
                # For this general case, we'll assume the provided matrix for HOMOGRAPHY 
                # will implicitly handle the coordinate space or requires manual point transform scaling.
                # For now, it's left as is for HOMOGRAPHY.
                pass

        except cv2.error as e:
            print(f"Error during coarse ECC for {img_name}: {e}. Skipping.")
            skipped_images.append(img_name)
            continue 

        # --- FINE Registration Pass (on full-resolution images, starting from coarse guess) ---
        pairwise_warp_matrix_fine = scaled_pairwise_matrix.copy() 
        cc_fine = -1

        try:
            (cc_fine, pairwise_warp_matrix_fine) = cv2.findTransformECC(
                np.float32(current_ref_gray_for_ecc_full), 
                np.float32(img_moving_gray_processed_full), 
                pairwise_warp_matrix_fine, 
                transformation_type, 
                criteria_fine
            )

            if cc_fine < cc_threshold_fine:
                print(f"Warning: Low fine CC ({cc_fine:.2f}) for {img_name}. Skipping and not updating cumulative transform.")
                skipped_images.append(img_name)
                continue 
            
            current_pairwise_matrix = pairwise_warp_matrix_fine
            used_cc = cc_fine

        except cv2.error as e:
            print(f"Error during fine ECC for {img_name}: {e}. Skipping.")
            skipped_images.append(img_name)
            continue 

        # --- Compose the new pairwise transformation with the cumulative transformation ---
        if transformation_type == cv2.MOTION_HOMOGRAPHY:
            cumulative_warp_matrix = np.dot(cumulative_warp_matrix, current_pairwise_matrix)
        else: # Affine, Euclidean, Translation (all 2x3 matrices)
            M_prev_3x3 = np.vstack([cumulative_warp_matrix, [0,0,1]])
            M_pairwise_3x3 = np.vstack([current_pairwise_matrix, [0,0,1]])
            M_composed_3x3 = np.dot(M_prev_3x3, M_pairwise_3x3)
            cumulative_warp_matrix = M_composed_3x3[:2, :] 

        # --- Warp the current moving image with the cumulative transform ---
        if transformation_type == cv2.MOTION_HOMOGRAPHY:
            img_registered_padded = cv2.warpPerspective(img_moving_color_padded, cumulative_warp_matrix, (new_width_padded, new_height_padded))
        else:
            img_registered_padded = cv2.warpAffine(img_moving_color_padded, cumulative_warp_matrix, (new_width_padded, new_height_padded))

        # --- Crop to original reference size ---
        img_registered_cropped = img_registered_padded[crop_y1:crop_y2, crop_x1:crop_x2]

        output_path = os.path.join(output_folder, f"registered_{img_name}")
        cv2.imwrite(output_path, img_registered_cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"Registered and saved: {img_name} (CC_coarse: {cc_coarse:.4f}, CC_fine: {cc_fine:.4f}) -> registered_{img_name}")
        processed_count += 1

        # --- Update current reference for the next iteration ---
        current_ref_color_for_warp = img_moving_color_padded 
        current_ref_gray_for_ecc_full = img_moving_gray_processed_full
        current_ref_gray_for_ecc_coarse = img_moving_gray_processed_coarse

        # --- Optional: Visualize registration for debugging ---
        if visualize_registration:
            pairwise_registered_padded_debug = None
            if transformation_type == cv2.MOTION_HOMOGRAPHY:
                pairwise_registered_padded_debug = cv2.warpPerspective(img_moving_color_padded, current_pairwise_matrix, (new_width_padded, new_height_padded))
            else:
                pairwise_registered_padded_debug = cv2.warpAffine(img_moving_color_padded, current_pairwise_matrix, (new_width_padded, new_height_padded))
            
            overlay_pairwise = cv2.addWeighted(current_ref_color_for_warp, 0.5, pairwise_registered_padded_debug, 0.5, 0)
            overlay_cumulative = cv2.addWeighted(img_ref_abs_color_padded, 0.5, img_registered_padded, 0.5, 0)

            overlay_pairwise_rgb = cv2.cvtColor(overlay_pairwise, cv2.COLOR_BGR2RGB)
            overlay_cumulative_rgb = cv2.cvtColor(overlay_cumulative, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(15, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(overlay_pairwise_rgb[crop_y1:crop_y2, crop_x1:crop_x2]) 
            plt.title(f"Pairwise Overlay: {image_files[i-1]} vs {img_name} (CC_fine: {cc_fine:.2f})")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(overlay_cumulative_rgb[crop_y1:crop_y2, crop_x1:crop_x2]) 
            plt.title(f"Cumulative Overlay: {ref_image_name} vs {img_name}")
            plt.axis('off')
            plt.show(block=False) 
            plt.pause(0.5) 
            plt.close()

    # Close the pool and wait for all tasks to complete
    pool.close()
    pool.join()

    print(f"\nImage registration complete. Successfully processed {processed_count} images.")
    if skipped_images:
        print(f"The following images were skipped due to poor correlation or errors: {', '.join(skipped_images)}")

# --- Example Usage ---
if __name__ == "__main__":
    # Ensure your actual input_dir and output_dir are set correctly
    input_dir = "/home/projects/medimg/supriti/brain-registration/141/141_nissl"
    output_dir = "/home/projects/medimg/supriti/141_nissl_reg_opt" # New output folder name

    # --- Dummy Image Creation for Demonstration (REMOVE THIS BLOCK WHEN USING YOUR REAL DATA) ---
    if not os.path.exists(input_dir) or not os.listdir(input_dir): 
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        print(f"Creating dummy images in '{input_dir}' for demonstration. PLEASE REMOVE THIS BLOCK FOR REAL DATA.")
        
        ref_img = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.circle(ref_img, (250, 250), 100, (0, 0, 255), -1) 
        cv2.rectangle(ref_img, (100, 100), (200, 200), (255, 0, 0), -1) 
        cv2.putText(ref_img, "Ref", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(input_dir, "1.jpg"), ref_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        current_dummy_img = ref_img.copy()
        for i in range(2, 21): 
            angle = np.random.uniform(-1.5, 1.5)
            tx = np.random.uniform(-5, 5) 
            ty = np.random.uniform(-5, 5)
            
            M_pairwise = np.eye(2, 3, dtype=np.float32)
            cos_theta = np.cos(np.radians(angle))
            sin_theta = np.sin(np.radians(angle))
            M_pairwise[0, 0] = cos_theta
            M_pairwise[0, 1] = -sin_theta
            M_pairwise[1, 0] = sin_theta
            M_pairwise[1, 1] = cos_theta
            M_pairwise[0, 2] = tx
            M_pairwise[1, 2] = ty
            
            next_dummy_img = cv2.warpAffine(current_dummy_img, M_pairwise, (500, 500))
            cv2.imwrite(os.path.join(input_dir, f"{i}.jpg"), next_dummy_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            current_dummy_img = next_dummy_img

        # Add an image that is intentionally very different to cause skipping
        very_diff_img = np.zeros((500,500,3), dtype=np.uint8)
        cv2.rectangle(very_diff_img, (50,50), (100,100), (255,0,0), -1)
        cv2.circle(very_diff_img, (450,450), 30, (0,255,0), -1)
        cv2.putText(very_diff_img, "Skip Me", (200,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imwrite(os.path.join(input_dir, "21.jpg"), very_diff_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
        print(f"Created dummy images in '{input_dir}' for demonstration.")
    else:
        print(f"Using existing images in '{input_dir}'.")
    # --- END OF DUMMY IMAGE CREATION BLOCK ---

    # --- Run the sequential registration ---
    print("\n--- Running Multi-Resolution ECC Enhanced Pre-padded & Cropped Registration (Optimized with Multiprocessing) ---")
    register_nissl_images_sequential_ecc_multires(input_dir, output_dir, 
                                            transformation_type=cv2.MOTION_EUCLIDEAN, 
                                            padding_factor=0.75, 
                                            cc_threshold_coarse=0.4, 
                                            cc_threshold_fine=0.7, 
                                            downsample_factor=0.25, 
                                            max_iters_coarse=2000,   
                                            eps_coarse=1e-5,         
                                            max_iters_fine=5000,     
                                            eps_fine=1e-6,           
                                            visualize_registration=False, # Set to True for debugging!
                                            num_workers=os.cpu_count(), # Use all available CPU cores for workers
                                            buffer_size=min(os.cpu_count() * 2, 10) # Buffer a few more images
                                            )

    # Optional: Display some final results
    print("\nAttempting to display final comparison for a processed image...")
    original_img_name_to_display = "20.jpg" 
    
    original_img_path_for_display = os.path.join(input_dir, original_img_name_to_display)
    registered_img_path_for_display = os.path.join(output_dir, f"registered_{original_img_name_to_display}")
    ref_img_path_for_display = os.path.join(input_dir, "1.jpg") 

    if os.path.exists(original_img_path_for_display) and \
       os.path.exists(registered_img_path_for_display) and \
       os.path.exists(ref_img_path_for_display):
        
        original = cv2.imread(original_img_path_for_display)
        registered = cv2.imread(registered_img_path_for_display)
        reference_original = cv2.imread(ref_img_path_for_display)

        if original is not None and registered is not None and reference_original is not None:
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(reference_original, cv2.COLOR_BGR2RGB))
            plt.title("Absolute Reference (Original 1.jpg)")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            plt.title(f"Original Moving Image ({original_img_name_to_display})")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(registered, cv2.COLOR_BGR2RGB))
            plt.title(f"Registered (Cropped {original_img_name_to_display})")
            plt.axis('off')
            plt.show()
        else:
            print(f"Failed to load one or more images for final display. Check file corruption or paths.")
    else:
        print("Could not find images to display for final comparison. Ensure the chosen files exist in respective folders.")
