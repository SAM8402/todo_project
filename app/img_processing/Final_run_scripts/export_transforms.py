import json
import os
import re
import numpy as np
import cv2 # Only for cv2.getRotationMatrix2D if you were to inline that part

# --- Configuration (Copied and adapted from app.py) ---
# This should be identical to the BRAIN_SAMPLES_CONFIG in your app.py
BRAIN_SAMPLES_CONFIG = {
    "142": {
        "nissl_dims": (1333, 1333), "bfw_dims": (2076, 3088), "apply_nissl_zoom": False,
        "key_slices": [
            {"slice_num": 175,  "bfw_css_rotation_deg": -90, "align_params": {"scale": 1.6316, "angle": 0.00, "tx": -290.59,  "ty": 615.03}},
            {"slice_num": 319,  "bfw_css_rotation_deg": -90, "align_params": {"scale": 1.3409, "angle": 0.01, "tx": -275.28,  "ty": 551.96}},
            {"slice_num": 1024,  "bfw_css_rotation_deg": -90, "align_params": {"scale": 0.8686, "angle": -0.01, "tx": 106.39, "ty": 242.86}},
            {"slice_num": 1216, "bfw_css_rotation_deg": -90, "align_params": {"scale": 0.9633 , "angle": 0.00 , "tx": 37.35 , "ty": 358.52}},
            {"slice_num": 1360, "bfw_css_rotation_deg": -90, "align_params": {"scale": 1.1718, "angle": 0.10,"tx": -97.97,  "ty": 376.60}},
            {"slice_num": 1510, "bfw_css_rotation_deg": -90, "align_params": {"scale": 1.0166, "angle": 0.00, "tx": -176.03,  "ty": 315.81}},
        ],
        "nissl_file_template": "nissl-{slice_num}", "bfw_file_template": "bfi-{slice_num}"
    }
}    
#    "244": {
#        "nissl_dims": (1333, 1333), "bfw_dims": (2000, 2032), "apply_nissl_zoom": False,
#        "key_slices": [
#            {"slice_num": 289,  "bfw_css_rotation_deg": 75, "align_params": {"scale": 1.4079, "angle": 0.00, "tx": 247.06, "ty": 154.06}},
#            {"slice_num": 1102, "bfw_css_rotation_deg": 60, "align_params": {"scale": 1.3581, "angle": 0.01, "tx": 222.87, "ty": 121.08}},
#            {"slice_num": 1120, "bfw_css_rotation_deg": 60, "align_params": {"scale": 1.3429, "angle": 0.00, "tx": 216.57, "ty": 126.86}},
#            {"slice_num": 1324, "bfw_css_rotation_deg": 60, "align_params": {"scale": 1.3585, "angle": 0.01, "tx": 257.07, "ty": 149.90}},
#            {"slice_num": 1489, "bfw_css_rotation_deg": 66, "align_params": {"scale": 1.3073, "angle": 0.00, "tx": 218.84, "ty": 25.96}},
#            {"slice_num": 1774, "bfw_css_rotation_deg": 80, "align_params": {"scale": 1.3383, "angle": 0.00, "tx": 194.01, "ty": -7.42}},
#            {"slice_num": 2038, "bfw_css_rotation_deg": 90, "align_params": {"scale": 1.3498, "angle": 0.01, "tx": 190.57, "ty": 28.95}},
#        ],
#        "nissl_file_template": "nissl-{slice_num}", "bfw_file_template": "bfi-{slice_num}"
#    } 
#}

# Common constants (Copied from app.py)
#CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#IMAGES_BASE_DIR = os.path.join(CURRENT_SCRIPT_DIR, 'images_data') # Assuming 'images_data' is in the same dir as this script
IMAGES_BASE_DIR = "/home/projects/medimg/supriti/brain-registration/142/images_data/"
TARGET_PADDED_DIMS = (5000, 5000)
PADDED_IMAGE_CENTER = (TARGET_PADDED_DIMS[1] / 2.0, TARGET_PADDED_DIMS[0] / 2.0)

# --- Helper Functions (Copied VERBATIM from app.py) ---
def get_affine_matrix_3x3(M_2x3):
    return np.vstack([M_2x3, [0, 0, 1]])

def linear_interpolate(p1, p2, x1, x2, x_target):
    if x1 == x2:
        return p1
    # Ensure float division for Python 2, though not strictly necessary in Python 3
    return p1 + (p2 - p1) * float(x_target - x1) / float(x2 - x1)


def get_interpolated_params(key_slices_config, requested_slice_num):
    sorted_keys = sorted(key_slices_config, key=lambda k: k['slice_num'])
    if not sorted_keys:
        raise ValueError("No key slices defined for interpolation.")
    if requested_slice_num <= sorted_keys[0]['slice_num']:
        return sorted_keys[0]
    if requested_slice_num >= sorted_keys[-1]['slice_num']:
        return sorted_keys[-1]
    key1, key2 = None, None
    for i in range(len(sorted_keys) - 1):
        if sorted_keys[i]['slice_num'] <= requested_slice_num < sorted_keys[i+1]['slice_num']:
            key1 = sorted_keys[i]
            key2 = sorted_keys[i+1]
            break
    if not key1 or not key2:
        # This fallback should ideally not be hit if extrapolation logic is correct
        return min(sorted_keys, key=lambda k: abs(k['slice_num'] - requested_slice_num))
    s1, s2 = key1['slice_num'], key2['slice_num']
    interp_bfw_rot = linear_interpolate(key1['bfw_css_rotation_deg'], key2['bfw_css_rotation_deg'], s1, s2, requested_slice_num)
    interp_align_params = {}
    for param_name in ['scale', 'angle', 'tx', 'ty']:
        interp_align_params[param_name] = linear_interpolate(
            key1['align_params'][param_name], key2['align_params'][param_name],
            s1, s2, requested_slice_num
        )
    return {
        "slice_num": requested_slice_num,
        "bfw_css_rotation_deg": interp_bfw_rot,
        "align_params": interp_align_params
    }

def calculate_matrices_with_interpolated_params(sample_main_config, interpolated_slice_params):
    orig_nissl_dims = sample_main_config['nissl_dims']
    orig_bfw_dims = sample_main_config['bfw_dims']
    apply_nissl_zoom = sample_main_config.get('apply_nissl_zoom', False)
    interp_bfw_css_rot_deg = interpolated_slice_params['bfw_css_rotation_deg']
    interp_align = interpolated_slice_params['align_params']
    top_n = (TARGET_PADDED_DIMS[0] - orig_nissl_dims[0]) // 2
    left_n = (TARGET_PADDED_DIMS[1] - orig_nissl_dims[1]) // 2
    T_pad_N = np.array([[1, 0, left_n], [0, 1, top_n], [0, 0, 1]], dtype=np.float64)
    top_b = (TARGET_PADDED_DIMS[0] - orig_bfw_dims[0]) // 2
    left_b = (TARGET_PADDED_DIMS[1] - orig_bfw_dims[1]) // 2
    T_pad_B = np.array([[1, 0, left_b], [0, 1, top_b], [0, 0, 1]], dtype=np.float64)
    nissl_zoom_factor = 1.7743 if apply_nissl_zoom else 1.0
    M_zoom_N_2x3 = cv2.getRotationMatrix2D(center=PADDED_IMAGE_CENTER, angle=0, scale=nissl_zoom_factor)
    T_zoom_N = get_affine_matrix_3x3(M_zoom_N_2x3)
    bfw_cv2_rot_deg = -interp_bfw_css_rot_deg
    M_rot_B_2x3 = cv2.getRotationMatrix2D(center=PADDED_IMAGE_CENTER, angle=bfw_cv2_rot_deg, scale=1.0)
    T_rot_B = get_affine_matrix_3x3(M_rot_B_2x3)
    M_align_2x3_base = cv2.getRotationMatrix2D(
        center=PADDED_IMAGE_CENTER, angle=interp_align['angle'], scale=interp_align['scale']
    )
    M_align_2x3 = M_align_2x3_base.copy()
    M_align_2x3[0, 2] += interp_align['tx']
    M_align_2x3[1, 2] += interp_align['ty']
    T_align = get_affine_matrix_3x3(M_align_2x3)
    H_Norig_to_Borig = np.linalg.inv(T_pad_B) @ np.linalg.inv(T_rot_B) @ T_align @ T_zoom_N @ T_pad_N
    H_Borig_to_Norig = np.linalg.inv(H_Norig_to_Borig)
    return H_Norig_to_Borig, H_Borig_to_Norig, interp_bfw_css_rot_deg

# --- Function to find available physical slices (Copied and adapted from app.py) ---
def find_available_slices(sample_id, sample_config):
    sample_image_dir = os.path.join(IMAGES_BASE_DIR, sample_id)
    if not os.path.isdir(sample_image_dir):
        print(f"Warning: Image directory for sample {sample_id} not found at {sample_image_dir}")
        return []

    nissl_base_pattern = sample_config['nissl_file_template'].replace("{slice_num}", "(\\d+)")
    bfi_base_pattern = sample_config['bfw_file_template'].replace("{slice_num}", "(\\d+)")
    image_extensions_pattern = "\\.(?:jpg|jpeg|png|tif|tiff|gif|bmp)"
    nissl_pattern_str = f"^{nissl_base_pattern}{image_extensions_pattern}$"
    bfi_pattern_str = f"^{bfi_base_pattern}{image_extensions_pattern}$"
    nissl_regex = re.compile(nissl_pattern_str, re.IGNORECASE)
    bfi_regex = re.compile(bfi_pattern_str, re.IGNORECASE)
    
    nissl_slices_files = {}
    bfi_slices_files = {}
    for filename in os.listdir(sample_image_dir):
        nissl_match = nissl_regex.match(filename)
        if nissl_match: nissl_slices_files[int(nissl_match.group(1))] = filename
        bfi_match = bfi_regex.match(filename)
        if bfi_match: bfi_slices_files[int(bfi_match.group(1))] = filename
            
    common_slice_numbers = sorted(list(set(nissl_slices_files.keys()).intersection(set(bfi_slices_files.keys()))))
    
    # Return just the numbers for iteration
    return common_slice_numbers


# --- Main Script Logic ---
def generate_all_transforms(output_filename="all_slice_transformations.json"):
    all_brains_data = {}

    for sample_id, sample_main_config in BRAIN_SAMPLES_CONFIG.items():
        print(f"Processing brain sample: {sample_id}...")
        sample_data = {
            "sample_id": sample_id,
            "nissl_dims_original": list(sample_main_config['nissl_dims']), # Convert tuple to list for JSON
            "bfw_dims_original": list(sample_main_config['bfw_dims']),   # Convert tuple to list for JSON
            "apply_nissl_zoom_preprocessing": sample_main_config.get('apply_nissl_zoom', False),
            "slices": []
        }
        
        key_slices_config = sample_main_config.get('key_slices', [])
        if not key_slices_config:
            print(f"  Warning: No key slices defined for sample {sample_id}. Skipping interpolation.")
            all_brains_data[sample_id] = sample_data
            continue

        available_slice_numbers = find_available_slices(sample_id, sample_main_config)
        if not available_slice_numbers:
            print(f"  No physical image slices found for sample {sample_id}. Skipping.")
            all_brains_data[sample_id] = sample_data
            continue
            
        print(f"  Found {len(available_slice_numbers)} available physical slices. Calculating transforms...")

        for requested_slice_num in available_slice_numbers:
            try:
                interpolated_params = get_interpolated_params(key_slices_config, requested_slice_num)
                H_N_to_B, H_B_to_N, bfw_css_rotation = calculate_matrices_with_interpolated_params(
                    sample_main_config, interpolated_params
                )
                
                slice_transform_data = {
                    "slice_num": requested_slice_num,
                    "interpolated_bfw_css_rotation_deg": bfw_css_rotation,
                    "H_nissl_to_bfw": H_N_to_B.tolist(), # Convert numpy array to list for JSON
                    "H_bfw_to_nissl": H_B_to_N.tolist(), # Convert numpy array to list for JSON
                    # Optionally include the interpolated align_params themselves
                    "interpolated_align_params": interpolated_params['align_params']
                }
                sample_data["slices"].append(slice_transform_data)
            except Exception as e:
                print(f"    Error processing slice {requested_slice_num} for sample {sample_id}: {e}")
        
        all_brains_data[sample_id] = sample_data
        print(f"  Finished processing sample {sample_id}.")

    # Save to JSON file
    try:
        with open(output_filename, 'w') as f:
            json.dump(all_brains_data, f, indent=4)
        print(f"\nSuccessfully saved all transformation data to: {output_filename}")
    except IOError as e:
        print(f"\nError writing to file {output_filename}: {e}")

if __name__ == "__main__":
    generate_all_transforms()
    # Example to generate for a specific brain if needed:
    # generate_all_transforms(specific_brain_ids=["222"], output_filename="brain_222_transforms.json")


