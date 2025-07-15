from django.shortcuts import render
from django.conf import settings

import requests
import json
import os
import re
import logging
from django.shortcuts import render, redirect
from django.http import HttpResponse, Http404
from django.conf import settings
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

def fetch_brain_viewer_details(biosample_id):
    bfidic = {
        "222": 100,
        "244": 116,
    }
    bfi_value = bfidic.get(str(biosample_id))
    if not bfi_value:
        logger.error(f"BFI value not found for biosample_id: {biosample_id}")
        return None
    url = f"http://dev2adi.humanbrain.in:8000/GW/getBrainViewerDetails/IIT/V1/SS-{bfi_value}:-1:-1"
    logger.info(f"Sending request to: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error if the response code isn't 200
        logger.info(f"Successfully fetched data from {url}")
        return response.json()  # Or response.text based on the expected data format
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from {url}: {e}")
        return None
    

def get_jp2_metadata(biosample_id, section_number):
    logger.info(
        f"Fetching JP2 metadata for biosample: {biosample_id}, section: {section_number}")

    # Fetch JSON data
    json_data = fetch_brain_viewer_details(biosample_id)
    if not json_data:
        logger.error(
            f"Failed to fetch JSON data for biosample {biosample_id}, section {section_number}")
        return None

    # Extract NISSL data from the JSON response
    jp2_data = json_data.get('thumbNail', {}).get('NISSL', [])
    if not jp2_data:
        logger.error(
            f"No 'NISSL' data found in the JSON for biosample {biosample_id}, section {section_number}")
        return None

    # Try exact match first
    logger.info(
        f"Searching for exact match for section {section_number} in NISSL data.")
    exact_match = next(
        (item for item in jp2_data if str(
            item.get('position_index')) == str(section_number)),
        None
    )

    match_item = exact_match

    # If exact match not found, find the nearest match by numeric distance
    if not match_item:
        logger.info(
            f"No exact match found for section {section_number}. Searching for the nearest match.")
        try:
            section_number_int = int(section_number)
        except ValueError:
            logger.error(
                f"Invalid section_number: {section_number}. It must be an integer.")
            return None

        valid_items = [item for item in jp2_data if isinstance(
            item.get('position_index'), int)]
        if not valid_items:
            logger.error(
                f"No valid position_index found in NISSL data for biosample {biosample_id}.")
            return None

        nearest_item = min(
            valid_items,
            key=lambda item: abs(item['position_index'] - section_number_int)
        )
        match_item = nearest_item
        logger.info(
            f"Nearest match found for section {section_number}: position_index {match_item['position_index']}.")

    # If no match is found, log the failure and return None
    if not match_item:
        logger.error(
            f"Could not find any matching data for section {section_number} in NISSL.")
        return None

    # Extract only required fields and log the values
    filtered = {
        'height': match_item['height'],
        'width': match_item['width'],
        'rotation': match_item.get('rigidrotation', 0),
        'jp2_path_fragment': match_item['jp2Path'],
        'position_index': match_item['position_index']
    }

    logger.info(
        f"Successfully retrieved JP2 metadata for biosample {biosample_id}, section {section_number}: {filtered}")

    return filtered


def get_haematoxylin_and_eosin_metadata(biosample_id, section_number):
    logger.info(
        f"Fetching Haematoxylin and Eosin metadata for biosample: {biosample_id}, section: {section_number}")

    # Fetch JSON data
    json_data = fetch_brain_viewer_details(biosample_id)
    if not json_data:
        logger.error(
            f"Failed to fetch JSON data for biosample {biosample_id}, section {section_number}")
        return None

    # Extract Haematoxylin and Eosin data from the JSON response
    he_data = json_data.get('thumbNail', {}).get('Haematoxylin and Eosin', [])
    if not he_data:
        logger.error(
            f"No 'Haematoxylin and Eosin' data found in the JSON for biosample {biosample_id}, section {section_number}")
        return None

    # Try exact match first
    logger.info(
        f"Searching for exact match for section {section_number} in Haematoxylin and Eosin data.")
    exact_match = next(
        (item for item in he_data if str(
            item.get('position_index')) == str(section_number)),
        None
    )

    match_item = exact_match

    # If exact match not found, find the nearest match by numeric distance
    if not match_item:
        logger.info(
            f"No exact match found for section {section_number}. Searching for the nearest match.")
        try:
            section_number_int = int(section_number)
        except ValueError:
            logger.error(
                f"Invalid section_number: {section_number}. It must be an integer.")
            return None

        valid_items = [item for item in he_data if isinstance(
            item.get('position_index'), int)]
        if not valid_items:
            logger.error(
                f"No valid position_index found in Haematoxylin and Eosin data for biosample {biosample_id}.")
            return None

        nearest_item = min(
            valid_items,
            key=lambda item: abs(item['position_index'] - section_number_int)
        )
        match_item = nearest_item
        logger.info(
            f"Nearest match found for section {section_number}: position_index {match_item['position_index']}.")

    # If no match is found, log the failure and return None
    if not match_item:
        logger.error(
            f"Could not find any matching data for section {section_number} in Haematoxylin and Eosin.")
        return None

    # Extract only required fields and log the values
    filtered = {
        'height': match_item['height'],
        'width': match_item['width'],
        'rotation': match_item.get('rigidrotation', 0),
        'jp2_path_fragment': match_item['jp2Path'],
        'position_index': match_item['position_index']
    }

    logger.info(
        f"Successfully retrieved Haematoxylin and Eosin metadata for biosample {biosample_id}, section {section_number}: {filtered}")

    return filtered


def get_mri_metadata(biosample_id, section_number):
    logger.info(
        f"Fetching MRI metadata for biosample: {biosample_id}, section: {section_number}")

    # Fetch JSON data
    json_data = fetch_brain_viewer_details(biosample_id)
    if not json_data:
        logger.error(
            f"Failed to fetch JSON data for biosample {biosample_id}, section {section_number}")
        return None

    # Extract MRI data from the JSON response
    mri_data = json_data.get('thumbNail', {}).get('MRI', [])
    if not mri_data:
        logger.error(
            f"No 'MRI' data found in the JSON for biosample {biosample_id}, section {section_number}")
        return None

    # Try exact match first
    logger.info(
        f"Searching for exact match for section {section_number} in MRI data.")
    exact_match = next(
        (item for item in mri_data if str(
            item.get('position_index')) == str(section_number)),
        None
    )

    match_item = exact_match

    # If exact match not found, find the nearest match by numeric distance
    if not match_item:
        logger.info(
            f"No exact match found for section {section_number}. Searching for the nearest match.")
        try:
            section_number_int = int(section_number)
        except ValueError:
            logger.error(
                f"Invalid section_number: {section_number}. It must be an integer.")
            return None

        valid_items = [item for item in mri_data if isinstance(
            item.get('position_index'), int)]
        if not valid_items:
            logger.error(
                f"No valid position_index found in MRI data for biosample {biosample_id}.")
            return None

        nearest_item = min(
            valid_items,
            key=lambda item: abs(item['position_index'] - section_number_int)
        )
        match_item = nearest_item
        logger.info(
            f"Nearest match found for section {section_number}: position_index {match_item['position_index']}.")

    # If no match is found, log the failure and return None
    if not match_item:
        logger.error(
            f"Could not find any matching data for section {section_number} in MRI.")
        return None

    # Extract only required fields and log the values
    filtered = {
        'height': match_item['height'],
        'width': match_item['width'],
        'rotation': match_item.get('rigidrotation', 0),
        'jp2_path_fragment': match_item['jp2Path'],
        'position_index': match_item['position_index']
    }

    logger.info(
        f"Successfully retrieved MRI metadata for biosample {biosample_id}, section {section_number}: {filtered}")

    return filtered


def get_transformation_data(biosample_id, section_number_str):
    json_filename = os.path.join(
        settings.BASE_DIR, 'brainviewer', 'data', f"{biosample_id}_original_to_bfiw.json")
    logger.info(
        f"Fetching transformation data from: {json_filename} for section key: {section_number_str}.jpg")
    try:
        json_filepath = json_filename

        if not os.path.exists(json_filepath):
            logger.error(
                f"JSON file not found at path: {os.path.abspath(json_filepath)}")
            return None

        with open(json_filepath, 'r') as f:
            all_transform_data = json.load(f)

        section_key = f"{section_number_str}.jpg"
        if section_key in all_transform_data:
            section_data = all_transform_data[section_key]
            logger.debug(
                f"Raw section data from JSON for {section_key}: {section_data}")
            if section_data.get("status") == "success":
                h_matrix = section_data.get("H_canvas_to_bfw")
                bfi_dims = section_data.get("bfw_dims_original")

                if not h_matrix:
                    logger.warning(
                        f"H_canvas_to_bfw (H_jp2_to_bfi) is missing in JSON for {section_key}")
                if not bfi_dims or not isinstance(bfi_dims, list) or len(bfi_dims) < 2:
                    logger.warning(
                        f"bfw_dims_original (bfi_natural_dims) is missing or invalid (not a list of 2+) in JSON for {section_key}")

                transform_details = {
                    "H_jp2_to_bfi": h_matrix,
                    "bfi_natural_dims": bfi_dims,
                }
                logger.info(
                    f"Processed transformation data for {section_key}: {transform_details}")
                return transform_details
            else:
                logger.warning(
                    f"Transformation status not 'success' ({section_data.get('status')}) for {section_key} in {json_filename}")
        else:
            logger.warning(
                f"Section key {section_key} not found in {json_filename}")
    except FileNotFoundError:
        logger.error(f"JSON file not found: {json_filename}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {json_filename}")
    except Exception as e:
        logger.error(
            f"Error reading transformation data from {json_filename}: {e}")
    return None


def get_available_biosample_ids():
    return ["222", "244"]


def get_available_slice_numbers_for_biosample(biosample_id):
    json_filename = os.path.join(
        settings.BASE_DIR, 'brainviewer', 'data', f"{biosample_id}_original_to_bfiw.json")
    slice_numbers = []
    try:
        json_filepath = json_filename
        if not os.path.exists(json_filepath):
            logger.warning(
                f"Slice number source file not found: {json_filepath} for biosample_id {biosample_id}")
            return []

        with open(json_filepath, 'r') as f:
            all_transform_data = json.load(f)

        for key in all_transform_data.keys():
            match = re.match(r"(\d+)\.jpg$", key)
            if match:
                slice_num_str = match.group(1)
                slice_numbers.append(slice_num_str)
            else:
                logger.debug(
                    f"Key '{key}' in {json_filename} does not match expected slice format (e.g., '123.jpg').")
        slice_numbers.sort(key=int)
        logger.info(
            f"Found available slice numbers for biosample {biosample_id}: {slice_numbers}")
    except FileNotFoundError:
        logger.error(
            f"JSON file for slice numbers not found: {json_filename}")
    except json.JSONDecodeError:
        logger.error(
            f"Error decoding JSON for slice numbers from {json_filename}")
    except Exception as e:
        logger.error(
            f"Error getting available slice numbers for {biosample_id}: {e}")
    return slice_numbers


def home_view(request):
    default_biosample_id = get_available_biosample_ids(
    )[0] if get_available_biosample_ids() else None
    if default_biosample_id:
        default_slices = get_available_slice_numbers_for_biosample(
            default_biosample_id)
        default_slice_number = default_slices[0] if default_slices else None
        if default_slice_number:
            # CORRECTED PARAMETER NAME HERE
            return redirect('brainviewer:viewer', biosample_id=default_biosample_id, slice_number_str=default_slice_number)

    logger.warning(
        "Could not determine a default slice to redirect to. Serving simple message.")
    return "Welcome to the Brain Slice Viewer. No default slice configured or available."


def viewer_view(request, biosample_id, slice_number_str, port_no=10803):
    logger.info(
        f"Received request for /viewer/{biosample_id}/{slice_number_str}")

    try:
        slice_number_int = int(slice_number_str)
    except ValueError:
        logger.error(
            f"'slice_number' in URL must be an integer, got: {slice_number_str}")
        raise Http404(
            f"'slice_number' ({slice_number_str}) must be an integer.")

    jp2_meta = get_jp2_metadata(biosample_id, slice_number_int)
    transform_data = get_transformation_data(biosample_id, slice_number_str)
    if not transform_data:
        raise Http404(
            f"Transformation Data not found for biosample {biosample_id}, slice {slice_number_str}.")

    if not jp2_meta:
        logger.error(
            f"Failed to get JP2 metadata for biosample {biosample_id}, slice {slice_number_str}.")
        raise Http404(
            f"JP2 Data not found for biosample {biosample_id}, slice {slice_number_str}.")
    if not transform_data:
        logger.error(
            f"Failed to get transformation data for biosample {biosample_id}, slice {slice_number_str}.")
        raise Http404(
            f"Transformation Data not found for biosample {biosample_id}, slice {slice_number_str}.")

    h_jp2_to_bfi = transform_data.get("H_jp2_to_bfi")
    bfi_natural_dims = transform_data.get("bfi_natural_dims")

    if not h_jp2_to_bfi:
        logger.error(
            f"H_jp2_to_bfi matrix is missing from transformation data for biosample {biosample_id}, slice {slice_number_str}.")
        raise Http404(
            f"Critical transformation matrix H_jp2_to_bfi is missing for slice {slice_number_str}.")
    if not bfi_natural_dims or not isinstance(bfi_natural_dims, list) or len(bfi_natural_dims) < 2:
        logger.error(
            f"BFI natural dimensions are missing or invalid from transformation data for biosample {biosample_id}, slice {slice_number_str}.")
        raise Http404(
            f"Critical BFI natural dimensions are missing or invalid for slice {slice_number_str}.")

    jp2_base_url = "https://apollo2.humanbrain.in/iipsrv/fcgi-bin/iipsrv.fcgi?FIF=/"
    jp2_path = jp2_meta['jp2_path_fragment']
    if jp2_path.startswith('/'):
        jp2_path = jp2_path[1:]

    jp2_suffix = "&WID=1024&GAM=1.4&MINMAX=1:0,255&MINMAX=2:0,255&MINMAX=3:0,255&JTL={z},{tileIndex}"
    full_jp2_url = f"{jp2_base_url}{jp2_path}{jp2_suffix}"

    # bfi_image_url = f"/static/images_data/{biosample_id}/bfi-{slice_number_str}.png"
    # bfi_image_url = f"images_data/{biosample_id}/bfi-{slice_number_str}.png"   # this is not working
    # bfi_image_url = f"http://dgx3.humanbrain.in:10803/images/222/bfi-763.png"
    bfi_image_url = f"http://dgx3.humanbrain.in:{port_no}/images/{biosample_id}/bfi-{slice_number_str}.png"
    logger.info(f"Generated BFI Image URL: {bfi_image_url}")

    available_b_ids = get_available_biosample_ids()
    available_s_nums = get_available_slice_numbers_for_biosample(biosample_id)

    if slice_number_str not in available_s_nums and available_s_nums:
        logger.warning(
            f"Current slice {slice_number_str} for biosample {biosample_id} is not in the dynamically generated available_slice_numbers list: {available_s_nums}. The dropdown might not pre-select it correctly if it's missing from the JSON keys.")

    bfi_w = bfi_natural_dims[0] if bfi_natural_dims and len(
        bfi_natural_dims) > 0 else 0
    bfi_h = bfi_natural_dims[1] if bfi_natural_dims and len(
        bfi_natural_dims) > 1 else 0
    print("BFI image url:", json.dumps(bfi_image_url))
    template_data = {
        "jp2_map_url": json.dumps(full_jp2_url),
        "jp2_full_size": json.dumps([jp2_meta["width"], jp2_meta["height"]]),
        "jp2_initial_view_rotation_deg": json.dumps(jp2_meta["rotation"]),
        "bfi_image_url": json.dumps(bfi_image_url),
        "bfi_css_rotation_deg": json.dumps(jp2_meta["rotation"]),
        "h_jp2_to_bfi": json.dumps(h_jp2_to_bfi),
        "bfi_natural_width": json.dumps(bfi_w),
        "bfi_natural_height": json.dumps(bfi_h),
        "test_points_jp2": [],
        "title": f"Slice Viewer - BSID {biosample_id}, Slice {slice_number_str}",
        "current_biosample_id": json.dumps(biosample_id),
        "current_slice_number": json.dumps(slice_number_str),
        "available_biosample_ids": json.dumps(available_b_ids),
        "available_slice_numbers": json.dumps(available_s_nums),
        # CORRECTED PARAMETER NAME HERE
        "viewer_url_template": json.dumps(f"/viewer/__BID__/__SID__/")
    }
    logger.debug(
        f"Data being passed to template for /viewer/{biosample_id}/{slice_number_str}: {template_data}")

    return render(request, 'brainviewer/viewer.html', template_data)


def viewer_view_split(request, biosample_id, slice_number_str, port_no=10803):
    logger.info(
        f"Received request for /viewer/{biosample_id}/{slice_number_str}")

    try:
        slice_number_int = int(slice_number_str)
    except ValueError:
        logger.error(
            f"'slice_number' in URL must be an integer, got: {slice_number_str}")
        raise Http404(
            f"'slice_number' ({slice_number_str}) must be an integer.")
    
    if biosample_id not in get_available_biosample_ids():
        logger.warning(f"Invalid biosample_id: {biosample_id}")
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>No Brain ID Found</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    text-align: center;
                    padding: 40px;
                }}
                .error-box {{
                    background-color: #fff;
                    border: 1px solid #ccc;
                    padding: 30px;
                    display: inline-block;
                    border-radius: 10px;
                    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
                }}
                a {{
                    display: block;
                    margin-top: 20px;
                    text-decoration: none;
                    color: #007BFF;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="error-box">
                <h2>🚫 Biosample ID Not Found</h2>
                <p>The ID <strong>{biosample_id}</strong> does not exist in our records.</p>
            </div>
        </body>
        </html>
        """
        return HttpResponse(html_content)
    elif slice_number_str not in get_available_slice_numbers_for_biosample(biosample_id):
        logger.warning(f"Invalid slice_id: {slice_number_str}")
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>No Slice ID Found</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    text-align: center;
                    padding: 40px;
                }}
                .error-box {{
                    background-color: #fff;
                    border: 1px solid #ccc;
                    padding: 30px;
                    display: inline-block;
                    border-radius: 10px;
                    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
                }}
                a {{
                    display: block;
                    margin-top: 20px;
                    text-decoration: none;
                    color: #007BFF;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="error-box">
                <h2>🚫 Slice ID Not Found</h2>
                <p>The ID <strong>{slice_number_str}</strong> does not exist in our records.</p>
            </div>
        </body>
        </html>
        """
        return HttpResponse(html_content)
    else:
        jp2_meta = get_jp2_metadata(biosample_id, slice_number_int)
        hae_meta=get_haematoxylin_and_eosin_metadata(biosample_id, slice_number_int)
        mri_meta=get_mri_metadata(biosample_id, slice_number_int)
        transform_data = get_transformation_data(biosample_id, slice_number_str)
        if not transform_data:
            raise Http404(
                f"Transformation Data not found for biosample {biosample_id}, slice {slice_number_str}.")

        if not jp2_meta:
            logger.error(
                f"Failed to get JP2 metadata for biosample {biosample_id}, slice {slice_number_str}.")
            raise Http404(
                f"JP2 Data not found for biosample {biosample_id}, slice {slice_number_str}.")
        if not hae_meta:
            logger.error(
                f"Failed to get JP2 metadata for biosample {biosample_id}, slice {slice_number_str}.")
            raise Http404(
                f"JP2 Data not found for biosample {biosample_id}, slice {slice_number_str}.")
        if not mri_meta:
            logger.error(
                f"Failed to get JP2 metadata for biosample {biosample_id}, slice {slice_number_str}.")
            raise Http404(
                f"JP2 Data not found for biosample {biosample_id}, slice {slice_number_str}.")
        if not transform_data:
            logger.error(
                f"Failed to get transformation data for biosample {biosample_id}, slice {slice_number_str}.")
            raise Http404(
                f"Transformation Data not found for biosample {biosample_id}, slice {slice_number_str}.")

        h_jp2_to_bfi = transform_data.get("H_jp2_to_bfi")
        bfi_natural_dims = transform_data.get("bfi_natural_dims")

        if not h_jp2_to_bfi:
            logger.error(
                f"H_jp2_to_bfi matrix is missing from transformation data for biosample {biosample_id}, slice {slice_number_str}.")
            raise Http404(
                f"Critical transformation matrix H_jp2_to_bfi is missing for slice {slice_number_str}.")
        if not bfi_natural_dims or not isinstance(bfi_natural_dims, list) or len(bfi_natural_dims) < 2:
            logger.error(
                f"BFI natural dimensions are missing or invalid from transformation data for biosample {biosample_id}, slice {slice_number_str}.")
            raise Http404(
                f"Critical BFI natural dimensions are missing or invalid for slice {slice_number_str}.")

        base_url = "https://apollo2.humanbrain.in/iipsrv/fcgi-bin/iipsrv.fcgi?FIF=/"
        # jp2_base_url = "https://apollo2.humanbrain.in/iipsrv/fcgi-bin/iipsrv.fcgi?FIF=/"s
        jp2_path = jp2_meta['jp2_path_fragment']
        if jp2_path.startswith('/'):
            jp2_path = jp2_path[1:]
        # print(jp2_meta)
        # jp2_suffix = "&WID=1024&GAM=1.4&MINMAX=1:0,255&MINMAX=2:0,255&MINMAX=3:0,255&JTL={z},{tileIndex}"
        suffix = "&WID=1024&GAM=1.4&MINMAX=1:0,255&MINMAX=2:0,255&MINMAX=3:0,255&JTL={z},{tileIndex}"
        
        hae_path = hae_meta['jp2_path_fragment']
        if hae_path.startswith('/'):
            hae_path = hae_path[1:]
        
        mri_path = mri_meta['jp2_path_fragment']
        if mri_path.startswith('/'):
            mri_path = mri_path[1:]
        
        full_jp2_url = f"{base_url}{jp2_path}{suffix}"
        full_hae_url = f"{base_url}{hae_path}{suffix}"
        full_mri_url = f"{base_url}{mri_path}{suffix}"

        # print("jp2 meta:", jp2_meta)
        # print("haematoxylin and eosin meta:",hae_meta)
        # print("mri meta:",mri_meta)
        

        # bfi_image_url = f"/static/images_data/{biosample_id}/bfi-{slice_number_str}.png"
        # bfi_image_url = f"images_data/{biosample_id}/bfi-{slice_number_str}.png"   # this is not working
        # bfi_image_url = f"http://dgx3.humanbrain.in:10803/images/222/bfi-763.png"
        # bfi_image_url = f"http://dgx3.humanbrain.in:{port_no}/images/{biosample_id}/bfi-{slice_number_str}.png"
        bfi_image_url = f"https://apollo2.humanbrain.in/bfiViewerServer/images/{biosample_id}/bfi-{slice_number_str}.png"
        logger.info(f"Generated BFI Image URL: {bfi_image_url}")

        available_b_ids = get_available_biosample_ids()
        available_s_nums = get_available_slice_numbers_for_biosample(biosample_id)

        if slice_number_str not in available_s_nums and available_s_nums:
            logger.warning(
                f"Current slice {slice_number_str} for biosample {biosample_id} is not in the dynamically generated available_slice_numbers list: {available_s_nums}. The dropdown might not pre-select it correctly if it's missing from the JSON keys.")

        bfi_w = bfi_natural_dims[0] if bfi_natural_dims and len(
            bfi_natural_dims) > 0 else 0
        bfi_h = bfi_natural_dims[1] if bfi_natural_dims and len(
            bfi_natural_dims) > 1 else 0
        print("BFI image url:", json.dumps(bfi_image_url))
        

        
        template_data = {
            "jp2_map_url": json.dumps(full_jp2_url),
            "jp2_full_size": json.dumps([jp2_meta["width"], jp2_meta["height"]]),
            "jp2_initial_view_rotation_deg": json.dumps(jp2_meta["rotation"]),
            "bfi_image_url": json.dumps(bfi_image_url),
            "bfi_css_rotation_deg": json.dumps(jp2_meta["rotation"]),
            "h_jp2_to_bfi": json.dumps(h_jp2_to_bfi),
            "bfi_natural_width": json.dumps(bfi_w),
            "bfi_natural_height": json.dumps(bfi_h),
            "test_points_jp2": [],
            "title": f"Slice Viewer - BSID {biosample_id}, Slice {slice_number_str}",
            "current_biosample_id": json.dumps(biosample_id),
            "current_slice_number": json.dumps(slice_number_str),
            "available_biosample_ids": json.dumps(available_b_ids),
            "available_slice_numbers": json.dumps(available_s_nums),
            "mri_image_url": json.dumps(full_mri_url),
            "mri_full_size": json.dumps([mri_meta["width"], mri_meta["height"]]),
            "mri_initial_view_rotation_deg": json.dumps(mri_meta["rotation"]),
            "nissl_image_url": json.dumps(full_hae_url),
            "nissl_full_size": json.dumps([hae_meta["width"], hae_meta["height"]]),
            "nissl_initial_view_rotation_deg": json.dumps(hae_meta["rotation"]),
            "viewer_url_template": json.dumps(f"/viewer/__BID__/__SID__/")
        }
        logger.debug(
            f"Data being passed to template for /viewer/{biosample_id}/{slice_number_str}: {template_data}")

        return render(request, 'brainviewer/viewer_working_v1.html', template_data)
        # return render(request, 'brainviewer/no_complete_sync.html', template_data)
        # return render(request, 'brainviewer/test.html', template_data)
