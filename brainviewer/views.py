from django.shortcuts import render
from django.conf import settings

# Create your views here.
import json
import os
import re
import logging
from django.shortcuts import render, redirect
from django.http import HttpResponse, Http404
from django.conf import settings
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

# --- Database Configuration ---
MySQL_db_user = "root"
MySQL_db_password = "Health#123"
MySQL_db_host = "dev2mani.humanbrain.in"
MySQL_db_port = "3306"
MySQL_db_name = "HBA_V2"
MySQL_DATABASE_URL = f"mysql+pymysql://{MySQL_db_user}:{MySQL_db_password}@{MySQL_db_host}:{MySQL_db_port}/{MySQL_db_name}"
MySQL_engine = create_engine(MySQL_DATABASE_URL)


def MySQL_db_retriever(sql_query):
    try:
        with MySQL_engine.connect() as connection:
            result = connection.execute(text(sql_query))
            data = result.fetchall()
            return data
    except Exception as e:
        logger.error(f"Database query error: {e} for query: {sql_query}")
        return None


def get_jp2_metadata(biosample_id, section_number):
    logger.info(
        f"Fetching JP2 metadata for biosample: {biosample_id}, section: {section_number}")
    query = f"""
    SELECT sct.rigidrotation, sct.width, sct.height, sct.jp2Path
    FROM HBA_V2.seriesset ss
    JOIN HBA_V2.series s ON ss.id = s.seriesset
    JOIN HBA_V2.section sct ON s.id = sct.series
    WHERE ss.biosample = '{biosample_id}'
    AND s.seriestype = 1
    AND sct.positionindex = {section_number};
    """
    result = MySQL_db_retriever(query)
    if result and len(result) > 0:
        metadata = {
            "rotation": result[0][0] if result[0][0] is not None else 0,
            "width": result[0][1],
            "height": result[0][2],
            "jp2_path_fragment": result[0][3]
        }
        logger.info(f"JP2 Metadata found: {metadata}")
        return metadata
    logger.warning(
        f"No JP2 metadata found for biosample {biosample_id}, section {section_number}")
    return None


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
    # return ["222"]
    return ["222","244"]


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


def viewer_view_test(request, biosample_id, slice_number_str, port_no=10803):
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
        "viewer_url_template": json.dumps(f"/viewer/test/__BID__/__SID__/")
    }
    logger.debug(
        f"Data being passed to template for /viewer/{biosample_id}/{slice_number_str}: {template_data}")

    return render(request, 'brainviewer/test_viewer_v1.html', template_data)
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

    return render(request, 'brainviewer/viewer_working_v1.html', template_data)
    # return render(request, 'brainviewer/no_complete_sync.html', template_data)
    # return render(request, 'brainviewer/test.html', template_data)
