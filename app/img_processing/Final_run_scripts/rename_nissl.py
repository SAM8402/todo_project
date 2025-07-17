import os
import re
from pathlib import Path

folder = Path("/home/projects/medimg/supriti/brain-registration/142/images_data/142/142_nissl_reg/")  # Change to your folder path

for file in folder.iterdir():
    if file.is_file() and file.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif"}:
        # Extract the number after 'SE_' and before the next underscore or dot
        match = re.search(r"registered_(\d+)", file.name)
        if match:
            slice_no = match.group(1)
            new_name = "nissl-" + f"{slice_no}{file.suffix.lower()}"
            new_path = file.with_name(new_name)
            # Avoid overwriting existing files
            if not new_path.exists():
                file.rename(new_path)
                print(f"Renamed {file.name} → {new_name}")
            else:
                print(f"Skipped {file.name} (target exists)")
        else:
            print(f"Skipped {file.name} (no slice number found)")


