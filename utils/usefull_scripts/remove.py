import os
import glob
from pathlib import Path
import shutil

xml_path = ""
txt_files = Path(xml_path).glob("*.txt")
path_to_cp = os.path.join("dataset_webapp/removed")
for txt_f in txt_files:
    _, for_txt_file = os.path.split(txt_f)
    txt_file, _ = os.path.splitext(for_txt_file)
    print(txt_file)

    img_copy = os.path.join("", txt_file + ".png")
    shutil.copy2(img_copy, path_to_cp)
