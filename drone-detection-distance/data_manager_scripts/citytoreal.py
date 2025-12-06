import shutil, os
from pathlib import Path

src_root = Path(r"C:\Users\giorg\Downloads\drone-detection-distance-template\drone-detection-distance\virtual_city")
dst_root = Path(r"C:\Users\giorg\Downloads\drone-detection-distance-template\drone-detection-distance\data\virtual\train")

for split in ["train", "test"]:
    for sub in ["images", "labels"]:
        src = src_root / split / sub
        dst = dst_root / sub
        dst.mkdir(parents=True, exist_ok=True)
        for file in src.glob("*"):
            # Optionally rename to avoid overwriting
            new_name = f"{split}_{file.name}"
            shutil.copy2(file, dst / new_name)
print("✅ All virtual_city data copied into data/virtual/train/")
