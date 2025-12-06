import os, random, shutil

# === CONFIG ===
base_dir = r"C:\Users\giorg\Downloads\drone-detection-distance-template\drone-detection-distance\data\real_LRDD"
images_all = os.path.join(base_dir, "images_all")
labels_all = os.path.join(base_dir, "labels_all")

splits = {
    "train": 1500,   # training images
    "val": 1000,     # validation images
    "test": 2000     # test images
}

# --- create output folders ---
for split in splits:
    for sub in ["images", "labels"]:
        os.makedirs(os.path.join(base_dir, split, sub), exist_ok=True)

# --- collect and shuffle all images ---
image_files = [f for f in os.listdir(images_all)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)

print(f"Found {len(image_files)} images in total")

# --- split list ---
train_files = image_files[:splits["train"]]
val_files   = image_files[splits["train"]:splits["train"] + splits["val"]]
test_files  = image_files[splits["train"] + splits["val"]:splits["train"] + splits["val"] + splits["test"]]

# --- helper to copy pairs ---
def copy_pairs(file_list, dest_folder):
    missing = 0
    copied = 0
    for f in file_list:
        img_src = os.path.join(images_all, f)
        label_name = os.path.splitext(f)[0] + ".txt"
        lbl_src = os.path.join(labels_all, label_name)

        img_dst = os.path.join(base_dir, dest_folder, "images", f)
        lbl_dst = os.path.join(base_dir, dest_folder, "labels", label_name)

        if os.path.exists(lbl_src):
            shutil.copy(img_src, img_dst)
            shutil.copy(lbl_src, lbl_dst)
            copied += 1
        else:
            missing += 1
    print(f"{dest_folder}: copied {copied}, missing labels for {missing} images")

# --- copy files ---
copy_pairs(train_files, "train")
copy_pairs(val_files, "val")
copy_pairs(test_files, "test")

print("✅ Done.")
