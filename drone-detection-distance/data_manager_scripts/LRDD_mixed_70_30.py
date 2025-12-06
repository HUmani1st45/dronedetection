import os, random, shutil

# === CONFIG ===
base_dir = r"C:\Users\giorg\Downloads\drone-detection-distance-template\drone-detection-distance\data"
real_train = os.path.join(base_dir, "real_LRDD", "train")
virtual_train = os.path.join(base_dir, "virtual", "train")

# Create a new folder for the 70/30 dataset
mixed_dir = os.path.join(base_dir, "mixed_70_30")   # 👈 changed folder name
output_train = os.path.join(mixed_dir, "train")
output_val = os.path.join(mixed_dir, "val")
output_test = os.path.join(mixed_dir, "test")

# Create output dirs
for split in ["train", "val", "test"]:
    for sub in ["images", "labels"]:
        os.makedirs(os.path.join(mixed_dir, split, sub), exist_ok=True)

# === PARAMETERS ===
target_train = 1500   # total images in mixed training set
ratio_real = 0.7      # 👈 70% real
ratio_virtual = 0.3   # 👈 30% virtual

n_real = int(target_train * ratio_real)
n_virtual = target_train - n_real   # same result, keeps total consistent

# === FILE LISTS ===
def get_images(folder):
    return [f for f in os.listdir(os.path.join(folder, "images"))
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

real_imgs = get_images(real_train)
virtual_imgs = get_images(virtual_train)

random.shuffle(real_imgs)
random.shuffle(virtual_imgs)

selected_real = real_imgs[:n_real]
selected_virtual = virtual_imgs[:n_virtual]

# --- helper to copy pairs ---
def copy_pairs(files, src_folder, dest_folder, prefix):
    for f in files:
        img_src = os.path.join(src_folder, "images", f)
        lbl_src = os.path.join(src_folder, "labels", os.path.splitext(f)[0] + ".txt")

        img_dst = os.path.join(dest_folder, "images", prefix + f)
        lbl_dst = os.path.join(dest_folder, "labels", prefix + os.path.splitext(f)[0] + ".txt")

        if os.path.exists(lbl_src):
            shutil.copy(img_src, img_dst)
            shutil.copy(lbl_src, lbl_dst)

# --- build MIXED TRAIN ---
copy_pairs(selected_real, real_train, output_train, "real_")
copy_pairs(selected_virtual, virtual_train, output_train, "virt_")

# --- build VAL + TEST (REAL ONLY) ---
def copy_folder(src, dest):
    for sub in ["images", "labels"]:
        src_sub = os.path.join(src, sub)
        dest_sub = os.path.join(dest, sub)
        os.makedirs(dest_sub, exist_ok=True)
        for f in os.listdir(src_sub):
            shutil.copy(os.path.join(src_sub, f), dest_sub)

copy_folder(os.path.join(base_dir, "real_LRDD", "val"), output_val)
copy_folder(os.path.join(base_dir, "real_LRDD", "test"), output_test)

print(f"✅ Mixed 70/30 training set built: {n_real} real + {n_virtual} virtual")
print(f"✅ Validation and test sets copied from real dataset.")
