"""
Dual-drone dataset generator (AirSim)
=====================================
YOLOv8-ready version
-------------------------------------
• Creates clean RGB images (no drawn boxes)
• Writes YOLO-format labels (.txt) with normalized coordinates
• Splits scenes 80% train / 20% test automatically
• Folder structure:
      data/virtual/train/images
      data/virtual/train/labels
      data/virtual/test/images
      data/virtual/test/labels
"""

import airsim
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import time, math

# ======================= CONFIG ==========================
ROOT = Path("data/virtual")

TRAIN_IMG_DIR = ROOT / "train" / "images"
TRAIN_LBL_DIR = ROOT / "train" / "labels"
TEST_IMG_DIR  = ROOT / "test"  / "images"
TEST_LBL_DIR  = ROOT / "test"  / "labels"
DIST_PATH     = ROOT / "metadata.csv"

for p in [TRAIN_IMG_DIR, TRAIN_LBL_DIR, TEST_IMG_DIR, TEST_LBL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

VEHICLE_TARGET   = "Drone1"
VEHICLE_OBSERVER = "Observer"
CAMERA_NAME      = "0"

TARGET_ALT      = -3.0
CAM_MOUNT_POS   = airsim.Vector3r(0.8, 0.0, -0.2)
OBSERVER_SPEED  = 3.0
N_FRAMES_CAPTURE = 10
SLEEP_BETWEEN   = 0.2
IMAGE_TYPE      = airsim.ImageType.Scene
JPEG_QUALITY    = 95
DEBUG_DRAW      = False   # True => save debug images with green boxes

# ======================= CONNECT TO AIRSIM ==========================
client = airsim.MultirotorClient()
client.confirmConnection()

def try_enable(vehicle):
    try:
        client.enableApiControl(True, vehicle_name=vehicle)
    except Exception as e:
        print(f"[WARN] enableApiControl({vehicle}): {e}")

def try_arm_takeoff_go(vehicle, x=0, y=0, z=-10, speed=3):
    try:
        client.armDisarm(True, vehicle_name=vehicle)
        client.takeoffAsync(vehicle_name=vehicle).join()
        client.moveToPositionAsync(x, y, z, speed, vehicle_name=vehicle).join()
    except Exception as e:
        print(f"[WARN] init vehicle {vehicle}: {e}")

# ======================= UTILITY MATH ==========================
def quat_to_R(q):
    qw, qx, qy, qz = q.w_val, q.x_val, q.y_val, q.z_val
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
    ])

def _normalize(v): return v / (np.linalg.norm(v) + 1e-9)

def _R_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2,1] - R[1,2]) / s
        qy = (R[0,2] - R[2,0]) / s
        qz = (R[1,0] - R[0,1]) / s
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / s
            qx = 0.25 * s
            qy = (R[0,1] + R[1,0]) / s
            qz = (R[0,2] + R[2,0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / s
            qx = (R[0,1] + R[1,0]) / s
            qy = 0.25 * s
            qz = (R[1,2] + R[2,1]) / s
        else:
            s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / s
            qx = (R[0,2] + R[2,0]) / s
            qy = (R[1,2] + R[2,1]) / s
            qz = 0.25 * s
    return airsim.Quaternionr(qx, qy, qz, qw)

def aim_camera_lookat(observer, target, camera):
    obs_pose = client.simGetVehiclePose(vehicle_name=observer)
    tgt_pose = client.simGetVehiclePose(vehicle_name=target)
    R_wo = quat_to_R(obs_pose.orientation)
    t_wo = np.array([obs_pose.position.x_val, obs_pose.position.y_val, obs_pose.position.z_val])
    t_cam = t_wo + R_wo @ np.array([CAM_MOUNT_POS.x_val, CAM_MOUNT_POS.y_val, CAM_MOUNT_POS.z_val])
    t_tgt = np.array([tgt_pose.position.x_val, tgt_pose.position.y_val, tgt_pose.position.z_val])
    x_fwd = _normalize(t_tgt - t_cam)
    z_down_world = np.array([0, 0, 1])
    y_right = _normalize(np.cross(z_down_world, x_fwd))
    z_down = _normalize(np.cross(x_fwd, y_right))
    R_wc = np.column_stack((x_fwd, y_right, z_down))
    q_cam = _R_to_quat(R_wc)
    client.simSetCameraPose(camera, airsim.Pose(CAM_MOUNT_POS, q_cam), vehicle_name=observer)

def get_frame():
    resp = client.simGetImages(
        [airsim.ImageRequest(CAMERA_NAME, IMAGE_TYPE, pixels_as_float=False, compress=False)],
        vehicle_name=VEHICLE_OBSERVER,
    )
    if not resp or resp[0].width == 0 or resp[0].height == 0:
        return None, 0, 0
    w, h = resp[0].width, resp[0].height
    img_rgb = np.frombuffer(resp[0].image_data_uint8, dtype=np.uint8).reshape(h, w, 3)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), w, h

# ======================= INITIALIZE DRONES ==========================
print("🚀 Launching drones...")
try_enable(VEHICLE_TARGET)
try_enable(VEHICLE_OBSERVER)
try_arm_takeoff_go(VEHICLE_TARGET, 0, 0, TARGET_ALT, 3)
time.sleep(2)
tgt_pose = client.simGetVehiclePose(vehicle_name=VEHICLE_TARGET).position
obs_x, obs_y, obs_z = tgt_pose.x_val - 10.0, tgt_pose.y_val, tgt_pose.z_val - 3.0
try_arm_takeoff_go(VEHICLE_OBSERVER, obs_x, obs_y, obs_z, OBSERVER_SPEED)
time.sleep(2)

# ======================= DETECTION FILTER ==========================
client.simClearDetectionMeshNames(CAMERA_NAME, IMAGE_TYPE, vehicle_name=VEHICLE_OBSERVER)
client.simSetDetectionFilterRadius(CAMERA_NAME, IMAGE_TYPE, 2000.0, vehicle_name=VEHICLE_OBSERVER)
for key in ["*Drone*", "*SimpleFlight*", "*Body*", "*Rotor*"]:
    client.simAddDetectionFilterMeshName(CAMERA_NAME, IMAGE_TYPE, key, vehicle_name=VEHICLE_OBSERVER)
print("✅ Detection filter configured.")

# ======================= DATA CAPTURE LOOP ==========================
SCENE_OFFSETS = [(0,0), (40,0), (-40,0), (0,40)]
records = []

for scene_id, (ox, oy) in enumerate(SCENE_OFFSETS, start=1):
    split = "train" if scene_id <= int(len(SCENE_OFFSETS)*0.8) else "test"
    IMG_DIR = TRAIN_IMG_DIR if split == "train" else TEST_IMG_DIR
    LBL_DIR = TRAIN_LBL_DIR if split == "train" else TEST_LBL_DIR
    print(f"\n🌄 Scene {scene_id} ({split.upper()}): offset=({ox},{oy})")

    client.moveToPositionAsync(ox, oy, TARGET_ALT, 3, vehicle_name=VEHICLE_TARGET).join()
    time.sleep(1)

    tgt_pose = client.simGetVehiclePose(vehicle_name=VEHICLE_TARGET).position
    obs_x = tgt_pose.x_val - 10.0
    obs_y = tgt_pose.y_val
    obs_z = tgt_pose.z_val - 3.0
    client.moveToPositionAsync(obs_x, obs_y, obs_z, OBSERVER_SPEED,
                               vehicle_name=VEHICLE_OBSERVER).join()
    time.sleep(1)
    print(f"📸 Capturing frames for {split} split...")

    start_dist, end_dist = 10.0, 4.0
    start_alt, end_alt = -8.0, -3.0
    dist_step = (start_dist - end_dist) / (N_FRAMES_CAPTURE - 1)
    alt_step = (end_alt - start_alt) / (N_FRAMES_CAPTURE - 1)

    for i in range(N_FRAMES_CAPTURE):
        d = start_dist - i * dist_step
        z = start_alt + i * alt_step
        obs_x = tgt_pose.x_val - d
        obs_y = tgt_pose.y_val
        obs_z = z
        client.moveToPositionAsync(obs_x, obs_y, obs_z, OBSERVER_SPEED,
                                   vehicle_name=VEHICLE_OBSERVER).join()
        aim_camera_lookat(VEHICLE_OBSERVER, VEHICLE_TARGET, CAMERA_NAME)
        time.sleep(0.1)

        img_bgr, w, h = get_frame()
        if img_bgr is None:
            print("⚠️ Frame capture failed — skipping")
            continue

        detections = client.simGetDetections(CAMERA_NAME, IMAGE_TYPE, vehicle_name=VEHICLE_OBSERVER)
        drone_boxes = [det.box2D for det in (detections or [])
                       if any(k in det.name for k in ["Drone", "SimpleFlight", "Body", "Rotor"])]
        if not drone_boxes:
            print("⚠️ No detections this frame")
            continue

        x1 = min(b.min.x_val for b in drone_boxes)
        y1 = min(b.min.y_val for b in drone_boxes)
        x2 = max(b.max.x_val for b in drone_boxes)
        y2 = max(b.max.y_val for b in drone_boxes)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        label_line = f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"

        fname = f"scene{scene_id:02d}_{i:06d}.jpg"
        cv2.imwrite(str(IMG_DIR / fname), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        with open(LBL_DIR / fname.replace(".jpg", ".txt"), "w") as f:
            f.write(label_line)

        dist = math.dist([obs_x, obs_y, obs_z],
                         [tgt_pose.x_val, tgt_pose.y_val, tgt_pose.z_val])
        records.append({"scene": scene_id, "split": split,
                        "image": fname, "distance_m": round(dist, 3)})
        print(f"✅ {split} | Frame {i:02d}: {fname}")

    time.sleep(SLEEP_BETWEEN)

# ======================= CLEANUP ==========================
for name in [VEHICLE_TARGET, VEHICLE_OBSERVER]:
    try:
        client.landAsync(vehicle_name=name).join()
        client.armDisarm(False, vehicle_name=name)
        client.enableApiControl(False, vehicle_name=name)
    except Exception as e:
        print(f"[WARN] landing/cleanup {name}: {e}")

pd.DataFrame(records).to_csv(DIST_PATH, index=False)
print("\n✅ All done — dataset ready for YOLOv8 training.")
