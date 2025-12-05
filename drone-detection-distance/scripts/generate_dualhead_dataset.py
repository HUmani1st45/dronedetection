"""
Dual-drone dataset generator (AirSim)
=====================================
Geometric-only version — no AirSim detections required

Workflow
--------
1. Drone1 (target) hovers at origin.
2. Observer flies in a spiral (or any pattern).
3. For each step:
   - Camera auto-aims at Drone1.
   - If target is geometrically visible, start capturing frames.
   - Each frame:
       - RGB image is saved.
       - YOLO label from geometric projection (guaranteed box).
       - Distance logged.
4. Land and save logs.

Outputs
-------
dataset_orbit/
 ├── images/train/*.jpg
 ├── labels/train/*.txt
 ├── distances.csv
 └── images.csv
"""

import airsim
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import trange
import time, math

# =======================
# Config
# =======================
ROOT = Path("dataset_orbit")
IMG_DIR = ROOT / "images" / "train"
LBL_DIR = ROOT / "labels" / "train"
DIST_PATH = ROOT / "distances.csv"
IMG_CSV   = ROOT / "images.csv"

VEHICLE_TARGET   = "Drone1"
VEHICLE_OBSERVER = "Observer"
CAMERA_NAME = "0"

# Altitudes (NED -> negative is up)
TARGET_ALT   = -10.0
OBSERVER_ALT = -12.0  
RADIUS_START = 2
RADIUS_END   = 10
RADIUS_STEPS = 3
YAW_STEP_DEG = 15.0
OBSERVER_SPEED = 5.0

# Capture phase
N_FRAMES_CAPTURE = 40
SLEEP_BETWEEN = 0.05
IMAGE_TYPE = airsim.ImageType.Scene
COMPRESS = False
JPEG_QUALITY = 95

# Camera aim
PITCH_OFFSET_DEG = -10.0

# Estimated physical size of target drone (for projection)
EST_SIZE_M = 1.0

# Defaults if we can't read a frame yet
DEFAULT_W, DEFAULT_H = 640, 480

# =======================
# Setup
# =======================
for p in [IMG_DIR, LBL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

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

print("🚁 Starting target:", VEHICLE_TARGET)
try_enable(VEHICLE_TARGET)
try_arm_takeoff_go(VEHICLE_TARGET, 0, 0, TARGET_ALT, 3)

print("📸 Starting observer:", VEHICLE_OBSERVER)
try_enable(VEHICLE_OBSERVER)
try_arm_takeoff_go(VEHICLE_OBSERVER, RADIUS_START, 0, OBSERVER_ALT, 5)

time.sleep(1.0)

# =======================
# Helper functions
# =======================
def _wrap_rad(a):
    while a > math.pi:  a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def quat_to_R(q):
    qw, qx, qy, qz = q.w_val, q.x_val, q.y_val, q.z_val
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)  # robust norm
    n = n if n > 1e-12 else 1.0
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
    ])

def aim_camera_at_target(client, observer, target, camera, pitch_offset_deg):
    obs = client.simGetVehiclePose(vehicle_name=observer)
    tgt = client.simGetVehiclePose(vehicle_name=target)
    ox, oy, oz = obs.position.x_val, obs.position.y_val, obs.position.z_val
    tx, ty, tz = tgt.position.x_val, tgt.position.y_val, tgt.position.z_val
    dx, dy, dz = tx - ox, ty - oy, tz - oz
    yaw_world = math.atan2(dy, dx)
    horiz = math.hypot(dx, dy)
    pitch_world = math.atan2(-dz, max(1e-6, horiz))
    op, _, oyaw = airsim.to_eularian_angles(obs.orientation)
    yaw_local = _wrap_rad(yaw_world - oyaw)
    pitch_local = _wrap_rad(pitch_world - op + math.radians(pitch_offset_deg))
    q = airsim.to_quaternion(pitch_local, 0, yaw_local)
    cam_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), q)
    client.simSetCameraPose(camera, cam_pose, vehicle_name=observer)

def get_frame():
    resp = client.simGetImages(
        [airsim.ImageRequest(CAMERA_NAME, IMAGE_TYPE, pixels_as_float=False, compress=COMPRESS)],
        vehicle_name=VEHICLE_OBSERVER
    )
    if not resp or resp[0].width == 0 or resp[0].height == 0:
        return None, 0, 0
    w, h = resp[0].width, resp[0].height
    buf = np.frombuffer(resp[0].image_data_uint8, dtype=np.uint8)
    if COMPRESS:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None, w, h
        return img, w, h
    else:
        try:
            img_rgb = buf.reshape(h, w, 3)
        except ValueError:
            return None, w, h
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr, w, h

def get_resolution():
    img, w, h = get_frame()
    if img is None or w == 0 or h == 0:
        return DEFAULT_W, DEFAULT_H
    return w, h

def save_jpg(path, img):
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

def project_target_bbox(client, observer, target, camera, w, h, est_size_m=1.0):
    obs = client.simGetVehiclePose(vehicle_name=observer)
    cam = client.simGetCameraInfo(camera, vehicle_name=observer)
    tgt = client.simGetVehiclePose(vehicle_name=target)

    # Rotation: world→observer and camera→observer
    R_wo = quat_to_R(obs.orientation)
    t_wo = np.array([obs.position.x_val, obs.position.y_val, obs.position.z_val])
    R_bc = quat_to_R(cam.pose.orientation)
    t_bc = np.array([cam.pose.position.x_val, cam.pose.position.y_val, cam.pose.position.z_val])
    R_wc = R_wo @ R_bc
    t_wc = t_wo + R_wo @ t_bc

    # Target in camera coordinates
    Xw = np.array([tgt.position.x_val, tgt.position.y_val, tgt.position.z_val])
    Xc = R_wc.T @ (Xw - t_wc)
    xc, yc, zc = Xc

    # AirSim camera looks along –Z
    if zc >= 0:
        return None, None, None, None, False

    fx = (w * 0.5) / math.tan(math.radians(cam.fov) * 0.5)
    fy = fx
    u = fx * (xc / -zc) + w / 2
    v = fy * (yc / -zc) + h / 2
    du, dv = fx * (est_size_m / -zc), fy * (est_size_m / -zc)

    x1, x2 = u - du / 2, u + du / 2
    y1, y2 = v - dv / 2, v + dv / 2
    xc_n, yc_n = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
    bw_n, bh_n = (x2 - x1) / w, (y2 - y1) / h
    in_frame = (0 <= x1 < w) and (0 <= y1 < h) and (0 <= x2 <= w) and (0 <= y2 <= h)
    return xc_n, yc_n, bw_n, bh_n, in_frame

# =======================
# Spiral search (geometric)
# =======================
def spiral_search_for_target():
    print("🔍 Starting geometric spiral search...")
     # <-- FIX: infer from an actual frame (or fallback)

    for r in np.linspace(RADIUS_START, RADIUS_END, RADIUS_STEPS):
        w, h = get_resolution() 
        for yaw_deg in np.arange(0, 360, YAW_STEP_DEG):
            x = r * math.cos(math.radians(yaw_deg))
            y = r * math.sin(math.radians(yaw_deg))
            client.moveToPositionAsync(x, y, OBSERVER_ALT, OBSERVER_SPEED, vehicle_name=VEHICLE_OBSERVER).join()
            client.rotateToYawAsync((yaw_deg + 180) % 360, vehicle_name=VEHICLE_OBSERVER).join()
            aim_camera_at_target(client, VEHICLE_OBSERVER, VEHICLE_TARGET, CAMERA_NAME, PITCH_OFFSET_DEG)
            time.sleep(0.3)

            _, _, _, _, in_frame = project_target_bbox(
                client, VEHICLE_OBSERVER, VEHICLE_TARGET, CAMERA_NAME, w, h, EST_SIZE_M
            )
            if in_frame:
                print(f"🎯 Target visible at r={r:.1f}m yaw={yaw_deg:.1f}°")
                return True
    print("❌ Target not visible in spiral search.")
    return False

# =======================
# Capture
# =======================
records, image_rows = [], []
found = spiral_search_for_target()
if not found:
    print("⚠️ No target visible after spiral. Exiting.")
    exit()

print("🎯 Target locked. Beginning capture...")

for i in trange(N_FRAMES_CAPTURE):
    aim_camera_at_target(client, VEHICLE_OBSERVER, VEHICLE_TARGET, CAMERA_NAME, PITCH_OFFSET_DEG)
    img_bgr, w, h = get_frame()
    if img_bgr is None:
        time.sleep(SLEEP_BETWEEN)
        continue

    xc, yc, bw, bh, ok = project_target_bbox(client, VEHICLE_OBSERVER, VEHICLE_TARGET, CAMERA_NAME, w, h, EST_SIZE_M)
    lines = []
    if ok:
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        x1, y1 = int((xc - bw/2) * w), int((yc - bh/2) * h)
        x2, y2 = int((xc + bw/2) * w), int((yc + bh/2) * h)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # distance from observer to target
    obs = client.simGetVehiclePose(vehicle_name=VEHICLE_OBSERVER).position
    tgt = client.simGetVehiclePose(vehicle_name=VEHICLE_TARGET).position
    dist = math.sqrt((obs.x_val - tgt.x_val)**2 + (obs.y_val - tgt.y_val)**2 + (obs.z_val - tgt.z_val)**2)
    records.append({"image": f"{i:06d}.jpg", "distance_m": round(dist, 3)})

    save_jpg(IMG_DIR / f"{i:06d}.jpg", img_bgr)
    with open(LBL_DIR / f"{i:06d}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    image_rows.append({"image": f"{i:06d}.jpg", "width": w, "height": h, "detections": len(lines)})
    time.sleep(SLEEP_BETWEEN)

# =======================
# Cleanup
# =======================
for name in [VEHICLE_TARGET, VEHICLE_OBSERVER]:
    try:
        client.landAsync(vehicle_name=name).join()
        client.armDisarm(False, vehicle_name=name)
        client.enableApiControl(False, vehicle_name=name)
    except Exception as e:
        print(f"[WARN] landing/cleanup {name}: {e}")

pd.DataFrame(records).to_csv(DIST_PATH, index=False)
pd.DataFrame(image_rows).to_csv(IMG_CSV, index=False)

print("\n✅ Done.")
print(f"Images dir: {IMG_DIR.resolve()}")
print(f"Labels dir: {LBL_DIR.resolve()}")
print(f"Distances:  {DIST_PATH.resolve()}")
print(f"Frames CSV: {IMG_CSV.resolve()}")
