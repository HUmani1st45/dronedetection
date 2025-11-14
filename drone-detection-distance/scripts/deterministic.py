"""
Dual-drone dataset generator (AirSim)
=====================================
Guaranteed visibility version — observer always faces target.

Workflow
--------
1. Target (Drone1) hovers at fixed world position.
2. Observer spawns nearby at offset and auto-faces target.
3. Captures RGB + YOLO + distance data.
"""

import airsim
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import time, math

# ======================= CONFIG ==========================
ROOT = Path("dataset_orbit")
IMG_DIR = ROOT / "images" / "train"
LBL_DIR = ROOT / "labels" / "train"
DIST_PATH = ROOT / "distances.csv"
IMG_CSV   = ROOT / "images.csv"

VEHICLE_TARGET   = "Drone1"
VEHICLE_OBSERVER = "Observer"
CAMERA_NAME = "0"

TARGET_ALT = -3.0  # Target hover altitude
OBS_OFFSET = airsim.Vector3r(-5.0, 0.0, 0.0)  # relative offset from target
OBSERVER_SPEED = 3.0

N_FRAMES_CAPTURE = 30
SLEEP_BETWEEN = 0.1
IMAGE_TYPE = airsim.ImageType.Scene
JPEG_QUALITY = 95
AUTO_AIM = True
EST_SIZE_M = 1.0

# ======================= SETUP ==========================
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

def quat_to_R(q):
    qw, qx, qy, qz = q.w_val, q.x_val, q.y_val, q.z_val
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
    ])

def aim_camera_at_target(observer, target, camera):
    """Force camera to look at target, yaw-only, level horizon."""
    obs_pose = client.simGetVehiclePose(vehicle_name=observer)
    tgt_pose = client.simGetVehiclePose(vehicle_name=target)
    dx = tgt_pose.position.x_val - obs_pose.position.x_val
    dy = tgt_pose.position.y_val - obs_pose.position.y_val
    print(f"Observer at ({obs_pose.position.x_val:.2f}, {obs_pose.position.y_val:.2f}, {obs_pose.position.z_val:.2f})")
    print(f"Target   at ({tgt_pose.position.x_val:.2f}, {tgt_pose.position.y_val:.2f}, {tgt_pose.position.z_val:.2f})")

    dx = tgt_pose.position.x_val - obs_pose.position.x_val
    dy = tgt_pose.position.y_val - obs_pose.position.y_val
    yaw_to_target = math.degrees(math.atan2(dy, dx))
    if yaw_to_target < 0:
        yaw_to_target += 360


    
    # Rotate drone yaw
    client.rotateToYawAsync(yaw_to_target, 5, vehicle_name=observer).join()
    # Keep camera forward
    cam_pose = airsim.Pose(
        airsim.Vector3r(-0.8, 0.0, -0.2),
        airsim.to_quaternion(math.radians(-2), 0, math.radians(180))
    )
    client.simSetCameraPose(camera, cam_pose, vehicle_name=observer)
    return yaw_to_target

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

def project_target_bbox(observer, target, camera, w, h, est_size_m=1.0):
    obs = client.simGetVehiclePose(vehicle_name=observer)
    cam = client.simGetCameraInfo(camera, vehicle_name=observer)
    tgt = client.simGetVehiclePose(vehicle_name=target)

    R_wo = quat_to_R(obs.orientation)
    t_wo = np.array([obs.position.x_val, obs.position.y_val, obs.position.z_val])
    R_bc = quat_to_R(cam.pose.orientation)
    t_bc = np.array([cam.pose.position.x_val, cam.pose.position.y_val, cam.pose.position.z_val])
    R_wc = R_wo @ R_bc
    t_wc = t_wo + R_wo @ t_bc

    Xw = np.array([tgt.position.x_val, tgt.position.y_val, tgt.position.z_val])
    Xc = R_wc.T @ (Xw - t_wc)
    xc, yc, zc = Xc
    if zc >= 0: return None, None, None, None, False

    fx = (w * 0.5) / math.tan(math.radians(cam.fov) * 0.5)
    fy = fx
    u = fx * (xc / -zc) + w / 2
    v = fy * (yc / -zc) + h / 2
    du, dv = fx * (est_size_m / -zc), fy * (est_size_m / -zc)
    x1, x2 = u - du/2, u + du/2
    y1, y2 = v - dv/2, v + dv/2
    xc_n, yc_n = ((x1+x2)/2)/w, ((y1+y2)/2)/h
    bw_n, bh_n = (x2-x1)/w, (y2-y1)/h
    in_frame = (0 <= x1 < w) and (0 <= y1 < h) and (0 <= x2 <= w) and (0 <= y2 <= h)
    return xc_n, yc_n, bw_n, bh_n, in_frame

# ======================= LAUNCH ==========================
print("🚀 Launching drones...")
try_enable(VEHICLE_TARGET)
try_enable(VEHICLE_OBSERVER)

try_arm_takeoff_go(VEHICLE_TARGET, 0, 0, TARGET_ALT, 3)
time.sleep(2)

tgt_pose = client.simGetVehiclePose(vehicle_name=VEHICLE_TARGET).position
obs_x = tgt_pose.x_val + OBS_OFFSET.x_val
obs_y = tgt_pose.y_val + OBS_OFFSET.y_val
obs_z = tgt_pose.z_val + OBS_OFFSET.z_val
try_arm_takeoff_go(VEHICLE_OBSERVER, obs_x, obs_y, obs_z, OBSERVER_SPEED)
time.sleep(2)

# Align at start
yaw = aim_camera_at_target(VEHICLE_OBSERVER, VEHICLE_TARGET, CAMERA_NAME)
print(f"🎯 Observer facing target at yaw {yaw:.2f}°")

# ======================= CAPTURE LOOP ==========================
records, image_rows = [], []
for i in range(N_FRAMES_CAPTURE):
    if AUTO_AIM:
        aim_camera_at_target(VEHICLE_OBSERVER, VEHICLE_TARGET, CAMERA_NAME)
    img_bgr, w, h = get_frame()
    if img_bgr is None:
        time.sleep(SLEEP_BETWEEN)
        continue
    xc, yc, bw, bh, ok = project_target_bbox(VEHICLE_OBSERVER, VEHICLE_TARGET, CAMERA_NAME, w, h, EST_SIZE_M)
    lines = []
    if ok:
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        x1, y1 = int((xc - bw/2)*w), int((yc - bh/2)*h)
        x2, y2 = int((xc + bw/2)*w), int((yc + bh/2)*h)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0,255,0), 2)

    obs = client.simGetVehiclePose(vehicle_name=VEHICLE_OBSERVER).position
    tgt = client.simGetVehiclePose(vehicle_name=VEHICLE_TARGET).position
    dist = math.dist(
        [obs.x_val, obs.y_val, obs.z_val],
        [tgt.x_val, tgt.y_val, tgt.z_val]
    )
    records.append({"image": f"{i:06d}.jpg", "distance_m": round(dist, 3)})
    cv2.imwrite(str(IMG_DIR / f"{i:06d}.jpg"), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    with open(LBL_DIR / f"{i:06d}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    image_rows.append({"image": f"{i:06d}.jpg", "width": w, "height": h, "detections": len(lines)})
    print(f"📸 Frame {i}: dist={dist:.2f}m, bbox={'OK' if ok else 'None'}")
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
pd.DataFrame(image_rows).to_csv(IMG_CSV, index=False)

print("\n✅ Done.")
print(f"Images dir: {IMG_DIR.resolve()}")
print(f"Labels dir: {LBL_DIR.resolve()}")
print(f"Distances:  {DIST_PATH.resolve()}")
print(f"Frames CSV: {IMG_CSV.resolve()}")
