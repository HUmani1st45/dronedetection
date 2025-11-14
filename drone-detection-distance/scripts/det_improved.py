"""
Dual-drone dataset generator (AirSim)
=====================================
Mesh-based version — uses AirSim's true object detection (simGetDetections).

Workflow
--------
1. Drone1 (target) hovers at fixed world position.
2. Observer spawns nearby at offset.
3. Camera auto-aims (+X axis) exactly at target.
4. Captures RGB + real mesh-based YOLO labels + distance.
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

TARGET_ALT = -4.0
OBS_OFFSET = airsim.Vector3r(-6.0, 0.0, 0.0)
CAM_MOUNT_POS = airsim.Vector3r(0.8, 0.0, -0.2)
OBSERVER_SPEED = 3.0

N_FRAMES_CAPTURE = 30
SLEEP_BETWEEN = 0.2
IMAGE_TYPE = airsim.ImageType.Scene
JPEG_QUALITY = 95
AUTO_AIM = True
EST_SIZE_M = 2.0

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

# ============ Utility math ============
def quat_to_R(q):
    qw, qx, qy, qz = q.w_val, q.x_val, q.y_val, q.z_val
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
    ])

def _normalize(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-9)

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

# ============ CAMERA AIM ============
def aim_camera_lookat(observer, target, camera):
    """Orient camera so its +X axis points exactly at the target."""
    obs_pose = client.simGetVehiclePose(vehicle_name=observer)
    tgt_pose = client.simGetVehiclePose(vehicle_name=target)

    R_wo = quat_to_R(obs_pose.orientation)
    t_wo = np.array([obs_pose.position.x_val, obs_pose.position.y_val, obs_pose.position.z_val])
    t_cam = t_wo + R_wo @ np.array([CAM_MOUNT_POS.x_val, CAM_MOUNT_POS.y_val, CAM_MOUNT_POS.z_val])

    t_tgt = np.array([tgt_pose.position.x_val, tgt_pose.position.y_val, tgt_pose.position.z_val])

    x_fwd = _normalize(t_tgt - t_cam)          # +X points to target
    z_down_world = np.array([0, 0, 1])         # keep horizon level
    y_right = _normalize(np.cross(z_down_world, x_fwd))
    z_down = _normalize(np.cross(x_fwd, y_right))

    R_wc = np.column_stack((x_fwd, y_right, z_down))
    q_cam = _R_to_quat(R_wc)

    client.simSetCameraPose(camera, airsim.Pose(CAM_MOUNT_POS, q_cam), vehicle_name=observer)

# ============ FRAME CAPTURE ============
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

aim_camera_lookat(VEHICLE_OBSERVER, VEHICLE_TARGET, CAMERA_NAME)
print(f"Observer offset: {OBS_OFFSET}")

# ======================= DETECTION FILTER SETUP ==========================
print("🛠 Configuring detection filter...")
try:
    client.simClearDetectionMeshNames(CAMERA_NAME, IMAGE_TYPE, vehicle_name=VEHICLE_OBSERVER)
    client.simSetDetectionFilterRadius(CAMERA_NAME, IMAGE_TYPE, 2000.0,
                                       vehicle_name=VEHICLE_OBSERVER)
    for key in ["*Drone*", "*SimpleFlight*", "*Body*", "*Rotor*"]:
        try:
            client.simAddDetectionFilterMeshName(CAMERA_NAME, IMAGE_TYPE, key,
                                                 vehicle_name=VEHICLE_OBSERVER)
            print("  ✅ Added mesh filter:", key)
        except Exception as e:
            print("  ⚠️ Mesh add failed:", e)
    print("✅ Detection filter configured.")
except Exception as e:
    print(f"[WARN] Could not configure detection filter: {e}")

# ======================= CAPTURE LOOP ==========================
records, image_rows = [], []
for i in range(N_FRAMES_CAPTURE):
    if AUTO_AIM:
        aim_camera_lookat(VEHICLE_OBSERVER, VEHICLE_TARGET, CAMERA_NAME)

    img_bgr, w, h = get_frame()
    if img_bgr is None:
        time.sleep(SLEEP_BETWEEN)
        continue

    # --- TRUE DETECTION ---
    lines = []
    try:
        detections = client.simGetDetections(CAMERA_NAME, IMAGE_TYPE, vehicle_name=VEHICLE_OBSERVER)
    except Exception as e:
        print(f"[WARN] simGetDetections failed: {e}")
        detections = []

    best_det = None
    best_score = 1e9
    cx, cy = w / 2.0, h / 2.0

    for det in detections or []:
        name = getattr(det, "name", "") or ""
        u = (det.box2D.min.x_val + det.box2D.max.x_val) * 0.5
        v = (det.box2D.min.y_val + det.box2D.max.y_val) * 0.5
        dcenter = (u - cx) ** 2 + (v - cy) ** 2
        score = -1.0 if "Drone1" in name else dcenter
        if score < best_score:
            best_score = score
            best_det = det

        # Debug: print all detections
        print(f"Detected: {name} | Box: ({det.box2D.min.x_val:.1f}, {det.box2D.min.y_val:.1f}) "
              f"→ ({det.box2D.max.x_val:.1f}, {det.box2D.max.y_val:.1f})")

    if best_det is not None:
        x1 = int(best_det.box2D.min.x_val)
        y1 = int(best_det.box2D.min.y_val)
        x2 = int(best_det.box2D.max.x_val)
        y2 = int(best_det.box2D.max.y_val)
        x1, x2 = max(0, min(w-1, x1)), max(0, min(w-1, x2))
        y1, y2 = max(0, min(h-1, y1)), max(0, min(h-1, y2))

        xc = ((x1 + x2) / 2.0) / w
        yc = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lbl = getattr(best_det, "name", "det")
        cv2.putText(img_bgr, lbl, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(img_bgr, "no detection", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    # --- Save + distance ---
    obs = client.simGetVehiclePose(vehicle_name=VEHICLE_OBSERVER).position
    tgt = client.simGetVehiclePose(vehicle_name=VEHICLE_TARGET).position
    dist = math.dist([obs.x_val, obs.y_val, obs.z_val],
                     [tgt.x_val, tgt.y_val, tgt.z_val])
    records.append({"image": f"{i:06d}.jpg", "distance_m": round(dist, 3)})

    cv2.imwrite(str(IMG_DIR / f"{i:06d}.jpg"), img_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    with open(LBL_DIR / f"{i:06d}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    image_rows.append({
        "image": f"{i:06d}.jpg",
        "width": w,
        "height": h,
        "detections": len(lines)
    })
    print(f"📸 Frame {i}: dist={dist:.2f}m, detections={len(lines)}")
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
