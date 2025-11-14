# Dual-drone dataset generator (AirSim) — using AirSim API detections for YOLO labels
# ------------------------------------------------------------------------------
# What’s new vs your version:
# - setup_airsim_detection(): auto-finds target mesh names via simListSceneObjects("*Drone1*")
# - yolo_from_box2d(): converts AirSim Box2D → normalized YOLO
# - get_yolo_labels_from_airsim(): reads simGetDetections() each frame
# - Optional fallback to geometric projection if no detections are returned
# ------------------------------------------------------------------------------

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
OBS_OFFSET = airsim.Vector3r(-5.0, 0.0, 0.0)
OBSERVER_SPEED = 3.0

N_FRAMES_CAPTURE = 30
SLEEP_BETWEEN = 0.1
IMAGE_TYPE = airsim.ImageType.Scene
JPEG_QUALITY = 95

# AirSim detector config
DETECTION_RADIUS_M = 120.0   # big enough for your sweep; units in meters (converted to cm)
DETECTION_USE_FALLBACK_GEOMETRY = True  # set False to write empty labels if no detections

# ======================= SETUP FS ==========================
for p in [IMG_DIR, LBL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ======================= CLIENT ==========================
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

# --- Cam pose helper (keeps your forward-ish camera) ---
def set_camera_forward(observer, camera):
    cam_pose = airsim.Pose(
        position_val=airsim.Vector3r(0.8, 0.0, -0.2),
        orientation_val=airsim.to_quaternion(math.radians(-2), 0, 0)
    )
    client.simSetCameraPose(camera, cam_pose, vehicle_name=observer)

# ======================= IMAGE IO ==========================
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

# ======================= AIRSIM DETECTION ==========================
def setup_airsim_detection(camera_name, image_type, vehicle_observer, target_vehicle_name):
    """Discover likely scene object names for the target and add them to the detection filter."""
    # 1) Clear any previous filters
    client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name=vehicle_observer)

    # 2) Set a generous radius (cm)
    client.simSetDetectionFilterRadius(camera_name, image_type, int(DETECTION_RADIUS_M * 100), vehicle_name=vehicle_observer)

    # 3) Try to auto-discover objects related to Drone1
    patterns = [f"*{target_vehicle_name}*", "*SimpleFlight*", "*Drone*", "*Multirotor*"]
    added = []
    for pat in patterns:
        try:
            names = client.simListSceneObjects(pat)
            for n in names:
                if n not in added:
                    client.simAddDetectionFilterMeshName(camera_name, image_type, n, vehicle_name=vehicle_observer)
                    added.append(n)
        except Exception as e:
            print(f"[WARN] simListSceneObjects({pat}): {e}")

    # If auto-discovery didn’t find anything, you can manually add class/mesh names here:
    # client.simAddDetectionFilterMeshName(camera_name, image_type, "Cylinder", vehicle_name=vehicle_observer)  # Example from Blocks
    # client.simAddDetectionFilterMeshName(camera_name, image_type, "BP_FlyingPawn_C_*", vehicle_name=vehicle_observer)

    print("🔎 AirSim detection filter configured with the following mesh names:")
    for n in added:
        print("   •", n)
    if not added:
        print("   (No auto-matches found. Consider adding a specific wildcard for your target pawn/mesh.)")

def yolo_from_box2d(box2d, w, h):
    """Convert AirSim Box2D (min/max in pixels) to normalized YOLO (cx, cy, bw, bh)."""
    x_min = max(0, min(w, box2d.min_x))
    y_min = max(0, min(h, box2d.min_y))
    x_max = max(0, min(w, box2d.max_x))
    y_max = max(0, min(h, box2d.max_y))
    bw = max(0.0, x_max - x_min)
    bh = max(0.0, y_max - y_min)
    if bw <= 1 or bh <= 1:
        return None  # too small / degenerate
    cx = x_min + bw / 2.0
    cy = y_min + bh / 2.0
    return (cx / w, cy / h, bw / w, bh / h)

def get_yolo_labels_from_airsim(camera_name, image_type, vehicle_observer, w, h):
    """Call simGetDetections and convert all target-class matches into YOLO labels.
       This returns a list of strings (each a YOLO line)."""
    lines = []
    try:
        dets = client.simGetDetections(camera_name, image_type, vehicle_name=vehicle_observer)
    except Exception as e:
        print(f"[WARN] simGetDetections(): {e}")
        dets = None

    if not dets:
        return lines

    for d in dets:
        # d.name is the matched scene object name; we’ll treat all matches as class 0 (the target drone)
        yolo = yolo_from_box2d(d.box2D, w, h)
        if yolo is None:
            continue
        cx, cy, bw, bh = yolo
        # Optional sanity clamp
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines

# ======================= OPTIONAL GEOMETRIC FALLBACK ==========================
def quat_to_R(q):
    qw, qx, qy, qz = q.w_val, q.x_val, q.y_val, q.z_val
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
    ])

def project_target_bbox(observer, target, camera, w, h, est_size_m=1.0):
    obs = client.simGetVehiclePose(vehicle_name=observer)
    cam = client.simGetCameraInfo(camera, vehicle_name=observer)
    tgt = client.simGetVehiclePose(vehicle_name=target)

    R_wo = quat_to_R(obs.orientation)
    t_wo = np.array([obs.position.x_val, obs.position.y_val, obs.position.z_val])
    R_bc = quat_to_R(cam.pose.orientation)
    t_bc = np.array([cam.pose.position.x_val, cam.pose.position.y_val, cam.pose.position.z_val])

    # Convert to a camera frame consistent with AirSim image plane
    R_wc = R_wo @ R_bc @ np.diag([1, -1, -1])
    t_wc = t_wo + R_wo @ t_bc

    Xw = np.array([tgt.position.x_val, tgt.position.y_val, tgt.position.z_val])
    Xc = R_wc.T @ (Xw - t_wc)
    xc, yc, zc = Xc
    # Convert AirSim cam axes to OpenCV-style
    xc, yc, zc = zc, -yc, -xc
    if zc >= 0:
        return None

    fx = (w * 0.5) / math.tan(math.radians(cam.fov) * 0.5)
    fy = fx
    u = fx * (xc / -zc) + w / 2
    v = fy * (yc / -zc) + h / 2
    du, dv = fx * (1.0 / -zc), fy * (1.0 / -zc)
    x1, x2 = u - du/2, u + du/2
    y1, y2 = v - dv/2, v + dv/2
    # check frame bounds
    if not ((x2 > 0) and (y2 > 0) and (x1 < w) and (y1 < h)):
        return None
    x1 = max(0, min(w, x1)); y1 = max(0, min(h, y1))
    x2 = max(0, min(w, x2)); y2 = max(0, min(h, y2))
    bw, bh = (x2-x1), (y2-y1)
    if bw < 2 or bh < 2:
        return None
    cx, cy = (x1+x2)/2, (y1+y2)/2
    return f"0 {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}"

# ======================= LAUNCH ==========================
print("🚀 Launching drones...")
try_enable(VEHICLE_TARGET)
try_enable(VEHICLE_OBSERVER)

try:
    client.armDisarm(True, vehicle_name=VEHICLE_TARGET)
    client.takeoffAsync(vehicle_name=VEHICLE_TARGET).join()
    client.moveToPositionAsync(0, 0, TARGET_ALT, 3, vehicle_name=VEHICLE_TARGET).join()
except Exception as e:
    print(f"[WARN] init vehicle {VEHICLE_TARGET}: {e}")

time.sleep(1.5)

tgt_pose = client.simGetVehiclePose(vehicle_name=VEHICLE_TARGET).position
obs_x = tgt_pose.x_val + OBS_OFFSET.x_val
obs_y = tgt_pose.y_val + OBS_OFFSET.y_val
obs_z = tgt_pose.z_val + OBS_OFFSET.z_val
try_arm_takeoff_go(VEHICLE_OBSERVER, obs_x, obs_y, obs_z, OBSERVER_SPEED)
time.sleep(1.0)

# Align camera forward and configure AirSim detection
set_camera_forward(VEHICLE_OBSERVER, CAMERA_NAME)
setup_airsim_detection(CAMERA_NAME, IMAGE_TYPE, VEHICLE_OBSERVER, VEHICLE_TARGET)
print("🎯 Observer aligned; AirSim detection configured")

# ======================= DOLLY + ALTITUDE SWEEP ==========================
records, image_rows = [], []

start_dist, end_dist = 20.0, 4.0
start_alt,  end_alt  = -8.0, -3.0
dist_step = (start_dist - end_dist) / max(N_FRAMES_CAPTURE - 1, 1)
alt_step  = (end_alt - start_alt) / max(N_FRAMES_CAPTURE - 1, 1)

tgt_pos = client.simGetVehiclePose(vehicle_name=VEHICLE_TARGET).position
tgt_orient = client.simGetVehiclePose(vehicle_name=VEHICLE_TARGET).orientation

def yaw_from_quat(q):
    siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))

target_yaw_deg = yaw_from_quat(tgt_orient)
client.rotateToYawAsync(target_yaw_deg, 5, vehicle_name=VEHICLE_OBSERVER).join()

for i in range(N_FRAMES_CAPTURE):
    d = start_dist - i * dist_step
    z = start_alt + i * alt_step

    yaw_rad = math.radians(target_yaw_deg)
    dx = d * math.cos(yaw_rad)
    dy = d * math.sin(yaw_rad)

    obs_x = tgt_pos.x_val - dx
    obs_y = tgt_pos.y_val - dy
    obs_z = z

    client.moveToPositionAsync(obs_x, obs_y, obs_z, OBSERVER_SPEED, vehicle_name=VEHICLE_OBSERVER).join()
    client.rotateToYawAsync(target_yaw_deg, 5, vehicle_name=VEHICLE_OBSERVER).join()

    img_bgr, w, h = get_frame()
    if img_bgr is None:
        time.sleep(SLEEP_BETWEEN)
        continue

    # ===== Primary: AirSim detections → YOLO labels =====
    lines = get_yolo_labels_from_airsim(CAMERA_NAME, IMAGE_TYPE, VEHICLE_OBSERVER, w, h)

    # ===== Optional fallback: geometric projection =====
    if not lines and DETECTION_USE_FALLBACK_GEOMETRY:
        proj = project_target_bbox(VEHICLE_OBSERVER, VEHICLE_TARGET, CAMERA_NAME, w, h, est_size_m=1.0)
        if proj:
            lines = [proj]

    # Visualize first label (optional)
    if lines:
        # assume single class 0; draw the first one
        cls, cx, cy, bw, bh = lines[0].split()
        cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        x1, y1 = int((cx - bw/2)*w), int((cy - bh/2)*h)
        x2, y2 = int((cx + bw/2)*w), int((cy + bh/2)*h)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0,255,0), 2)

    # distance log
    dist = math.dist([obs_x, obs_y, obs_z], [tgt_pos.x_val, tgt_pos.y_val, tgt_pos.z_val])

    # write frame + labels
    fname = f"{i:06d}.jpg"
    cv2.imwrite(str(IMG_DIR / fname), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    with open(LBL_DIR / f"{i:06d}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    records.append({"image": fname, "distance_m": round(dist, 3), "num_dets": len(lines)})
    image_rows.append({"image": fname, "width": w, "height": h, "detections": len(lines)})

    print(f"📸 Frame {i}: dist={dist:.2f}m alt={obs_z:.2f}m yaw={target_yaw_deg:.1f}°, detections={len(lines)}")
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
