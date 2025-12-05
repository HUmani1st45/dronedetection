
"""Video inference scaffold:
- Per-frame detection (plug your detector here)
- Tracking (see tracking.py)
- EMA/Kalman smoothing on distance per track
"""
import cv2
from src.inference.tracking import SimpleKalmanBank

def run_video(input_path:str, output_path:str=None):
    cap = cv2.VideoCapture(input_path)
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w,h))

    kbank = SimpleKalmanBank()

    while True:
        ok, frame = cap.read()
        if not ok: break

        # TODO: your detector here → dets = [(x1,y1,x2,y2,score), ...]
        dets = []

        # TODO: your tracker association here → tracks = [(track_id, det), ...]
        tracks = [(i, det) for i, det in enumerate(dets)]

        for tid, det in tracks:
            # TODO: measure distance with your distance head
            d_meas = 100.0  # placeholder
            d_smooth = kbank.update(tid, d_meas)
            if det:
                x1,y1,x2,y2,score = det
            else:
                x1,y1,x2,y2,score = (0,0,100,100,1.0)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, f"ID {tid} ~ {d_smooth:.1f} m", (x1,max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        if writer: writer.write(frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
