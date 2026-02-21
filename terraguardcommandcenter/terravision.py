import time
import threading
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify

# ---- Load config safely ----
try:
    import config  # your private config.py
except Exception:
    import config_example as config  # fallback for public repo

# ---- TFLite Interpreter ----
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


app = Flask(__name__)

RTSP_URL = config.RTSP_URL
MODEL_PATH = config.MODEL_PATH
IMG_SIZE = int(getattr(config, "IMG_SIZE", 416))
CONF_THRES = float(getattr(config, "CONF_THRES", 0.35))
IOU_THRES = float(getattr(config, "IOU_THRES", 0.45))
NUM_THREADS = int(getattr(config, "NUM_THREADS", 4))
CLASS_NAMES = list(getattr(config, "CLASS_NAMES", ["fire", "smoke"]))

# Shared state
latest_frame = None
latest_jpeg = None
latest_dets = []
latest_meta = {"ts": 0, "fps": 0.0, "source": "reolink_rtsp"}
lock = threading.Lock()

# small buffer for FPS smoothing
fps_hist = deque(maxlen=30)


def letterbox(im, new_shape=(416, 416), color=(114, 114, 114)):
    """Resize + pad image to square without distortion (YOLO-style). Returns (img, ratio, (dw, dh))."""
    shape = im.shape[:2]  # h,w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # w,h
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, (left, top)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def nms(dets, iou_thres=0.45):
    """dets: list of (x1,y1,x2,y2,conf,cls)"""
    dets = sorted(dets, key=lambda x: x[4], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if (d[5] != best[5]) or (iou_xyxy(d[:4], best[:4]) < iou_thres)]
    return keep


def build_interpreter(model_path):
    it = Interpreter(model_path=model_path, num_threads=NUM_THREADS)
    it.allocate_tensors()
    in_details = it.get_input_details()[0]
    out_details = it.get_output_details()[0]
    return it, in_details, out_details


def parse_yolo_output(raw, conf_thres=0.35):
    """
    Robust-ish YOLO TFLite head parser.
    Handles common shapes:
      - (1, N, 6)  -> [x, y, w, h, conf, cls] OR [x1,y1,x2,y2,conf,cls]
      - (1, 6, N)  -> transpose
      - (1, N, 4+num_classes) -> [cx,cy,w,h, class_scores...]
    Returns dets in normalized coords (0..1): (x1,y1,x2,y2,conf,cls)
    """
    a = np.array(raw)
    a = np.squeeze(a)

    # If shape is (6, N), transpose to (N, 6)
    if a.ndim == 2 and a.shape[0] in (6, 7, 8) and a.shape[1] > a.shape[0]:
        a = a.T

    dets = []

    if a.ndim != 2:
        return dets

    # Case: (N, 6) style
    if a.shape[1] == 6:
        for row in a:
            x, y, w, h, conf, cls = row
            if conf < conf_thres:
                continue
            cls = int(cls)
            # Heuristic: if w/h look like x2/y2 (often > x,y), treat as xyxy
            if w > x and h > y and w <= 1.2 and h <= 1.2:
                x1, y1, x2, y2 = float(x), float(y), float(w), float(h)
            else:
                # assume xywh centered
                x1, y1, x2, y2 = float(x - w / 2), float(y - h / 2), float(x + w / 2), float(y + h / 2)
            dets.append((x1, y1, x2, y2, float(conf), cls))
        return dets

    # Case: (N, 4 + C) => cx,cy,w,h + class scores
    if a.shape[1] >= 6:
        for row in a:
            cx, cy, w, h = row[:4]
            scores = row[4:]
            cls = int(np.argmax(scores))
            conf = float(scores[cls])
            if conf < conf_thres:
                continue
            x1, y1, x2, y2 = float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2)
            dets.append((x1, y1, x2, y2, conf, cls))
        return dets

    return dets


def scale_coords(dets_norm, r, pad, orig_w, orig_h):
    """Convert normalized coords on letterboxed image -> pixel coords on original frame."""
    left, top = pad
    dets_px = []
    for x1, y1, x2, y2, conf, cls in dets_norm:
        # normalized -> pixels on padded image
        x1p = x1 * IMG_SIZE
        y1p = y1 * IMG_SIZE
        x2p = x2 * IMG_SIZE
        y2p = y2 * IMG_SIZE

        # remove padding
        x1u = (x1p - left) / r
        y1u = (y1p - top) / r
        x2u = (x2p - left) / r
        y2u = (y2p - top) / r

        # clip
        x1u = float(np.clip(x1u, 0, orig_w - 1))
        y1u = float(np.clip(y1u, 0, orig_h - 1))
        x2u = float(np.clip(x2u, 0, orig_w - 1))
        y2u = float(np.clip(y2u, 0, orig_h - 1))

        dets_px.append((x1u, y1u, x2u, y2u, conf, int(cls)))
    return dets_px


def draw_dets(frame, dets_px):
    for x1, y1, x2, y2, conf, cls in dets_px:
        cls_name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls)
        label = f"{cls_name} {conf:.2f}"
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1i, max(0, y1i - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def capture_loop():
    global latest_frame
    # Use FFMPEG backend for RTSP stability
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    # reduce internal latency if supported
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # reconnect
            cap.release()
            time.sleep(0.5)
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            continue

        with lock:
            latest_frame = frame

        time.sleep(0.001)


def infer_loop():
    global latest_jpeg, latest_dets, latest_meta

    it, in_det, out_det = build_interpreter(MODEL_PATH)
    in_idx = in_det["index"]
    out_idx = out_det["index"]

    # Determine input type
    in_dtype = in_det["dtype"]

    last_t = time.time()

    while True:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.02)
            continue

        t0 = time.time()

        h, w = frame.shape[:2]
        img, r, pad = letterbox(frame, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if in_dtype == np.float32:
            inp = (img_rgb.astype(np.float32) / 255.0)[None, ...]
        else:
            inp = img_rgb.astype(in_dtype)[None, ...]

        it.set_tensor(in_idx, inp)
        it.invoke()
        raw = it.get_tensor(out_idx)

        dets_norm = parse_yolo_output(raw, CONF_THRES)
        dets_norm = nms(dets_norm, IOU_THRES)
        dets_px = scale_coords(dets_norm, r, pad, w, h)

        # draw + encode
        out = draw_dets(frame, dets_px)
        ok, jpg = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            jpg_bytes = jpg.tobytes()
        else:
            jpg_bytes = b""

        # fps
        dt = time.time() - last_t
        last_t = time.time()
        fps = 1.0 / max(dt, 1e-6)
        fps_hist.append(fps)
        fps_smooth = float(np.mean(fps_hist)) if fps_hist else fps

        # publish shared state
        payload = []
        for x1, y1, x2, y2, conf, cls in dets_px:
            payload.append({
                "cls": cls,
                "label": CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls),
                "conf": float(conf),
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            })

        with lock:
            latest_jpeg = jpg_bytes
            latest_dets = payload
            latest_meta = {"ts": time.time(), "fps": fps_smooth, "source": "reolink_rtsp", "img": IMG_SIZE}

        # small pacing to avoid pegging CPU if stream is slow
        t1 = time.time()
        elapsed = t1 - t0
        if elapsed < 0.01:
            time.sleep(0.01 - elapsed)


def mjpeg_generator():
    while True:
        with lock:
            jpg = latest_jpeg
        if jpg:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(0.03)  # ~30fps max


@app.get("/video")
def video():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/detections")
def detections():
    with lock:
        dets = list(latest_dets)
        meta = dict(latest_meta)
    return jsonify({"meta": meta, "detections": dets})


@app.get("/health")
def health():
    with lock:
        meta = dict(latest_meta)
        ok = latest_frame is not None
    return jsonify({"ok": ok, "meta": meta, "model": MODEL_PATH})


def main():
    t1 = threading.Thread(target=capture_loop, daemon=True)
    t2 = threading.Thread(target=infer_loop, daemon=True)
    t1.start()
    t2.start()

    # host=0.0.0.0 so phone/app can reach it on LAN
    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()
#KEEP IN MIND THIS FILE IS UNABLE TO RUN ON NORMAL HARDWARE AS IT IS SPECIALIZED FOR TERRAGUARD'S SENSORS 