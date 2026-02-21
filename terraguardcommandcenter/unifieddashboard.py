cat > ~/terraguard_rgb_tflite_5000.py <<'EOF'
#!/usr/bin/env python3
"""RGB RTSP + TFLite inference server on port 5000"""
import time, threading
import numpy as np
import cv2
from flask import Flask, Response, jsonify
from datetime import datetime, timezone

RTSP_URL = "rtsp://admin:Anant2009@192.168.8.135/h264Preview_01_sub"
MODEL_PATH = "/home/anantarora/Desktop/best_float16.tflite"
CONF_THRESHOLD = 0.6
LABEL_SWAP = {0: "fire", 1: "smoke"}
COLORS = {"fire": (0, 0, 255), "smoke": (180, 180, 0)}
JPEG_QUALITY = 80

raw_frame = None
raw_lock = threading.Lock()
rgb_frame = None
rgb_ok = False
rgb_fps = 0.0
last_det = None
last_boxes = []
boxes_lock = threading.Lock()
infer_ms = 0.0

interpreter = None
input_details = None
output_details = None
model_h = model_w = 0
try:
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, model_h, model_w, _ = input_details[0]['shape']
    print(f"  TFLite: {model_h}x{model_w}")
except:
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        _, model_h, model_w, _ = input_details[0]['shape']
    except Exception as e:
        print(f"  TFLite failed: {e}")

def capture_worker():
    global raw_frame, rgb_frame, rgb_ok, rgb_fps
    while True:
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            rgb_ok = False
            print("  RTSP: failed, retry 3s...")
            time.sleep(3)
            continue
        rgb_ok = True
        print("  RTSP: connected")
        last_t = time.time()
        count = 0
        skip = 0
        while True:
            ret = cap.grab()
            if not ret:
                rgb_ok = False
                break
            skip += 1
            if skip % 2 != 0:
                continue
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                continue
            count += 1
            now = time.time()
            if now - last_t >= 1.0:
                rgb_fps = count / (now - last_t)
                count = 0
                last_t = now
            with raw_lock:
                raw_frame = frame.copy()
            with boxes_lock:
                for (x1, y1, x2, y2, label, conf) in last_boxes:
                    color = COLORS.get(label, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    txt = f"{label} {conf:.2f}"
                    (tw, th2), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1-th2-8), (x1+tw, y1), color, -1)
                    cv2.putText(frame, txt, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"RTSP:OK  FPS:{rgb_fps:.1f}  MS:{infer_ms:.0f}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                rgb_frame = jpg.tobytes()
        cap.release()
        time.sleep(1)

def inference_worker():
    global last_det, last_boxes, infer_ms
    while True:
        with raw_lock:
            frame = raw_frame.copy() if raw_frame is not None else None
        if frame is None or interpreter is None:
            time.sleep(0.1)
            continue
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (model_w, model_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)
        t0 = time.time()
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        infer_ms = (time.time() - t0) * 1000
        output = interpreter.get_tensor(output_details[0]['index'])
        new_boxes = []
        best_det = None
        best_conf = 0
        try:
            preds = output[0].T if len(output.shape) == 3 else output
            for det in preds:
                if len(det) < 6: continue
                cx, cy, bw, bh = det[0], det[1], det[2], det[3]
                scores = det[4:]
                cls_id = int(np.argmax(scores))
                conf = float(scores[cls_id])
                if conf < CONF_THRESHOLD: continue
                label = LABEL_SWAP.get(cls_id, f"cls{cls_id}")
                x1 = max(0, int((cx - bw/2) * w / model_w))
                y1 = max(0, int((cy - bh/2) * h / model_h))
                x2 = min(w, int((cx + bw/2) * w / model_w))
                y2 = min(h, int((cy + bh/2) * h / model_h))
                new_boxes.append((x1, y1, x2, y2, label, conf))
                if conf > best_conf:
                    best_conf = conf
                    best_det = {"label": label, "conf": round(conf, 3)}
        except Exception as e:
            print(f"  Infer err: {e}")
        last_det = best_det
        with boxes_lock:
            last_boxes = new_boxes
        time.sleep(0.01)

app = Flask(__name__)

@app.get("/stream")
def stream():
    def gen():
        while True:
            if rgb_frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + rgb_frame + b"\r\n"
            time.sleep(0.033)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
def status():
    return jsonify({
        "ok": bool(rgb_ok),
        "fps": float(rgb_fps),
        "infer_ms": float(infer_ms),
        "last_det": last_det,
        "utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    })

if __name__ == "__main__":
    print(f"  RGB+TFLite on :5000")
    print(f"  RTSP: {RTSP_URL}")
    threading.Thread(target=capture_worker, daemon=True).start()
    threading.Thread(target=inference_worker, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
EOF
python3 ~/terraguard_rgb_tflite_5000.py &
sleep 3
curl -s http://127.0.0.1:5000/status