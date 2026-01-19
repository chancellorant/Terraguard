# config_example.py
RTSP_URL = "rtsp://USERNAME:PASSWORD@192.168.1.50:554/h264Preview_01_sub"
MODEL_PATH = "./models/best_float16.tflite"

IMG_SIZE = 416            # must match your exported TFLite input size
CONF_THRES = 0.35
IOU_THRES = 0.45
NUM_THREADS = 4

# class names in your model output order
CLASS_NAMES = ["fire", "smoke"]
