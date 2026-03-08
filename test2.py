import cv2
import dxcam
import win32gui
import torch
from ultralytics import YOLO
import ctypes

# ----------------------------
# LOAD MODEL
# ----------------------------
model = YOLO("superhot_model.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Device:", device)

# ----------------------------
# FIND GAME WINDOW
# ----------------------------
WINDOW_NAME = "SUPERHOT"

hwnd = win32gui.FindWindow(None, WINDOW_NAME)

if hwnd == 0:
    raise Exception("ไม่พบหน้าต่าง SUPERHOT!")

left, top, right, bottom = win32gui.GetWindowRect(hwnd)

print("Window rect:", left, top, right, bottom)

# ----------------------------
# FIX REGION (กัน error dxcam)
# ----------------------------
user32 = ctypes.windll.user32
screen_w = user32.GetSystemMetrics(0)
screen_h = user32.GetSystemMetrics(1)

right = min(right, screen_w)
bottom = min(bottom, screen_h)

region = (left, top, right, bottom)

print("Capture region:", region)

# ----------------------------
# DXCAM START
# ----------------------------
camera = dxcam.create(output_color="BGR")

camera.start(region=region, target_fps=120)

# ----------------------------
# LOOP
# ----------------------------
while True:

    frame = camera.get_latest_frame()

    if frame is None:
        continue

    frame_h, frame_w = frame.shape[:2]

    # resize เพื่อให้ detect เร็ว
    input_size = 512
    resized = cv2.resize(frame, (input_size, input_size))

    results = model(
        resized,
        imgsz=input_size,
        conf=0.4,
        verbose=False
    )

    scale_x = frame_w / input_size
    scale_y = frame_h / input_size

    for r in results:

        for box in r.boxes:

            conf = float(box.conf[0])

            if conf < 0.45:
                continue

            x1, y1, x2, y2 = box.xyxy[0]

            # scale กลับ
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            cls = int(box.cls[0])
            label = model.names[cls]

            text = f"{label} {conf:.2f}"

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            cv2.putText(
                frame,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )

    cv2.imshow("SUPERHOT AI DETECTOR", frame)

    if cv2.waitKey(1) == 27:
        break


camera.stop()

cv2.destroyAllWindows()