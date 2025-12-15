import cv2
import numpy as np
import win32gui
import mss
import threading
from queue import Queue
from ultralytics import YOLO

model = YOLO("superhot_model.pt")
model.to("cuda")
WINDOW_NAME = "SUPERHOT"

# หา window handle
hwnd = win32gui.FindWindow(None, WINDOW_NAME)
if hwnd == 0:
    raise Exception("ไม่พบหน้าต่าง SUPERHOT!")

left, top, right, bottom = win32gui.GetWindowRect(hwnd)
monitor = {
    "left": left,
    "top": top,
    "width": right - left,
    "height": bottom - top
}

print("เริ่มจับหน้าจอ:", monitor)

frame_queue = Queue(maxsize=1)
result_frame = None
running = True


# ----------------------------
# THREAD 1 - SCREEN CAPTURE
# ----------------------------
def capture_thread():
    global result_frame
    sct_local = mss.mss()

    while running:
        frame = np.array(sct_local.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        frame_resized = cv2.resize(frame, (416, 416))

        if frame_queue.empty():
            frame_queue.put(frame_resized)


# ----------------------------
# THREAD 2 - YOLO + AUTO FIX GUN CLASS
# ----------------------------
def yolo_thread():
    global result_frame

    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = model(frame, imgsz=416, stream=False)

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < 0.75:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    aspect = w / (h + 1e-6)

                    cls = int(box.cls[0])
                    label = model.names[cls]

                    frame_h, frame_w = frame.shape[:2]

                    # ==========================================================
                    # 🔥 AUTO FIX: CLASS CORRECTION SYSTEM
                    # ==========================================================

                    # -------------------------
                    # FIX Rifle ↔ Pistol
                    # -------------------------
                    if label == "Pistol":

                        # Rifle ยาวกว่า pistol
                        if aspect > 1.35:
                            label = "Rifle"

                        # FPS มุมถือปืน → ถ้าอยู่ล่างจอมาก = Rifle
                        if y1 > frame_h * 0.55:
                            label = "Rifle"

                        # rifle จริงจะกว้างกว่า
                        if w > 140:
                            label = "Rifle"

                    elif label == "Rifle":

                        # Rifle ไม่ควร aspect ต่ำ
                        if aspect < 1.15:
                            label = "Pistol"

                        # Rifle ไม่ควรสั้นแบบปืนตกพื้น
                        if h < 80:
                            label = "Pistol"

                        if w < 120:
                            label = "Pistol"

                        if aspect < 1.5:
                            label = "Pistol"


                    # -------------------------
                    # FIX Shotgun ↔ Pistol
                    # -------------------------
                    if label == "Shotgun":

                        # shotgun ตัวจริงจะสูงกว่า pistol เสมอ
                        if h < 100:
                            label = "Pistol"

                        # shotgun ต้องอ้วนพอ
                        if aspect < 1.35:
                            label = "Pistol"

                        # shotgun ต้องกว้างพอ
                        if w < 120:
                            label = "Pistol"


                    # -------------------------
                    # FIX Pistol ↔ Shotgun
                    # -------------------------
                    if label == "Pistol":

                        # pistol ไม่ควรสูงมาก (ถ้าสูง = shotgun)
                        if h > 130:
                            label = "Shotgun"

                        # pistol ไม่ควรยาว/อ้วนมาก
                        if aspect > 2.2 and h > 90:
                            label = "Shotgun"

                    # ==========================================================

                    text = f"{label} {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.rectangle(frame, (x1, y1 - 20),
                                  (x1 + len(text) * 10, y1),
                                  (0,255,0), -1)
                    cv2.putText(frame, text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,0,0), 2)

            result_frame = frame.copy()


# ----------------------------
# START THREADS
# ----------------------------
t1 = threading.Thread(target=capture_thread)
t2 = threading.Thread(target=yolo_thread)
t1.start()
t2.start()

while True:
    if result_frame is not None:
        cv2.imshow("SUPERHOT DETECTOR", result_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        running = False
        break

cv2.destroyAllWindows()
