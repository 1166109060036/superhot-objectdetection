# Superhot Object Detection (YOLO)

โปรเจกต์นี้เป็นระบบ **AI Object Detection สำหรับเกม SUPERHOT**  
โดยใช้การจับภาพหน้าจอของหน้าต่างเกม (Window Capture) แล้วนำมาประมวลผลด้วย **YOLO (Ultralytics)**  
เหมาะสำหรับงานทดลอง AI, Computer Vision และ Game Analysis  

**เครื่องมือและไลบรารีที่ใช้**
Python 3.9+

OpenCV (cv2)

NumPy

PyWin32 (win32gui)

MSS (Screen Capture)

Ultralytics YOLO

threading / queue


**การติดตั้ง Python (Windows)**
ดาวน์โหลด Python จาก : https://www.python.org/downloads/

ตรวจสอบการติดตั้ง : python --version , pip --version

**การติดตั้งโปรเจกต์**
1. Clone Repository
git clone https://github.com/1166109060036/superhot-objectdetection.git
cd superhot-objectdetection
2. ติดตั้งไลบรารีที่จำเป็น
pip install opencv-python numpy pywin32 mss ultralytics

**วิธีใช้งาน AI Detection**
1. เปิดเกม SUPERHOT
2. แก้ไขชื่อหน้าต่างเกม
เปิดไฟล์ test.py แล้วแก้ไขค่าตัวแปร WINDOW_NAME
WINDOW_NAME = "ชื่อหน้าต่างของเกม"
ต้องตั้งชื่อให้ตรงกับ Title ของหน้าต่างเกมจริง 
3. รันโปรแกรม
python test.py
เมื่อรันสำเร็จ ระบบจะ:
จับภาพจากหน้าต่างเกม SUPERHOT
ส่งภาพเข้าโมเดล YOLO (superhot_model.pt)
แสดงผลการตรวจจับแบบเรียลไทม์
