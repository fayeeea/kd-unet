import cv2
import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import time
from collections import deque

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MobileNetV3 기반 DeepLabV3 모델 로드
model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
model.to(device).eval().half()  # FP16

# 전처리
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 카메라
gst_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# FPS 계산용 deque (최근 N 프레임)
frame_times = deque(maxlen=30)  # 최근 30프레임 평균 FPS

while True:
    # 최신 frame만 가져오기
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        continue

    start_time = time.time()

    frame_resized = cv2.resize(frame, (224, 224))
    img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(img).unsqueeze(0).to(device).half()  # FP16

    # GPU inference
    with torch.no_grad():
        output = model(input_tensor)['out'][0]

    pred = output.argmax(0).byte().cpu().numpy()
    human_mask = (pred == 15).astype(np.uint8)
    result = frame_resized * human_mask[:, :, np.newaxis]

    # FPS 계산
    end_time = time.time()
    frame_times.append(end_time - start_time)
    avg_fps = 1.0 / (sum(frame_times) / len(frame_times))  # 평균 FPS

    # FPS 화면 표시
    cv2.putText(result, f"FPS: {avg_fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Human Only (Black Background)", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
