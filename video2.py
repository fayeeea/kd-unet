#!/usr/bin/env python3

import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import time
from collections import deque

# model 폴더에서 import
from model import HumanSegmentation 

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ★ 1. 화면에 보여줄 크기 설정 (원하는 대로 키우세요)
w = 960
h = 540

# --------------------------
# 모델 초기화
# --------------------------
model = HumanSegmentation('light').to(device).eval()

# ★ 2. AI가 추론할 크기 (작을수록 빠름, 224 추천)
resize_size = 224

# --------------------------
# 전처리
# --------------------------
preprocess = transforms.Compose([
            transforms.ToTensor(),  # [0,255] -> [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

# --------------------------
# 카메라
# --------------------------
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
frame_times = deque(maxlen=30)

while True:
    # 최신 frame만 가져오기
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        continue

    start_time = time.time()

    # -----------------------------------------------------------
    # [핵심 변경] 이미지를 두 개로 나눕니다.
    # 1. 보기용 (크게): 960x540
    frame_display = cv2.resize(frame, (w, h))
    
    # 2. 추론용 (작게): 224x224 (작아야 AI가 빠름)
    frame_infer = cv2.resize(frame, (resize_size, resize_size))
    # -----------------------------------------------------------

    # AI 입력 준비
    img = Image.fromarray(cv2.cvtColor(frame_infer, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # GPU inference
    with torch.no_grad():
        # [224, 224] 크기의 마스크가 나옴 (0 또는 1)
        human_mask = model(input_tensor)[0].cpu().numpy()

    # -----------------------------------------------------------
    # [핵심 변경] 마스크 확대 및 합성
    # -----------------------------------------------------------
    
    # 1. 마스크 데이터 타입 안전하게 변환
    human_mask = human_mask.astype(np.uint8)

    # 2. 작은 마스크(224)를 화면 크기(960)로 뻥튀기
    # INTER_NEAREST를 써야 가장 빠르고 경계선이 흐려지지 않음
    mask_large = cv2.resize(human_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 3. 큰 이미지와 큰 마스크 합성
    # mask_large[:, :, np.newaxis]는 (H, W, 1) 형태로 차원을 늘려줌
    result = (frame_display * mask_large[:, :, np.newaxis]).astype(np.uint8)

    # FPS 계산
    end_time = time.time()
    frame_times.append(end_time - start_time)
    if len(frame_times) > 0:
        avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
    else:
        avg_fps = 0

    # FPS 화면 표시 (글자 크기도 0.8 -> 1.0으로 살짝 키움)
    cv2.putText(result, f"FPS: {avg_fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Human Only (Big Screen)", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
