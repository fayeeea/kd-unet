import cv2
import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MobileNetV3 기반 DeepLabV3 모델 로드
model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
model.to(device)
model.eval()
model = model.half()  # FP16로 속도 향상

# 전처리 (224x224)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# GStreamer CSI 카메라 파이프라인
gst_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("카메라를 열 수 없습니다. GStreamer 파이프라인을 확인하세요.")
    exit()

# 프레임 건너뛰기 설정
frame_skip = 15  # 5프레임마다 한 번만 세그멘테이션
frame_count = 0
seg_mask = None  # 마지막 mask 저장

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # 224x224로 리사이즈
    frame_resized = cv2.resize(frame, (224, 224))

    # 매 frame_skip 프레임마다 세그멘테이션 수행
    if frame_count % frame_skip == 0:
        img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(img).unsqueeze(0).to(device).half()  # FP16

        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        pred = output.argmax(0).byte().cpu().numpy()
        seg_mask = (pred == 15).astype(np.uint8)  # Human mask

    frame_count += 1

    # Human mask 적용, 검정 배경
    if seg_mask is not None:
        human_mask_3c = np.stack([seg_mask]*3, axis=2)
        result = (frame_resized * human_mask_3c).astype(np.uint8)
    else:
        result = np.zeros_like(frame_resized)

    cv2.imshow("Human Only (Black Background)", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()