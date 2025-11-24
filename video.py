import cv2
import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MobileNetV3 기반 DeepLabV3 모델
model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
model.to(device)
model.eval()

# 전처리 (224x224)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# GStreamer 파이프라인 (CSI 카메라)
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

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # 224x224로 리사이즈
    frame_resized = cv2.resize(frame, (224, 224))
    img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # 세그멘테이션 추론
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    pred = output.argmax(0).byte().cpu().numpy()

    # 사람(Human)만 foreground (COCO class 15)
    human_mask = (pred == 15).astype(np.uint8)

    # 검정 배경
    result = frame_resized * human_mask[:, :, np.newaxis]

    # 화면 표시
    cv2.imshow("Human Only (Black Background)", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()