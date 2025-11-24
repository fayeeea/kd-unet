import cv2
import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image

# GPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MobileNetV3 기반 DeepLabV3 모델
model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
model.to(device)
model.eval()

# 전처리
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 비디오 캡처
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 224x224로 리사이즈
    frame_resized = cv2.resize(frame, (224, 224))
    img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # 세그멘테이션 추론
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    pred = output.argmax(0).byte().cpu().numpy()

    # Human mask (COCO 클래스 15)
    human_mask = (pred == 15).astype(np.uint8)

    # 검정 배경 + Human
    result = frame_resized * human_mask[:, :, np.newaxis]

    # 화면 표시
    cv2.imshow("Human Only (Black Background)", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()