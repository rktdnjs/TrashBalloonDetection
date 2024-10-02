import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import warnings

# 경고문 무시
warnings.filterwarnings("ignore", category=FutureWarning)

# GPU 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("CUDA is available. GPU will be used.")
else:
    print("CUDA is not available. CPU will be used.")

model = YOLO('yolo_Detection/models/YOLOv9c_best_trash_balloon.pt')  # 커스텀 모델 로드

# 객체 탐지를 위한 함수 정의
def detect_objects(image, conf_threshold=0.25):
    # 모델에 이미지 입력 (YOLOv5, YOLOv9은 같은 방식으로 동작)
    results = model(image)
    
    # 탐지된 객체들 추출
    detections = results.xyxy[0]  # format: [x1, y1, x2, y2, confidence, class]
    
    # 이미지에 바운딩 박스 그리기
    for *box, conf, cls in detections:
        if conf > conf_threshold:
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image

# 이미지 파일에 대해서 객체 탐지
def detect_in_image(image_path, output_path):
    image = cv2.imread(image_path)
    result_image = detect_objects(image)
    
    # 결과 이미지 저장
    cv2.imwrite(output_path, result_image)
    print(f"Results saved to {output_path}")

# 실시간 비디오 스트림을 통한 객체 탐지
def detect_in_video(video_source=0):
    # 비디오 스트림 열기
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Unable to open video source {video_source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # 객체 탐지 수행
        result_frame = detect_objects(frame)

        # 결과 프레임을 화면에 출력
        cv2.imshow("YOLOv9 Object Detection", result_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 스트림 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()

# 이미지에서 객체 탐지
image_path = 'path_to_input_image.jpg'
output_image_path = 'path_to_output_image.jpg'
detect_in_image(image_path, output_image_path)

# 실시간 웹캠 또는 비디오에서 객체 탐지 (기본적으로 웹캠 사용, 비디오 파일 경로를 넣으면 비디오로 탐지)
video_source = 0  # 웹캠 사용하려면 0, 비디오 파일 경로를 넣을 수도 있음
detect_in_video(video_source)
