import time
import cv2
import numpy as np
import requests
import torch
from datetime import datetime
import asyncio
from ultralytics import YOLO
from nats.aio.client import Client as NATS

# NATS 서버 주소 설정 (NATS 서버를 배포한 호스트 머신의 공인 IP로 수정)
nats_server = "nats://(IP Address):4222"

# GPU 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("CUDA is available. GPU will be used.")
else:
    print("CUDA is not available. CPU will be used.")

# YOLOv5mu 모델 로드 (ultralytics 프레임워크 사용)
model = YOLO('yolo_Detection/models/YOLOv5mu_best_trash_balloon.pt')  # 커스텀 모델 로드

# 이미지 캡쳐 + 타임 스탬프 정보 + GPS 정보 전송 함수
# 이미지 캡쳐 + 타임 스탬프 정보 + GPS 정보 전송 함수
def send_data_to_backend(image_path, latitude, longitude, timestamp):
    with open(image_path, 'rb') as f:
        files = {'detection_image': f}
        data = {'latitude': latitude, 'longitude': longitude, 'detection_time': timestamp}
        response = requests.post('https://(API Address)', files=files, data=data)
        print(f"Response from server: {response.status_code}, {response.text}")

# 탐지된 객체 수 추적
captured_num = 0
last_capture_time = 0
CAPTURE_INTERVAL = 5  # 5초 간격으로 캡쳐

def detectAndDisplay(frame):
    global captured_num, last_capture_time
    current_time = time.time()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLOv5mu(YOLOv8과 유사함)
    results = model(frame_rgb)

    # 결과에서 NMS 적용된 바운딩 박스 가져오기
    detections = results[0].boxes

    # 바운딩 박스 그리기
    for box in detections:
        # 바운딩 박스 좌표, 신뢰도, 클래스 정보 가져오기
        xyxy = box.xyxy[0].cpu().numpy()  # 바운딩박스 좌표를 추출
        conf = box.conf[0].cpu().numpy()  # 신뢰도
        class_id = int(box.cls[0].cpu().numpy())  # 클래스 ID를 int로 변환

        if class_id == 0 and conf > 0.386:  # 클래스 ID 0는 '풍선'일 경우
            # 좌표를 정수형으로 변환하여 사용
            x1, y1, x2, y2 = map(int, xyxy)

            # 바운딩박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # label = f"balloon {conf:.2f}"
            # print(f'balloon detected at {x1}, {y1}, {x2}, {y2} with confidence {conf:.2f}')

            # 텍스트 그리기
            # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 첫 번째 줄 텍스트 ("balloon")
            label1 = "trash"
            cv2.putText(frame, label1, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 두 번째 줄 텍스트 ("place")
            label2 = f"balloon {conf:.2f}"
            cv2.putText(frame, label2, (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 5초 간격으로 이미지 캡쳐 및 백엔드 전송
            if len(detections) > 0 and current_time - last_capture_time >= CAPTURE_INTERVAL:
                captured_num += 1
                last_capture_time = current_time
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                # 수정된 frame (바운딩박스가 그려진 이미지)을 저장합니다.
                img_path = f'yolo_Detection/detection_result_image/{file_timestamp}_{captured_num}_output.jpg'

                # 바운딩 박스가 그려진 이미지를 저장
                cv2.imwrite(img_path, frame)  # 'frame'을 저장

                # 이미지를 백엔드로 전송
                send_data_to_backend(img_path, 35.1520, 126.9179, timestamp)

    # 화면 크기를 조정 (예: 640x480 크기로 변경)
    resized_frame = cv2.resize(frame, (1920, 1080))  # 원하는 크기로 조정

    # 바운딩박스가 그려진 프레임을 화면에 표시
    cv2.imshow("YOLOv5mu Trash Balloon Detection", resized_frame)

# 영상을 이용해서 객체탐지를 수행할경우 사용
# video_path = 'yolo_Detection/conf/balloon_sample_video1.mp4'
# cap = cv2.VideoCapture(video_path)

async def receive_message():
    nc = NATS()
    await nc.connect(nats_server)

    async def video_handler(msg):
        # NATS에서 수신한 영상을 numpy 배열로 변환
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(480, 640, 3)
        
        # 객체 탐지 수행
        detectAndDisplay(frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            await nc.close()
            cv2.destroyAllWindows()
            return

    await nc.subscribe("video.stream", cb=video_handler)

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        print("Subscription cancelled.")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(receive_message())
    finally:
        loop.close()
