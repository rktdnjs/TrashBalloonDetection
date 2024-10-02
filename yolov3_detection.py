import time
import cv2
import numpy as np

def detectAndDisplay(frame):
    start_time = time.time()
    img = cv2.resize(frame, None, fx=0.8, fy=0.8)
    height, width, channels = img.shape

    # 창 크기 설정
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # 탐지한 객체의 클래스 예측
    class_ids = [] # 라벨명칭을 담을 리스트 변수
    confidences = [] # 정확도를 담을 리스트 변수
    boxes = [] # 바운딩 박스의 좌표를 담을 리스트 변수

    # 인식된 여러개의 객체에 대해서
    for out in outs:
        for detection in out:
            scores = detection[5:]            # 인식 데이터의 정확도
            class_id = np.argmax(scores)      # 정확도가 가장 높은 인덱스 위치 얻기(라벨의 위치값)
            confidence = scores[class_id]     # 정확도 값 추출
            if confidence > min_confidence:   # 정확도가 min_confidence(0.5) 이상인 경우에만 처리
                # 탐지한 객체의 중앙 x, y값 추출
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                # 바운딩 박스의 실제 너비 및 높이 계산
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # 바운딩 박스의 시작 좌표 계산
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 중복된 바운딩 박스를 제거하는 코드
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_DUPLEX
    # font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "{}: {:.2f}".format(classes[class_ids[i]], confidences[i]*100)
            print(i, label)
            color = colors[i] #-- 경계 상자 컬러 설정 / 단일 생상 사용시 (255,255,255)사용(B,G,R)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    cv2.imshow("YOLO_test", img)

# Load YOLO
model_file = 'object_Detection\conf\yolov3.weights'
config_file = 'object_Detection\conf\yolov3.cfg'
net = cv2.dnn.readNet(model_file, config_file)

# Load class names
names_file = 'object_Detection\conf\coco.names'
with open(names_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

min_confidence = 0.5

# 따로 구비한 동영상 사용시 video_path 명시
# 웹캠 사용시 vedio_path = 0으로 변경 & video_path 주석 처리
# video_path = 'objectDetection\conf\sample_apple_video.mp4'
# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0) # 웹캠 사용시

# 현재 영상이 받아와지고 있는지 여부에 따라 처리
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    # q 입력시 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()