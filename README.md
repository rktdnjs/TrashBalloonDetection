# 실시간 웹캠 & 영상 객체 탐지

### YOLOv3 기반 객체탐지를 위한 cfg & coco & weights 다운로드 링크

-   [YOLOv3 cfg & weights](https://pjreddie.com/darknet/yolo/#google_vignette)
-   [YOLOv3 coco.names](https://github.com/pjreddie/darknet/tree/master/data)

### Colab & VSC Python 가상환경에서 코드 실행 방법

```
1. Yolo3v 관련 소스 파일 다운로드
- coco.names(YOLOv3가 감지할 수 있는 객체목록을 작성한 파일)
- yolov3.cfg(YOLOv3 알고리즘 관련 내용, 살펴보니 레이어 내용이었음)
- yolov3.weights(공식 YOLOv3 가중치 파일, 모델이라고 보면 됨)

2. 환경 설정 및 참조 파일 경로 설정
- coco.names & yolo3v.cfg & yolo3.weights 파일을 참조해야하는 부분 경로 제대로 설정
- ultralytics 패키지 설치
- opencv-python 패키지 설치
- deep-sort-realtime 패키지 설치(당장에 쓰지는 않지만 추후 추적까지 적용하면 사용)

3. 코드 실행
```

### VSC Python 가상환경 구축 및 YOLOv3 실시간 웹캠 & 영상 객체 탐지(세부 내용)

> **파이썬 가상환경 생성**

```
$ python -m venv (만들 가상환경 이름)
```

> **만든 가상환경 디렉터리로 이동 & 가상환경 활성화**

![image](https://github.com/user-attachments/assets/89cd0545-26ce-4ecd-ae9a-d36db6f3098d)

```
$ cd (만든 가상환경 이름)
여기서 이제
1. Ctrl + Shift + P을 눌러 Select Interpreter 선택
2. 생성한 가상환경 폴더 선택
3. 이후에 터미널을 지웠다가 다시 생성하면 가상환경이 활성화
4. (만든 가상환경 이름) << 요게 앞에 뜨면 가상환경이 활성화 된 것
```

> **만든 가상환경 비활성화 & 활성화 방법**

```
$ cd (가상환경 이름)
$ deactivate // 비활성화
$ source 가상환경이름/Scripts/activate // 활성화(Script 폴더 아래에 activate 파일이 있음)
```

> **현재 가상환경에 설치된 패키지 확인 & 타 Python 환경 패키지 목록 참고하여 패키지 설치 방법**

```
$ pip3 freeze // 현재 접속한 가상환경의 설치된 패키지 확인
$ pip3 freeze > requirements.txt // 현재 접속한 가상환경의 설치 패키지 목록을 따로 저장
$ pip3 install -r requirements.txt // 앞서 만든 설치 패키지 목록 파일을 바탕으로 현재 접속한 가상환경에 모두 그대로 설치
```

> **필요한 패키지 설치**

```
$ python -m pip install --upgrade pip // 파이썬 패키지 매니저인 pip 업데이트
$ pip3 install opencv-python // OpenCV 사용을 위해 패키지 설치
$ pip3 install ultralytics   // YOLO 사용을 위해 패키지 설치
```

> **가끔씩 맞이하는 pip 버전이 뜨지 않는 버그**

```
$ ModuleNotFoundError: No module named 'pip'
$ python -m ensurepip --upgrade // 요걸 써서 버전 업그레이드를 잘 마무리 해주도록 하자.
$ pip3 --version
$ pip --version
```

### 파이썬 가상환경 GPU 연동(중요)

> **진행방법**

```
- NVIDIA 드라이버 설치
- CUDA 설치
  - 설치한 NVIDIA 드라이버의 버전과 호환되는 CUDA 설치하기
- cuDNN 설치
  - 설치한 CUDA 버전과 호환되는 cuDNN 설치하기
- 파이썬 가상환경 설정 및 PyTorch 설치
  - 설치한 CUDA 버전과 호환되는 PyTorch 설치하기
  - 이후 가상환경에서 PyTorch 라이브러리 설치시, 일반 버전 말고 cu121과 같은 CUDA와 호환되어 GPU를 지원하는 버전으로 깔아야함.
  - ex) torch==2.4.0+cu121 / torchaudio==2.4.0+cu121 / torchvision==0.19.0+cu121(CUDA 12.1에 호환되는 PyTorch 라이브러리들)
	- GPU 사용 여부 확인
```

```python
# Pytorch가 설치 될 경우 다음 코드를 확인해보기
import torch

torch.cuda.get_device_name() # CUDA를 실행하고 있는 기기 이름을 나타낸다.
torch.cuda.is_available() # CUDA의 활성 여부를 나타낸다.
```
