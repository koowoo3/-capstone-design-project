
# -capstone-design-project

## 아나콘다 가상 환경 설정

1. `capstone.yaml` 파일을 사용하여 가상환경을 생성합니다:
   ```bash
   conda env create --file capstone.yaml
   ```

2. 가상환경을 활성화합니다:
   ```bash
   conda activate koo1
   ```

## 전체 과정 실행

1. `api_key` 부분에 LLM을 실행하기 위한 API 키를 입력합니다.
2. 아래 명령어를 실행하여 전체 과정을 실행합니다:
   ```bash
   python main.py
   ```

## 경로 예측 실행

### 비디오 데이터를 이용한 실행

1. `trajectory_predict.py` 파일에서 `video_path` 변수에 원하는 비디오 데이터 경로를 입력합니다.
   ```python
   video_path = 'C:\Users\kooo\Documents\VIRAT_S_010204_05_000856_000890.mp4'
   cap = cv2.VideoCapture(video_path)
   ```

2. 아래 명령어를 실행합니다:
   ```bash
   python trajectory_predict.py
   ```

### 카메라를 이용한 실행

1. `trajectory_predict.py` 파일에서 `cap` 변수를 아래와 같이 수정합니다:
   ```python
   cap = cv2.VideoCapture(0)
   ```

2. 아래 명령어를 실행합니다:
   ```bash
   python trajectory_predict.py
   ```
