# SignLanguage - 손 제스처 기반 수화 인식 시스템 (2024)

실시간 웹캠을 통해 손 제스처를 인식하여 영어/한글 수화를 텍스트로 변환하는 프로젝트입니다. MediaPipe를 활용한 손 랜드마크 추출과 KNN 머신러닝 모델을 사용합니다.

## 주요 기능

### 🤚 실시간 수화 인식
- **영어 모드**: 알파벳 A-Z (26개) + 특수 제스처 (space, clear, backspace, change)
- **한글 모드**: 자음 14개 (ㄱ-ㅎ) + 모음 21개 (ㅏ-ㅢ) + 특수 제스처 (change, 합치기)
- 실시간 웹캠을 통한 손 제스처 감지 및 변환
- 모드 전환 기능 (영어 ↔ 한글)

### 🎤 음성 기능
- **TTS (Text-to-Speech)**: 인식된 텍스트를 음성으로 출력
- **STT (Speech-to-Text)**: 음성을 텍스트로 변환하여 입력

### 💡 특수 기능
- 한글 자모 조합 기능 (자음+모음 → 완성된 글자)
- 백스페이스, 스페이스, 전체 삭제 제스처
- 실시간 화면 상 텍스트 표시

## 기술 스택

### 핵심 라이브러리
- **MediaPipe**: 손 랜드마크 21개 포인트 실시간 추출
- **OpenCV**: 비디오 스트리밍 및 이미지 처리
- **KNN (K-Nearest Neighbors)**: 제스처 분류 머신러닝 모델
- **pyttsx3**: 텍스트-음성 변환
- **SpeechRecognition**: 음성-텍스트 변환

### 추가 기능
- PIL (Pillow): 한글 폰트 렌더링
- NumPy: 수치 연산 및 각도 계산
- keyboard: 키보드 입력 제어

## 시스템 구조

### 1. 손 랜드마크 추출 및 각도 계산
```
손 감지 (21개 랜드마크)
    ↓
벡터 계산 (20개 관절 벡터)
    ↓
각도 계산 (15개 관절 각도)
    ↓
KNN 모델 입력
```

### 2. 제스처 인식 프로세스
- 손가락 관절 사이의 각도를 15개 특징값으로 추출
- 사전 학습된 KNN 모델로 제스처 분류
- 1.5초 지연(recognizeDelay)으로 의도적 제스처만 인식
- 연속된 동일 제스처 필터링

### 3. 데이터셋
- `dataSet.txt`: 영어 제스처 학습 데이터 (30개 클래스)
- `dataSet1.txt`: 한글 제스처 학습 데이터 (37개 클래스)
- 각 행: 15개 각도 특징 + 레이블 (총 16개 값)

## 설치 방법

```bash
pip install opencv-python mediapipe numpy pillow pyttsx3 SpeechRecognition keyboard
```

### 필수 파일
1. 학습 데이터: `dataSet.txt`, `dataSet1.txt`
2. 한글 폰트: `NanumGothic.ttf` (경로 수정 필요)
3. 오버레이 이미지: `add.png`
4. 한글 처리 모듈: `unicode.py`

## 사용 방법

### 실행
```bash
python accuracy.py
```

### 제스처 인식
1. 웹캠 앞에서 수화 제스처 취하기
2. 1.5초 유지하면 자동 인식
3. 화면 하단에 인식된 텍스트 표시

### 특수 제스처
- **영어 모드**:
  - 제스처 26: Space (공백)
  - 제스처 27: Clear (전체 삭제)
  - 제스처 28: Backspace (한 글자 삭제)
  - 제스처 29: Change (한글 모드로 전환)

- **한글 모드**:
  - 제스처 35: Change (영어 모드로 전환)
  - 제스처 36: 합치기 (자모 조합)
  - 키보드 'h': 자모 조합

### UI 버튼
- **음성 버튼** (좌상단): 인식된 텍스트 음성 출력
- **텍스트 버튼** (좌상단): 음성을 텍스트로 변환
- **모드 표시** (우상단): "영" 또는 "한" 표시

### 종료
- 키보드 'b' 키 입력

## 데이터 수집

새로운 제스처 데이터 수집:
```python
# accuracy.py 실행 중
# 키보드 'a' 키를 누르면 현재 제스처의 각도 데이터를 test1.txt에 저장
```

## YOLOv8 학습 (선택사항)

`signlanguage.ipynb`에서 YOLOv8 객체 탐지 모델 학습:
- Roboflow 데이터셋 사용 (미국 수화 알파벳)
- 26개 클래스 (A-Z)
- 100 에폭 학습
- 이미지 크기: 416x416

```python
model.train(data='SignLanguage_Data.yaml', epochs=100, batch=32, imgsz=416)
```

## 프로젝트 구조

```
SignLanguage/
├── accuracy.py           # 메인 실행 파일
├── unicode.py           # 한글 자모 조합 모듈
├── dataSet.txt          # 영어 제스처 학습 데이터
├── dataSet1.txt         # 한글 제스처 학습 데이터
├── signlanguage.ipynb   # YOLOv8 학습 노트북
├── add.png              # UI 오버레이 이미지
├── NanumGothic.ttf      # 한글 폰트
└── README.md
```

## 주요 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| max_num_hands | 1 | 동시 인식 손 개수 |
| recognizeDelay | 1.5초 | 제스처 인식 지연 시간 |
| min_detection_confidence | 0.5 | 손 감지 신뢰도 |
| min_tracking_confidence | 0.5 | 손 추적 신뢰도 |
| KNN k 값 | 3 | 최근접 이웃 개수 |

## 한계점 및 개선 방향

- 조명 조건에 민감할 수 있음
- 손 각도에 따라 인식률 변동
- 더 많은 학습 데이터로 정확도 향상 가능
- 실시간 성능 최적화 필요

## 라이센스

이 프로젝트는 교육 목적으로 제작되었습니다.

---

**개발 환경**: Python 3.x, Windows/Linux/Mac 지원
