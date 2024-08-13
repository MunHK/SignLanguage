from PIL import ImageFont, ImageDraw, Image
from unicode import join_jamos
import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time
import pyttsx3
import speech_recognition as sr


max_num_hands = 1

# 영어 제스처와 한글 제스처 정의
gesture_en = {
    0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h',
    8:'i',9:'j',10:'k',11:'l',12:'m',13:'n',14:'o',
    15:'p',16:'q',17:'r',18:'s',19:'t',20:'u',21:'v',
    22:'w',23:'x',24:'y',25:'z',26:'space',27:'clear', 28:'backspace', 29:'change'
}

gesture_kr = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ',
    7: 'ㅇ', 8: 'ㅈ', 9: 'ㅊ', 10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ',
    14: 'ㅏ', 15: 'ㅐ', 16: 'ㅑ', 17: 'ㅒ', 18: 'ㅓ', 19: 'ㅔ', 20: 'ㅕ',
    21: 'ㅖ', 22: 'ㅗ', 23: 'ㅛ', 24: 'ㅜ', 25: 'ㅠ', 26: 'ㅡ', 27: 'ㅣ',
    28: 'ㅘ', 29: 'ㅚ', 30: 'ㅙ', 31: 'ㅝ', 32: 'ㅟ', 33: 'ㅞ', 34: 'ㅢ', 35: 'change', 36:'합치기'
}

# pyttsx3 텍스트-음성 변환 엔진 초기화
engine = pyttsx3.init()

# SpeechRecognition 초기화
recognizer = sr.Recognizer()
mic = sr.Microphone()

# MediaPipe hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

f = open('test1.txt', 'w')

file_en = np.genfromtxt('dataSet.txt', delimiter=',')
file_kr = np.genfromtxt('dataSet1.txt', delimiter=',')

angleFile_en = file_en[:, :-1]
labelFile_en = file_en[:, -1]
angle_en = angleFile_en.astype(np.float32)
label_en = labelFile_en.astype(np.float32)

angleFile_kr = file_kr[:, :-1]
labelFile_kr = file_kr[:, -1]
angle_kr = angleFile_kr.astype(np.float32)
label_kr = labelFile_kr.astype(np.float32)

knn_en = cv2.ml.KNearest_create()
knn_en.train(angle_en, cv2.ml.ROW_SAMPLE, label_en)

knn_kr = cv2.ml.KNearest_create()
knn_kr.train(angle_kr, cv2.ml.ROW_SAMPLE, label_kr)

cap = cv2.VideoCapture(0)

startTime = time.time()
prev_index = 0
sentence = ''
recognizeDelay = 1.5
is_english = True  # 초기 설정: 영어 모드

# 한글 폰트 경로 설정
font_path = 'C:\\Users\\User\\Downloads\\NanumGothic.ttf'  # 한글 폰트 파일 경로 입력

# 오버레이할 이미지 읽기
overlay_img_path = 'add.png'  # 오버레이할 이미지 파일 경로
overlay_img = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)

# 오버레이 이미지 크기 조정
overlay_img = cv2.resize(overlay_img, (120, 200))  # 원하는 크기로 조정

if overlay_img is None:
    raise FileNotFoundError(f"오버레이 이미지 파일을 찾을 수 없습니다: {overlay_img_path}")

# 오버레이 이미지를 4채널로 변환 (필요한 경우)
if overlay_img.shape[2] == 3:
    overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2BGRA)

# 버튼이 눌렸을 때 호출할 함수 정의 (음성으로 읽기)
def read_sentence():
    engine.say(sentence)
    engine.runAndWait()

# 버튼이 눌렸을 때 호출할 함수 정의 (음성을 텍스트로 변환)
def speech_to_text():
    global sentence
    with mic as source:
        print("Listening...")
        try:
            # 음성 인식에 5초 제한 시간 설정
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text = recognizer.recognize_google(audio, language="ko-KR")  # 한국어 인식 설정
            print("You said: " + text)
            sentence += text  # 변환된 텍스트를 sentence에 저장
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start")
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

# 마우스 이벤트 콜백 함수
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 버튼1 영역 확인
        if 0 < x < 60 and 0 < y < 60:
            read_sentence()
        # 버튼2 영역 확인
        elif 90 < x < 150 and 0 < y < 60:
            speech_to_text()

# 마우스 콜백 함수 설정
cv2.namedWindow("HandTracking")
cv2.setMouseCallback("HandTracking", mouse_callback)

def overlay_image(bg_img, fg_img, x, y):
    """오버레이 이미지를 배경 이미지 위에 지정된 위치에 배치"""
    bg_h, bg_w, bg_channels = bg_img.shape
    fg_h, fg_w, fg_channels = fg_img.shape

    if x + fg_w > bg_w or y + fg_h > bg_h:
        print("오버레이 이미지가 배경 이미지 범위를 벗어납니다.")
        return bg_img

    # 배경 이미지를 4채널로 변환 (필요한 경우)
    if bg_channels == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    # 오버레이를 위한 빈 배열 생성
    overlay = np.zeros((bg_h, bg_w, 4), dtype=np.uint8)
    overlay[y:y+fg_h, x:x+fg_w] = fg_img

    # 마스크와 반전 마스크 생성
    mask = overlay[:, :, 3] / 255.0
    inverse_mask = 1.0 - mask

    # 배경 이미지와 오버레이 이미지를 혼합
    for c in range(3):
        bg_img[:, :, c] = (inverse_mask * bg_img[:, :, c] + mask * overlay[:, :, c])

    return bg_img

# 비디오 해상도를 원하는 크기로 설정 (예: 1280x720)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

while True:
    ret, img = cap.read()
    if not ret:
        continue
    
    #img = cv2.resize(img, (800, 600))
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]

            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
            compareV1 = v[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17], :]
            compareV2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
            angle = np.arccos(np.einsum('nt,nt->n', compareV1, compareV2))

            angle = np.degrees(angle)
            if keyboard.is_pressed('a'):
                for num in angle:
                    num = round(num, 6)
                    f.write(str(num))
                    f.write(',')
                f.write("32.000000")  # 데이터를 저장할 gesture의 label번호
                f.write('\n')
                print("next")
            
            data = np.array([angle], dtype=np.float32)
            if is_english:
                ret, results, neighbours, dist = knn_en.findNearest(data, 3)
                index = int(results[0][0])
                gesture = gesture_en
            else:
                ret, results, neighbours, dist = knn_kr.findNearest(data, 3)
                index = int(results[0][0])
                gesture = gesture_kr

            if index in gesture.keys():
                if index != prev_index:
                    startTime = time.time()
                    prev_index = index
                else:
                    if time.time() - startTime > recognizeDelay:
                        if is_english:
                            if index == 26:  # spacing 제스처
                                sentence += ' '
                            elif index == 27:  # clear 제스처
                                sentence = ''
                            elif index == 28:
                                sentence = sentence[:-1]  # 뒤에서 한 글자를 제거합니다.
                            elif index == 29:  # change 제스처
                                if 'ㅚ' not in sentence:
                                    is_english = not is_english
                                    sentence = ''  # 모드 변경 시 문장 초기화
                                    startTime = time.time()  # 변경 후 딜레이 초기화
                                    print("Mode changed:", "English" if is_english else "Korean")
                            else:
                                sentence += gesture[index]  # 다른 제스처 입력 처리
                        else:
                            if index == 35:  # change 제스처
                                is_english = not is_english
                                sentence = ''  # 모드 변경 시 문장 초기화
                                startTime = time.time()  # 변경 후 딜레이 초기화
                                print("Mode changed:", "English" if is_english else "Korean")
                            elif index ==36:
                                merge_jamo = join_jamos(sentence)
                                sentence = merge_jamo
                                print(merge_jamo)
                            else:
                                sentence += gesture[index]  # 다른 제스처 입력 처리
                        startTime = time.time()

                # 한글 폰트 설정
                font = ImageFont.truetype(font_path, 36)

                # 한글 표시
                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)
                draw.text((int(res.landmark[0].x * img.shape[1] - 10),
                           int(res.landmark[0].y * img.shape[0] + 40)),
                          gesture[index], font=font, fill=(255, 255, 255))
                img = np.array(img_pil)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
    
    # 하얀 배경의 정사각형을 이미지에 추가합니다.
    top_left = (00, 430)  # 정사각형의 왼쪽 상단 좌표
    bottom_right = (800, 500)  # 정사각형의 오른쪽 하단 좌표
    # 네모 상자 그리기 (시작점 좌표 (x1, y1), 끝점 좌표 (x2, y2), 색상 (B, G, R), 두께)
    cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), thickness=-1)  # 하얀색 정사각형 (배경이 하얀색)

    # 비디오 프레임에 버튼 표시
    cv2.rectangle(img, (0, 0), (60, 60), (0, 0, 0), thickness=-1)  # 버튼1 표시
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, 18)  # 한글 폰트 설정
    draw.text((10, 20), '음성', font=font, fill=(255, 255, 255))  # 'Read' 텍스트 추가
    img = np.array(img_pil)

    cv2.rectangle(img, (90, 0), (150, 60), (0, 0, 0), thickness=-1)  # 버튼2 표시
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((95, 20), '텍스트', font=font, fill=(255, 255, 255))  # 'Listen' 텍스트 추가
    img = np.array(img_pil)

    # 현재 모드를 표시하기 위해 오른쪽 상단에 텍스트 추가
    mode_text = "영" if is_english else "한"
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, 36)
    draw.text((img.shape[1] - 60, 10), mode_text, font=font, fill=(255, 255, 255))
    img = np.array(img_pil)

    if keyboard.is_pressed('h'):
        merge_jamo = join_jamos(sentence)
        sentence = merge_jamo
        print(merge_jamo)

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, 36)
    draw.text((10,430),sentence,(0,0,0),font=font)
    img = np.array(img_pil)

    # 오버레이 이미지 추가 (왼쪽 아래에 작게)
    img = overlay_image(img, overlay_img, 0, img.shape[0] - overlay_img.shape[0] - 50)

    cv2.imshow('HandTracking', img)
    cv2.waitKey(5)
    if keyboard.is_pressed('b'):
        break
    
f.close()
cap.release()
cv2.destroyAllWindows()