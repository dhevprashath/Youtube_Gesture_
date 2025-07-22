import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

# Volume setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# Camera and state
cap = cv2.VideoCapture(0)
prev_x = None
gesture_cooldown = 0
swipe_start_time = time.time()
swipe_threshold = 60
swipe_delay = 1

# Blink detection state
blink_threshold = 0.21
blink_cooldown = 1.0
blink_timestamps = deque(maxlen=5)
last_blink_time = 0

def count_fingers(hand_landmarks):
    tips = [8, 12, 16, 20]
    fingers = []

    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def calculate_ear(landmarks, eye_indices, image_width, image_height):
    eye = [(int(landmarks[i].x * image_width), int(landmarks[i].y * image_height)) for i in eye_indices]
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    ear = (A + B) / (2.0 * C)
    return ear

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    face_results = face_mesh.process(img_rgb)

    if gesture_cooldown > 0:
        gesture_cooldown -= 1

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            fingers = count_fingers(handLms)
            cx = int(handLms.landmark[9].x * img.shape[1])
            prev_x = cx

            # Volume Control
            x1, y1 = int(handLms.landmark[4].x * img.shape[1]), int(handLms.landmark[4].y * img.shape[0])
            x2, y2 = int(handLms.landmark[8].x * img.shape[1]), int(handLms.landmark[8].y * img.shape[0])
            length = np.hypot(x2 - x1, y2 - y1)
            vol = np.interp(length, [20, 200], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)
            vol_percent = int(np.interp(length, [20, 200], [0, 100]))
            cv2.putText(img, f'Vol: {vol_percent}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if gesture_cooldown == 0:
                if fingers == [1, 0, 0, 0, 0]:
                    pyautogui.press('k')
                    print("▶️ Play")
                    gesture_cooldown = 30

                elif fingers == [1, 1, 1, 1, 1]:
                    pyautogui.press('k')
                    print("⏸️ Pause")
                    gesture_cooldown = 30

                elif fingers == [0, 1, 1, 0, 0]:
                    current_time = time.time()
                    dx = cx - prev_x if prev_x is not None else 0
                    if abs(dx) > swipe_threshold and (current_time - swipe_start_time) > swipe_delay:
                        if dx > 0:
                            pyautogui.hotkey('ctrl', 'tab')
                            print("➡️ Swiped Right → Next Tab")
                        else:
                            pyautogui.hotkey('ctrl', 'shift', 'tab')
                            print("⬅️ Swiped Left → Previous Tab")
                        swipe_start_time = current_time
                    prev_x = cx

    # Blink Detection
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]

            left_ear = calculate_ear(face_landmarks.landmark, left_eye_indices, img.shape[1], img.shape[0])
            right_ear = calculate_ear(face_landmarks.landmark, right_eye_indices, img.shape[1], img.shape[0])
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < blink_threshold:
                current_time = time.time()
                if current_time - last_blink_time > 0.1:
                    blink_timestamps.append(current_time)
                    last_blink_time = current_time

            if len(blink_timestamps) >= 2 and blink_timestamps[-1] - blink_timestamps[-2] < 0.5:
                pyautogui.hotkey('shift', 'n')
                print("👁️👁️ Double Blink → Next Video")
                blink_timestamps.clear()
                time.sleep(1)
            elif len(blink_timestamps) == 1 and time.time() - blink_timestamps[0] > blink_cooldown:
                pyautogui.press('j')  # Rewind 10 seconds as previous video mimic
                print("👁️ Single Blink → Previous Video")
                blink_timestamps.clear()

    cv2.imshow("YouTube Gesture + Blink Controller", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()



#.\venv\Scripts\Activate.ps1
#pip install opencv-python mediapipe pyautogui pycaw comtypes
#python youtube_controller.py
