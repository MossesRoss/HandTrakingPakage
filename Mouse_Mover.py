import cv2
import mediapipe as mp
import pyautogui
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

THUMB_CONTROL_SIZE = 90
MOUSE_SPEED = 1
THUMB_POSITION_THRESHOLD = 1

def move_mouse():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:

        thumb_target_x = 0
        thumb_target_y = 0

        current_x, current_y = pyautogui.position()

        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 125, 100), thickness=4),
                        mp_drawing.DrawingSpec(color=(100, 125, 0), thickness=3))

                    thumb_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                    thumb_x = int(thumb_landmark.x * SCREEN_WIDTH) 
                    thumb_y = int(thumb_landmark.y * SCREEN_HEIGHT) 

                    thumb_target_x = thumb_x
                    thumb_target_y = thumb_y

                    control_left = max(thumb_x - THUMB_CONTROL_SIZE, 0)
                    control_top = max(thumb_y - THUMB_CONTROL_SIZE, 0)
                    control_right = min(thumb_x + THUMB_CONTROL_SIZE, SCREEN_WIDTH)
                    control_bottom = min(thumb_y + THUMB_CONTROL_SIZE, SCREEN_HEIGHT)
                    
                    if math.sqrt((thumb_target_x - current_x)**2 + (thumb_target_y - current_y)**2) > THUMB_POSITION_THRESHOLD:
                        current_x = thumb_target_x
                        current_y = thumb_target_y

                    if control_left <= current_x <= control_right and \
                            control_top <= current_y <= control_bottom:
                        pyautogui.moveTo(current_x, current_y)

            cv2.imshow('Mouse mover video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
