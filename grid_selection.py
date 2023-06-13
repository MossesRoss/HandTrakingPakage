import cv2
import mediapipe as mp
import numpy as np

def paint_hand_black():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    num_areas_x = 4
    num_areas_y = 4
    area_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / num_areas_x)
    area_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / num_areas_y)

    while True:
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
            results = hands.process(image_rgb)

            x_min = 0
            x_max = 0  
            y_min = 0  
            y_max = 0  

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_min, y_min, x_max, y_max = get_hand_bbox(hand_landmarks, image.shape)
                    image[y_min:y_max, x_min:x_max] = (0, 0, 0)

        for i in range(num_areas_y):
            for j in range(num_areas_x):
                y1 = i * area_height
                y2 = (i + 1) * area_height
                x1 = j * area_width
                x2 = (j + 1) * area_width

                if x_min < x2 and x_max > x1 and y_min < y2 and y_max > y1:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Real-time grids selection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_hand_bbox(hand_landmarks, image_shape):
    x_min = image_shape[1]
    y_min = image_shape[0]
    x_max = 0
    y_max = 0

    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * image_shape[1]), int(landmark.y * image_shape[0])
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)

    return x_min, y_min, x_max, y_max


#paint_hand_black()
