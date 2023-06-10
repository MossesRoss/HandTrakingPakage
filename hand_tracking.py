import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def track_hands():
    cap = cv2.VideoCapture(0)
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=3,
        min_detection_confidence=0.5) as hands:
        
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            results = hands.process(frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS)
            cv2.imshow('Real-Time Hand Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    
track_hands()
