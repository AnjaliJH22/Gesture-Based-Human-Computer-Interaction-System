import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self):
        print("Initializing MediaPipe...")

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,   # 🔥 important (prevents freeze)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_draw = mp.solutions.drawing_utils

        print("MediaPipe initialized successfully")

    def find_hands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hands = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand = []

                for lm in hand_landmarks.landmark:
                    hand.append(lm)

                hands.append(hand)

                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

        return frame, hands