import cv2
from hand_tracker import HandTracker
import math
import pyautogui
import joblib
import time

model = joblib.load("gesture_model.pkl")

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
time.sleep(2)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()
tracker = HandTracker()

mode = "normal"

prev_distance = 0
last_zoom_time = 0

xp, yp = 0, 0
canvas = None

print("Starting main loop...")

while True:
    cap.grab()
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        continue

    if canvas is None:
        canvas = frame.copy() * 0

    frame = cv2.flip(frame, 1)
    frame, hands = tracker.find_hands(frame)

    if hands:
        hand = hands[0]

        #  ML MODE 
        landmark_list = []
        for lm in hand:
            landmark_list.extend([lm.x, lm.y])

        if len(landmark_list) == 42:
            mode = model.predict([landmark_list])[0]

        #  LANDMARKS 
        thumb = hand[4]
        index = hand[8]

        h, w, _ = frame.shape
        x1, y1 = int(thumb.x * w), int(thumb.y * h)
        x2, y2 = int(index.x * w), int(index.y * h)

        distance = math.hypot(x2 - x1, y2 - y1)
        is_pinching = distance < 30

        #  MODES 

        # NORMAL -> click
        if mode == "normal":
            if is_pinching:
                pyautogui.click()

        # DRAW -> smooth drawing
        elif mode == "draw":
            if is_pinching and distance < 40:
                if xp == 0 and yp == 0:
                    xp, yp = x2, y2

                smoothening = 5
                cx = xp + (x2 - xp) // smoothening
                cy = yp + (y2 - yp) // smoothening

                cv2.line(canvas, (xp, yp), (cx, cy), (0, 255, 0), 6)
                xp, yp = cx, cy
            else:
                xp, yp = 0, 0

        # SCROLL -> up/down
        elif mode == "scroll":
            if index.y < 0.5:
                pyautogui.scroll(30)
            else:
                pyautogui.scroll(-30)

            cv2.putText(frame, "Scrolling...", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # ZOOM    
        zoom_active = (distance > 60) and (mode != "draw") and (not is_pinching)

        if zoom_active:
            current_time = time.time()

            if prev_distance != 0 and current_time - last_zoom_time > 0.3:
                diff = distance - prev_distance

                if diff > 12:
                    pyautogui.hotkey('ctrl', '+')
                    last_zoom_time = current_time

                elif diff < -12:
                    pyautogui.hotkey('ctrl', '-')
                    last_zoom_time = current_time

            prev_distance = distance

            cv2.putText(frame, "Zoom", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        else:
            prev_distance = 0

    #  DISPLAY 
    cv2.putText(frame, f"Mode: {mode}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    frame = cv2.add(frame, canvas)

    cv2.imshow("Gesture ML", frame)

    
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        canvas = frame.copy() * 0

    if key == 27:
        break
    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()

