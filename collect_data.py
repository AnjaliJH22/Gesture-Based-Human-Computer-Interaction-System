import cv2
from hand_tracker import HandTracker
import csv

print("Program started")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()
print("Camera initialized")

tracker = HandTracker()
print("HandTracker initialized")

file = open("gesture_data.csv", "a", newline="")
writer = csv.writer(file)

label = "normal"

print("Press:")
print("n → normal")
print("d → draw")
print("ESC → exit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)

    frame, hands = tracker.find_hands(frame)

    if hands:
        hand = hands[0]

        landmark_list = []
        for lm in hand:
            landmark_list.extend([lm.x, lm.y])

        if len(landmark_list) == 42:
            row = landmark_list + [label]
            writer.writerow(row)

            print(f"Saved: {label}")

            cv2.putText(frame, f"Recording: {label}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    key = cv2.waitKey(1) & 0xFF


    if key == ord('n'):
        label = "normal"
        print("Switched to NORMAL")

    elif key == ord('d'):
        label = "draw"
        print("Switched to DRAW")

    elif key == ord('s'):
        label = "scroll"
        print("Switched to SCROLL")

    elif key == ord('z'):
        label = "zoom"

    elif key == 27:
        print("Exiting...")
        break

    cv2.imshow("Data Collection", frame)

cap.release()
file.close()
cv2.destroyAllWindows()