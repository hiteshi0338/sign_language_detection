
import os
import cv2


actions = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

directory = 'Image/'

for action in actions:
    os.makedirs(os.path.join(directory, action), exist_ok=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Unable to access camera")
        break

    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)

    cv2.imshow("data", frame)
    cv2.imshow("ROI", frame[40:400, 0:300])

    roi = frame[40:400, 0:300]

    interrupt = cv2.waitKey(10) & 0xFF

    if interrupt != 255:
        key = chr(interrupt).upper()

        if key in actions:
            count = len(os.listdir(os.path.join(directory, key)))

            cv2.imwrite(
                os.path.join(directory, key, str(count) + '.png'),
                roi
            )

            print(f"Collected image {count + 1} for letter {key}")

        elif interrupt == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

