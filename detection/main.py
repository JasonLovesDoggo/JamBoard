import cv2
import mediapipe as mp
import sys
from .hands import MediaPipeHands
from .utils import calibrate, load_calibration
from .types import ShapeData


# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils


def start():
    hands_instance = MediaPipeHands()
    hands = hands_instance.hands
    mp_hands = mp.solutions.hands
    drawing_utils = hands_instance.drawing_utils

    if sys.platform == "darwin":  # Mac
        cap = cv2.VideoCapture(0)
    elif sys.platform in ["win32", "win64"]:  # Windows
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    try:
        print("done video intalization")
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        paper_roi = (100, 100, 540, 380)
        print("Press 'c' to recalibrate the shapes.")
        calibrated_shapes = load_calibration()
        if not calibrated_shapes:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting.")
                cap.release()
                cv2.destroyAllWindows()
                exit()
            calibrated_shapes = calibrate(frame, paper_roi)

        last_touched_shape = None
        touch_cooldown = 0
        COOLDOWN_FRAMES = 15  # Adjust this value to change the cooldown period

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Convert the BGR image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process the image and detect hands
            results = hands.process(image)

            # Draw the hand annotations on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw paper boundaries
            cv2.rectangle(
                image,
                (paper_roi[0], paper_roi[1]),
                (paper_roi[2], paper_roi[3]),
                (255, 0, 0),
                2,
            )

            finger_tip = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    drawing_utils.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    # Get index fingertip coordinates
                    h, w, c = image.shape
                    index_tip = hand_landmarks.landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP
                    ]
                    finger_tip = (int(index_tip.x * w), int(index_tip.y * h))
                    # print('the finger tip is currently at',finger_tip)
                    cv2.circle(image, finger_tip, 10, (0, 255, 0), cv2.FILLED)

            # shadow detection
            ret, frame = cap.read()
            touched_shape = None
            for shape_name, (cx, cy), color, area in calibrated_shapes:
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(
                    image,
                    shape_name,
                    (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            if touched_shape:
                if touch_cooldown == 0:
                    shape_name, (cx, cy), color, area = touched_shape
                    data = ShapeData(
                        size=area, color=color, center=(cx, cy), name=shape_name
                    )
                    print(f"Touched: {data}")
                    touch_cooldown = COOLDOWN_FRAMES

            if touch_cooldown > 0:
                touch_cooldown -= 1

            cv2.imshow("Shapes and Hand Detection", image)

            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord("c"):
                print("Recalibrating...")
                _, frame = cap.read()
                calibrated_shapes = calibrate(frame, paper_roi)
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting...")
    start()
