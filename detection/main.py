import cv2
import mediapipe as mp
import numpy as np
import sys
from .hands import MediaPipeHands
from .utils import calibrate, load_calibration
from .types import ShapeData

CURRENT_OBJECT: ShapeData | None = None

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

TOP_CAM = 1
SIDE_CAM = 2


def start():
    global CURRENT_OBJECT
    hands_instance = MediaPipeHands()
    hands = hands_instance.hands
    mp_hands = mp.solutions.hands
    drawing_utils = hands_instance.drawing_utils

    if sys.platform == "darwin":  # Mac
        cap_top = cv2.VideoCapture(TOP_CAM)
        cap_side = cv2.VideoCapture(SIDE_CAM)
    elif sys.platform in ["win32", "win64"]:  # Windows
        cap_top = cv2.VideoCapture(TOP_CAM, cv2.CAP_DSHOW)
        cap_side = cv2.VideoCapture(SIDE_CAM, cv2.CAP_DSHOW)
    else:
        cap_top = cv2.VideoCapture(TOP_CAM)
        cap_side = cv2.VideoCapture(SIDE_CAM)
        
    try:
        print("done video intalization")
        if not cap_top.isOpened():
            print("Error: Could not open top webcam.")
            exit()
        if not cap_side.isOpened():
            print("Error: Could not open side webcam.")
            exit()
            

        paper_roi = (100, 100, 540, 380)
        print("Press 'c' to recalibrate the shapes.")
        calibrated_shapes = load_calibration()
        if not calibrated_shapes:
            ret, frame = cap_top.read()
            if not ret:
                print("Failed to capture frame. Exiting.")
                cap_top.release()
                cv2.destroyAllWindows()
                exit()
            calibrated_shapes = calibrate(frame, paper_roi)

        while cap_top.isOpened():
            print(f'{CURRENT_OBJECT=}')
            success, image = cap_top.read()
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
                    h, w, _ = image.shape
                    index_tip = hand_landmarks.landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP
                    ]
                    finger_tip = (int(index_tip.x * w), int(index_tip.y * h))
                    # print('the finger tip is currently at',finger_tip)
                    cv2.circle(image, finger_tip, 10, (0, 255, 0), cv2.FILLED)

            # shadow detection
            ret, frame = cap_top.read()
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

            CURRENT_OBJECT = None  # Reset CURRENT_OBJECT at the start of each frame
            if finger_tip:
                for shape_name, (cx, cy), color, area in calibrated_shapes:
                    # Calculate the bounding box of the shape
                    # Assuming the shape is roughly circular, we'll use the area to estimate the radius
                    radius = np.sqrt(area / np.pi)
                    shape_left = cx - radius
                    shape_right = cx + radius
                    shape_top = cy - radius
                    shape_bottom = cy + radius

                    # Check if the finger_tip is within the shape's bounding box
                    if (
                        shape_left <= finger_tip[0] <= shape_right
                        and shape_top <= finger_tip[1] <= shape_bottom
                    ):
                        CURRENT_OBJECT = ShapeData(
                            size=area, color=color, center=(cx, cy), name=shape_name
                        )
                        break  # Exit the loop once we've found a shape the finger is hovering over

            # Draw shapes and their names

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

            # Highlight the shape being hovered over
            if CURRENT_OBJECT:
                cx, cy = CURRENT_OBJECT.center
                radius = int(np.sqrt(CURRENT_OBJECT.size / np.pi))
                cv2.circle(
                    image, (cx, cy), radius, (0, 255, 255), 2
                )  # Yellow highlight

            if CURRENT_OBJECT:
                print(f"Hovering over: {CURRENT_OBJECT}")

            cv2.imshow("Shapes and Hand Detection", image)

            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord("c"):
                print("Recalibrating...")
                _, frame = cap_top.read()
                calibrated_shapes = calibrate(frame, paper_roi)
    finally:
        hands.close()
        cap_top.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting...")
    start()