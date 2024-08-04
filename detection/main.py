import cv2
import mediapipe as mp
from .shape_utils import process_frame
from .hands import MediaPipeHands
from .utils import load_calibration, calibrate, initialize_video_captures, check_video_capture
from .types import ShapeData

CURRENT_OBJECT: ShapeData | None = None

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def main_loop(cap_top, cap_side, hands, drawing_utils, paper_roi):
    calibrated_shapes = load_calibration()
    if not calibrated_shapes:
        ret, frame = cap_top.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            cap_top.release()
            cv2.destroyAllWindows()
            exit()
        calibrated_shapes = calibrate(frame, paper_roi)

    while cap_top.isOpened() and cap_side.isOpened():
        success, image = cap_top.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = process_frame(hands, mp_hands, drawing_utils, image, paper_roi, calibrated_shapes)

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

def start():
    hands_instance = MediaPipeHands()
    hands = hands_instance.hands
    drawing_utils = hands_instance.drawing_utils

    cap_top, cap_side = initialize_video_captures()
    check_video_capture(cap_top, cap_side)

    paper_roi = (100, 100, 540, 380)
    print("Press 'c' to recalibrate the shapes.")

    try:
        main_loop(cap_top, cap_side, hands, drawing_utils, paper_roi)
    finally:
        hands.close()
        cap_top.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting...")
    start()
