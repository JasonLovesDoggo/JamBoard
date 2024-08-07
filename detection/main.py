import cv2
import mediapipe as mp
from .shape_utils import process_frame
from .hands import MediaPipeHands
from .utils import (
    load_calibration,
    calibrate,
    initialize_video_captures,
    check_video_capture,
    shapes_formatted,
    pairs_grouping as pairs_dict,
)

import pygame
from .calculations import (find_nearest_line_to_finger_tip, create_bounding_box_centers, check_if_finger_tip_is_in_bounding_box)

from .tapping import is_tapped
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)
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
        calibrated_shapes = calibrate(frame, paper_roi, cap_side=cap_side)
    _, frame = cap_top.read()
    calibrated_shapes = calibrate(frame, paper_roi, cap_side=cap_side)

    tapped = False  # Flag to track tapping state
    current_finger_position = (
        None  # Variable to store current position of the index finger
    )

    
    bounding_box_size = 100
    num_bounding_boxes = 8 # also mentioned at the bottom of callibrate 
    # shape_steps = [0 for i in range(4)]

    pygame.mixer.pre_init(frequency=16000, channels=1, allowedchanges=0)
    pygame.init()
    pygame.mixer.init()

    while cap_top.isOpened() and cap_side.isOpened():
        success, image = cap_top.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image, CURRENT_OBJECT, _ = process_frame(
            hands, mp_hands, drawing_utils, image, paper_roi, calibrated_shapes
        )
        # cv2.imshow("Shapes2", image) # debug prints
        
        # print(f"Current finger position: {finger_pos}")
            
            
        if is_tapped(cap_side):
            if CURRENT_OBJECT is not None:  # Ensure there is a current object
                if not tapped:
                    print(f"Tapped on: {CURRENT_OBJECT}")

                    # MINGLUN DO SHII

                    starting_note_info = CURRENT_OBJECT
                    if not current_finger_position:
                        current_finger_position = (459, 255)
                    finger_tip = current_finger_position

                    nearest_line_connection = None
                    if len(shapes_formatted[starting_note_info.name]) > 1:
                        points = []
                        for i in shapes_formatted[starting_note_info.name]:
                            points.append(i["center"])
                        nearest_line_connection = find_nearest_line_to_finger_tip(
                            finger_tip, points, starting_note_info
                        )
                        bounding_box_centers = create_bounding_box_centers(
                            starting_note_info.center, nearest_line_connection, num_bounding_boxes
                        )

                        selected_bounding_box_index = 0
                        for i, bounding_box in enumerate(bounding_box_centers):
                            if check_if_finger_tip_is_in_bounding_box(
                                finger_tip, bounding_box, bounding_box_size
                            ):
                                selected_bounding_box_index = i
                                break
                    else:
                        selected_bounding_box_index = 0

                    if selected_bounding_box_index is None:
                        print("Finger tip is not in bounding box")
                    else:
                        if nearest_line_connection != None:
                            channel = pairs_dict[starting_note_info.name][starting_note_info.center + nearest_line_connection][selected_bounding_box_index].play()
                        else:
                            channel = pairs_dict[starting_note_info.name][starting_note_info.center + starting_note_info.center][selected_bounding_box_index].play()
                    # MINGLUN DO SHII

                    tapped = True  # Set the flag to indicate tapping detected
        else:
            tapped = False  # Reset the flag when tapping is no longer detected

        cv2.imshow("Shapes and Hand Detection", image)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord("c"):
            print("Recalibrating...")
            _, frame = cap_top.read()
            calibrated_shapes = calibrate(
                frame, paper_roi, cap_side=cap_side
            )


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
