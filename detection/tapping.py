import cv2
import pickle
import numpy as np
from collections import deque
from .constants import *
from typing import MutableSequence
import time

X_DIST_THRESHOLD = (
    0.70  # if user is in bottom (1 - x)% of the image, then they are tapping
)
CALIBRATION_PATH = "calibration.pkl"
CHANGE_THRESHOLD = 3300

# Initialize the background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Define skin tone range in HSV
LOWER_SKIN = np.array([0, 10, 60], dtype = "uint8") 
UPPER_SKIN = np.array([20, 150, 255], dtype = "uint8")

recent_positives: deque[float] = deque(maxlen=POSITIVE_COUNT_THRESHOLD)




def calibrate_touch(cap_side, calibration_path=CALIBRATION_PATH):
    "Calibrate the touch area by capturing the bottom x% of the image."
    ret, frame = cap_side.read()
    if not ret:
        print("Failed to capture frame for calibration. Exiting.")
        cap_side.release()
        cv2.destroyAllWindows()
        exit()
    frame = cv2.flip(frame, 1)
    img_height, img_width, _ = frame.shape
    bottom_threshold = int(img_height * X_DIST_THRESHOLD)

    touch_area = frame[bottom_threshold:img_height, 0:img_width]

    # Save the calibration image as a NumPy array
    with open(calibration_path, "wb") as f:
        pickle.dump(touch_area, f)

    print("Calibration complete. Touch area captured.")


def is_tapped(cap_side, calibration_path=CALIBRATION_PATH) -> bool:
    global recent_positives
    "Detect if a finger is touching a surface from the pov of a camera that's on the floor looking sideways at the surface."
    current_time = time.time()
    
    # Remove old positives outside the time window
    while recent_positives and current_time - recent_positives[0] > TIME_WINDOW:
        recent_positives.popleft()
    
    ret, frame = cap_side.read()
    if not ret:
        print("Failed to capture frame. Exiting.")
        cap_side.release()
        cv2.destroyAllWindows()
        exit()
    frame = cv2.flip(frame, 1)
    img_height, img_width, _ = frame.shape

    # Draw the bottom x% area
    bottom_threshold = int(img_height * X_DIST_THRESHOLD)
    cv2.rectangle(frame, (0, bottom_threshold), (img_width, img_height), (0, 255, 0), 2)

    # Load the calibration image
    try:
        with open(calibration_path, "rb") as f:
            calibration_image = pickle.load(f)
    except Exception as e:
        print(f"Failed to load calibration image: {e}")
        return False

    touch_area = frame[bottom_threshold:img_height, 0:img_width]

    # Ensure both images have the same shape and type
    if not isinstance(calibration_image, np.ndarray):
        print("Calibration image is not a NumPy array.")
        return False

    if touch_area.shape != calibration_image.shape:
        # Resize the calibration image to match touch_area shape
        calibration_image = cv2.resize(
            calibration_image, (touch_area.shape[1], touch_area.shape[0])
        )
        calibrate_touch(cap_side=cap_side)
        print("Resized calibration image to match touch area.")

    # Convert images to HSV color space
    hsv_touch_area = cv2.cvtColor(touch_area, cv2.COLOR_BGR2HSV)
    hsv_calibration_image = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2HSV)

    # Create a mask for skin tones
    skin_mask_touch_area = cv2.inRange(hsv_touch_area, LOWER_SKIN, UPPER_SKIN)
    skin_mask_calibration_image = cv2.inRange(
        hsv_calibration_image, LOWER_SKIN, UPPER_SKIN
    )

    # Apply masks to touch area and calibration image
    masked_touch_area = cv2.bitwise_and(
        touch_area, touch_area, mask=skin_mask_touch_area
    )
    masked_calibration_image = cv2.bitwise_and(
        calibration_image, calibration_image, mask=skin_mask_calibration_image
    )

    # Convert to grayscale
    gray_touch_area = cv2.cvtColor(masked_touch_area, cv2.COLOR_BGR2GRAY)
    gray_calibration_image = cv2.cvtColor(masked_calibration_image, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(gray_touch_area)
    fg_mask = cv2.medianBlur(fg_mask, 5)  # Reduce noise in the foreground mask

    # Compare the current touch area with the calibration image
    diff = cv2.absdiff(gray_touch_area, gray_calibration_image)

    # Combine the diff with the foreground mask
    combined_diff = cv2.bitwise_and(diff, fg_mask)

    # Threshold the combined difference
    _, thresh_diff = cv2.threshold(combined_diff, 30, 255, cv2.THRESH_BINARY)
    non_zero_count = cv2.countNonZero(thresh_diff)
    # print(f"Non-zero count: {non_zero_count}")

    # If there are significant changes, detect a touch
    if non_zero_count > CHANGE_THRESHOLD:  # Adjust this threshold as needed
        recent_positives.append(current_time)
        if len(recent_positives) >= POSITIVE_COUNT_THRESHOLD:
            return True

    cv2.imshow("Frame", frame)
    return False
