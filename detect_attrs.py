import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import time
import sys
from typing import Type

CALIBRATION_PATH = "calibration.pkl"


class MediaPipeHands:
    _instance: Type["MediaPipeHands"] = None
    hands: mp.solutions.hands.Hands
    drawing_utils: mp.solutions.drawing_utils
    drawing_styles: mp.solutions.drawing_styles

    def __new__(cls: Type["MediaPipeHands"]) -> "MediaPipeHands":
        if cls._instance is None:
            cls._instance = super(MediaPipeHands, cls).__new__(cls)
            start_time = time.time()
            cls._instance.hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
            )
            cls._instance.drawing_utils = mp.solutions.drawing_utils
            cls._instance.drawing_styles = mp.solutions.drawing_styles
            print(
                f"MediaPipe Hands initialized in {time.time() - start_time:.2f} seconds"
            )
        return cls._instance


def get_shape_name(approx, contour):
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        else:
            return "Rectangle"
    elif 0.7 <= circularity:
        return "Circle"
    elif len(approx) == 5:
        return "Pentagon"
    elif len(approx) == 6:
        return "Hexagon"
    else:
        return "Unknown"


def get_dominant_color(image, contour):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, 1)
    mean_color = cv2.mean(image, mask=mask)
    return tuple(map(int, mean_color[:3]))


def calibrate(frame, paper_roi):
    paper_area = frame[paper_roi[1] : paper_roi[3], paper_roi[0] : paper_roi[2]]
    gray = cv2.cvtColor(paper_area, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5
    )
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    shapes = []
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue

        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        shape_name = get_shape_name(approx, contour)
        M = cv2.moments(contour)
        if M["m00"] != 0.0:
            cx = int(M["m10"] / M["m00"]) + paper_roi[0]
            cy = int(M["m01"] / M["m00"]) + paper_roi[1]
            color = get_dominant_color(paper_area, contour)
            area = cv2.contourArea(contour)
            shapes.append((shape_name, (cx, cy), color, area))

    with open(CALIBRATION_PATH, "wb") as f:
        pickle.dump(shapes, f)

    print(f"Calibration complete. Detected {len(shapes)} shapes.")
    return shapes


# Detect shadows
# Converts frame --> grayscale, apply gaussian blur, create shadow mask
def detect_shadows(frame, paper_roi) -> np.ndarray:
    paper_area = frame[paper_roi[1] : paper_roi[3], paper_roi[0] : paper_roi[2]]
    gray = cv2.cvtColor(paper_area, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    shadow_mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 0
    )
    cv2.imshow("shadow mask", shadow_mask)
    return shadow_mask


# Check shadows around fingertip to see if they meet threshold
# (given overhead lighting, less visible shadow means it's closer to the paper)
def is_fingertip_touching(fingertip_x, fingertip_y, shadow_mask, threshold=0.5):
    neighborhood = shadow_mask[fingertip_y - 2 : fingertip_y + 2, fingertip_x - 2 : fingertip_x + 2]
    # cv2.imshow("neighborhood", neighborhood)
    shadow_pixels = cv2.countNonZero(neighborhood)
    total_pixels = neighborhood.size
    print('total pixels',total_pixels)
    # division by zero ... means that total_pixels is zero
    # why is total pixels zero
    return shadow_pixels / total_pixels < threshold


# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils


def load_calibration():
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH, "rb") as f:
            return pickle.load(f)
    return None


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
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            finger_tip = (int(index_tip.x * w), int(index_tip.y * h))
            # print('the finger tip is currently at',finger_tip)
            cv2.circle(image, finger_tip, 10, (0, 255, 0), cv2.FILLED)

    # shadow detection
    ret, frame = cap.read()

    cv2.imshow("image before shadowmask", frame)
    shadow_mask = detect_shadows(frame, paper_roi)

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

        if finger_tip:
            # finger_tip[0] is x, finger_tip[1] is y
            if is_fingertip_touching(finger_tip[0], finger_tip[1], shadow_mask):
                distance = np.sqrt((finger_tip[0] - cx) ** 2 + (finger_tip[1] - cy) ** 2)
                if distance < 30:
                    touched_shape = (shape_name, (cx, cy), color, area)

    if touched_shape:
        if touch_cooldown == 0:
            shape_name, (cx, cy), color, area = touched_shape
            print(f"Touched shape: {shape_name}")
            print(f"  Color: RGB{color}")
            print(f"  Center: ({cx}, {cy})")
            print(f"  Size: {area:.2f}")
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

hands.close()
cap.release()
cv2.destroyAllWindows()
