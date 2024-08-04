import cv2
import pickle
import os
import sys
from .shape_utils import get_shape_name, get_dominant_color
from .constants import *
from .types import ShapeData
from typing import List

shapes_objects: List[ShapeData]

def calibrate(frame, paper_roi):
    global shapes_objects
    shapes_objects = []
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
            shapes_objects.append(ShapeData(name=shape_name, size=area, color=color, center=(cx, cy,)))

    with open(CALIBRATION_PATH, "wb") as f:
        pickle.dump(shapes, f)

    print(f"Calibration complete. Detected {len(shapes)} shapes.")
    # print(f'{shapes_objects=}')
    return shapes


def load_calibration():
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH, "rb") as f:
            return pickle.load(f)
        return None

    # Iterate through detected hands and check if the finger is pressed
    # for hand_landmarks in results.multi_hand_landmarks:
    #     if finger_is_pressed(hand_landmarks, threshold):
    #         return True



def initialize_video_captures():
    if sys.platform == "darwin":  # Mac
        return cv2.VideoCapture(TOP_CAM), cv2.VideoCapture(SIDE_CAM)
    elif sys.platform in ["win32", "win64"]:  # Windows
        return cv2.VideoCapture(TOP_CAM, cv2.CAP_DSHOW), cv2.VideoCapture(SIDE_CAM, cv2.CAP_DSHOW)
    else:
        return cv2.VideoCapture(TOP_CAM), cv2.VideoCapture(SIDE_CAM)

    
    
def check_video_capture(cap_top, cap_side):
    if not cap_top.isOpened():
        print("Error: Could not open top webcam.")
        exit()
    if not cap_side.isOpened():
        print("Error: Could not open side webcam.")
        exit()