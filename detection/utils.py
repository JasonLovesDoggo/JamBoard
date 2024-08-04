import cv2
import pickle
import os
import sys
from .shape_utils import get_shape_name, get_dominant_color
from .constants import *
from .types import ShapeData
from typing import List
from .tapping import calibrate_touch
from .sound_main import findclosest, create_pairs

shapes_formatted: dict = {}
pairs_grouping: dict = {}
shapes_objects: List[ShapeData] = []



def populate_shapes_formatted(shapes_objects):
    global shapes_formatted
    shapes_formatted.clear()
    for shape in shapes_objects:
        if shape.name in shapes_formatted:
            shapes_formatted[shape.name].append(
                {"center": shape.center, "color": findclosest(shape.color), "size": shape.size}
            )
        else:
            shapes_formatted[shape.name] = [
                {"center": shape.center, "color": findclosest(shape.color), "size": shape.size}
            ]
    return shapes_formatted

def calibrate(frame, paper_roi, /, cap_side):
    global shapes_objects, shapes_formatted
    shapes_objects.clear() # Clear the list of shapes
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
        print(shape_name)
        M = cv2.moments(contour)
        if M["m00"] != 0.0:
            cx = int(M["m10"] / M["m00"]) + paper_roi[0]
            cy = int(M["m01"] / M["m00"]) + paper_roi[1]
            color = get_dominant_color(paper_area, contour)
            area = cv2.contourArea(contour)
            shapes.append((shape_name, (cx, cy), color, area))
            shapes_objects.append(
                ShapeData(
                    name=shape_name,
                    size=area,
                    color=color,
                    center=(
                        cx,
                        cy,
                    ),
                )
            )

    with open(CALIBRATION_PATH, "wb") as f:
        pickle.dump(shapes, f)

    print(f"Calibration complete. Detected {len(shapes)} shapes.")
    # print(f'{shapes_objects=}')
    calibrate_touch(cap_side=cap_side)
    populate_shapes_formatted(shapes_objects)
    pairs_grouping.clear()
    for shape in shapes_formatted.keys():
        pairs = create_pairs(shapes_formatted[shape], 8, shape)
        pairs_grouping[shape] = pairs
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
        return cv2.VideoCapture(TOP_CAM, cv2.CAP_DSHOW), cv2.VideoCapture(
            SIDE_CAM, cv2.CAP_DSHOW
        )
    else:
        return cv2.VideoCapture(TOP_CAM), cv2.VideoCapture(SIDE_CAM)


def check_video_capture(cap_top, cap_side):
    if not cap_top.isOpened():
        print("Error: Could not open top webcam.")
        exit()
    if not cap_side.isOpened():
        print("Error: Could not open side webcam.")
        exit()
