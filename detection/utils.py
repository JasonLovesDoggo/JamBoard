import cv2
import pickle
import os
from .shape_utils import get_shape_name, get_dominant_color
from .constants import CALIBRATION_PATH


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


def load_calibration():
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH, "rb") as f:
            return pickle.load(f)
        return None
