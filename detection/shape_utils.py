import cv2
import numpy as np


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