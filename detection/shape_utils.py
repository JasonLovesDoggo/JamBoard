import cv2
import numpy as np
from .types import ShapeData

CURRENT_OBJECT: ShapeData | None = None


def get_shape_name(approx, contour):
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        # Commented out logic is for detecting squares instead of rectangles (but since we're using both as one instrument, we'll just call them rectangles)
        # _, _, w, h = cv2.boundingRect(approx)
        # aspect_ratio = float(w) / h
        # if 0.95 <= aspect_ratio <= 1.05:
        # return "Square"
        # else:
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


def process_frame(hands, mp_hands, drawing_utils, image, paper_roi, calibrated_shapes):
    global CURRENT_OBJECT

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
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            finger_tip = (int(index_tip.x * w), int(index_tip.y * h))
            cv2.circle(image, finger_tip, 10, (0, 255, 0), cv2.FILLED)

    CURRENT_OBJECT = None  # Reset CURRENT_OBJECT at the start of each frame
    if finger_tip:
        for shape_name, (cx, cy), color, area in calibrated_shapes:
            radius = np.sqrt(area / np.pi)
            shape_left = cx - radius
            shape_right = cx + radius
            shape_top = cy - radius
            shape_bottom = cy + radius

            if (
                shape_left <= finger_tip[0] <= shape_right
                and shape_top <= finger_tip[1] <= shape_bottom
            ):
                CURRENT_OBJECT = ShapeData(
                    size=area, color=color, center=(cx, cy), name=shape_name
                )
                break

    draw_shapes_and_hover(image, calibrated_shapes)
    return image, CURRENT_OBJECT


def draw_shapes_and_hover(image, calibrated_shapes):
    for shape_name, (cx, cy), _, _ in calibrated_shapes:
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

    if CURRENT_OBJECT:
        cx, cy = CURRENT_OBJECT.center
        radius = int(np.sqrt(CURRENT_OBJECT.size / np.pi))
        cv2.circle(image, (cx, cy), radius, (0, 255, 255), 2)  # Yellow highlight
