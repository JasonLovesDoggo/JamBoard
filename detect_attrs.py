import cv2
import numpy as np
import mediapipe as mp
import time

def get_shape_name(approx, contour):
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

    if len(approx) == 3:
        return 'Triangle'
    elif len(approx) == 4:
        # Check if it's a square or rectangle
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w)/h
        if 0.95 <= aspect_ratio <= 1.05:
            return 'Square'
        else:
            return 'Rectangle'
    elif 0.7 <= circularity:  # More lenient circularity check
        return 'Circle'
    elif len(approx) == 5:
        return 'Pentagon'
    elif len(approx) == 6:
        return 'Hexagon'
    else:
        return 'Unknown'

def calibrate(frame, paper_roi):
    paper_area = frame[paper_roi[1]:paper_roi[3], paper_roi[0]:paper_roi[2]]
    gray = cv2.cvtColor(paper_area, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to handle shadows better
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        
        shape_name = get_shape_name(approx, contour)
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            cx = int(M['m10'] / M['m00']) + paper_roi[0]
            cy = int(M['m01'] / M['m00']) + paper_roi[1]
            shapes.append((shape_name, (cx, cy)))
    
    print(f"Calibration complete. Detected {len(shapes)} shapes.")
    return shapes

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam (use index 1 for the second webcam)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


# Define paper boundaries (you may need to adjust these values)
paper_roi = (100, 100, 540, 380)  # (x1, y1, x2, y2)

# Calibration step
ret, frame = cap.read()
if not ret:
    print("Failed to capture frame. Exiting.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

calibrated_shapes = calibrate(frame, paper_roi)

last_touched_shape = None

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
    cv2.rectangle(image, (paper_roi[0], paper_roi[1]), (paper_roi[2], paper_roi[3]), (255, 0, 0), 2)

    finger_tip = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get index fingertip coordinates
            h, w, c = image.shape
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            finger_tip = (int(index_tip.x * w), int(index_tip.y * h))
            cv2.circle(image, finger_tip, 10, (0, 255, 0), cv2.FILLED)

    touched_shape = None
    for shape_name, (cx, cy) in calibrated_shapes:
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(image, shape_name, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if finger_tip:
            distance = np.sqrt((finger_tip[0] - cx)**2 + (finger_tip[1] - cy)**2)
            if distance < 30:  # Adjust this threshold as needed
                touched_shape = shape_name

    if touched_shape and touched_shape != last_touched_shape:
        print(f"Touched shape: {touched_shape}")
        last_touched_shape = touched_shape

    cv2.imshow('Shapes and Hand Detection', image)
    
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('c'):
        print("Recalibrating...")
        calibrated_shapes = calibrate(image, paper_roi)

hands.close()
cap.release()
cv2.destroyAllWindows()