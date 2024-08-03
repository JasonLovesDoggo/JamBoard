import numpy as np

def find_distance_from_point_to_line(point, line_start, line_end):
    A = np.array(point)
    B = np.array(line_start)
    C = np.array(line_end)
    
    # Vector from B to C
    BC = C - B
    # Vector from B to A
    BA = A - B
    # Projection of BA onto BC
    proj_len = np.dot(BA, BC) / np.dot(BC, BC)
    
    if proj_len < 0:
        # Closest point is B
        closest_point = B
    elif proj_len > 1:
        # Closest point is C
        closest_point = C
    else:
        # Closest point is the projection on the segment BC
        closest_point = B + proj_len * BC
    
    # Distance from A to the closest point on the segment
    distance = np.linalg.norm(A - closest_point)
    
    return distance


def create_bounding_box_centers(start, end, steps):
    x1, y1 = start
    x2, y2 = end
    
    bounding_box_centers = []
    x_interval = (x2 - x1) / (steps - 1)
    y_interval = (y2 - y1) / (steps - 1)
    for i in range(steps):
        bounding_box_centers.append([int(x1 + i * x_interval), int(y1 + i * y_interval)])
    return bounding_box_centers

def find_nearest_line_to_finger_tip(finger_tip, points, start_note):
    nearest_line = None
    min_distance = float("inf")
    for point in points.keys():
        if points[point] != points[start_note]:
            distance = find_distance_from_point_to_line(finger_tip, points[start_note], points[point])
            print(point, distance)
            if distance < min_distance:
                min_distance = distance
                nearest_line = point
    return nearest_line

def check_if_finger_tip_is_in_bounding_box(finger_tip, bounding_box, bounding_box_size):
    return (finger_tip[0] >= bounding_box[0] - bounding_box_size and finger_tip[0] <= bounding_box[0] + bounding_box_size and
            finger_tip[1] >= bounding_box[1] - bounding_box_size and finger_tip[1] <= bounding_box[1] + bounding_box_size)
    