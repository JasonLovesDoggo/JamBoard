from calculations import *
from fuck import *
import sounddevice as sd
import librosa

if __name__ == "__main__":
    points = {"A": (10, 10), "B": (20, 20), "C": (0, 0), "D": (9, 40)}
    finger_tip = (10, 40)
    starting_note = "A"
    bounding_box_size = 2
    num_bounding_boxes = 10
    shape_steps = [5, 12, 3, 8]

    pairs = create_pairs(list(points.keys()), num_bounding_boxes, shape_steps)

    nearest_line_connection = find_nearest_line_to_finger_tip(
        finger_tip, points, starting_note
    )
    print(nearest_line_connection)
    bounding_box_centers = create_bounding_box_centers(
        points[starting_note], points[nearest_line_connection], num_bounding_boxes
    )

    selected_bounding_box_index = None
    for i, bounding_box in enumerate(bounding_box_centers):
        if check_if_finger_tip_is_in_bounding_box(
            finger_tip, bounding_box, bounding_box_size
        ):
            selected_bounding_box_index = i
            break

    print(pairs)
    if selected_bounding_box_index is None:
        print("Finger tip is not in bounding box")
    else:
        sd.play(pairs["AD"][0], 16000)
        time.sleep(0.5)
        sd.play(
            pairs[starting_note + nearest_line_connection][selected_bounding_box_index],
            16000,
        )
        time.sleep(0.5)
        sd.play(pairs["AD"][-1], 16000)
        time.sleep(0.5)
