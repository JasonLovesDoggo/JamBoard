from calculations import *
from fuck import *
import sounddevice as sd
import librosa
import time
import pygame


def main():
    # sd.default.latency = 'low'

    points = {"A": (10, 10), "B": (20, 20), "C": (0, 0), "D": (9, 40)}
    finger_tip = (20,20)
    starting_note = "A"
    bounding_box_size = 2
    num_bounding_boxes = 8
    shape_steps = [0, -34, -3, -8]
    # shape_steps = [0 for i in range(4)]

    pygame.mixer.pre_init(frequency=16000, channels=1, allowedchanges=0)
    pygame.init()
    pygame.mixer.init()

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

    time.sleep(2)
    if selected_bounding_box_index is None:
        print("Finger tip is not in bounding box")
    else:
        prev_channel = None

        for i in range(10):
            for pair in pairs[starting_note + nearest_line_connection]:
                channel = pair.play()
                time.sleep(0.05)
                if prev_channel is not None:
                    prev_channel.stop()
                time.sleep(1)  # Adjust the sleep duration as needed
                
                prev_channel = channel
                
            for pair in pairs[starting_note + nearest_line_connection][::-1]:
                channel = pair.play()
                time.sleep(0.05)
                if prev_channel is not None:
                    prev_channel.stop()
                time.sleep(1)  # Adjust the sleep duration as needed

                prev_channel = channel
        # pairs[starting_note+nearest_line_connection][0].play()
        # time.sleep(0.5)
        # pairs[starting_note+nearest_line_connection][selected_bounding_box_index].play()
        # time.sleep(0.5)
        # pairs[starting_note+nearest_line_connection][-1].play()
        # time.sleep(0.5)
        pass


if __name__ == "__main__":
    main()
