from detection.calculations import *
import time
import pygame
from detection.utils import *
import math
import librosa
from detection.types import ShapeData



instrument_mapping = {
    "Rectangle": "piano",
    "Circle": "recorder",
    "Triangle": "sax",
    "Pentagon": "piano",
    "Hexagon": "drums",
    "Unknown": "sax",
}


shapes_objects = [
    ShapeData(size=1221.0, color=(126, 109, 97), center=(306, 218), name="Circle"),
    ShapeData(size=2136.5, color=(96, 90, 116), center=(259, 173), name="Rectangle"),
    ShapeData(size=1000, color=(255, 14, 85), center=(320, 320), name="Rectangle"),
    ShapeData(size=1000, color=(0, 0, 0), center=(50, 50), name="Rectangle"),
    ShapeData(size=2136.5, color=(96, 90, 116), center=(259, 173), name="Triangle"),
    ShapeData(size=1000, color=(255, 14, 85), center=(320, 320), name="Triangle"),
    ShapeData(size=1000, color=(0, 0, 0), center=(50, 50), name="Triangle"),
    ShapeData(size=1794.5, color=(39, 32, 38), center=(364, 299), name="Hexagon"),
]


def create_frequency(area):
    threshold = 60
    factor = 10
    edge_approx = int(np.sqrt(area))
    return int((edge_approx - threshold) / factor)*-1

colors = [
    ("black", (0, 0, 0)),
    ("red", (255, 0, 0)),
    ("purple", (128, 0, 128)),
    ("green", (0, 128, 0)),
    ("yellow", (255, 255, 0)),
    ("blue", (0, 0, 255)),
    ("orange", (255, 165, 0)),
]

color_mapping = {
    "red": 0,  # C 0 steps away since base note is a C
    "orange": 2,  # D
    "yellow": 4,  # E
    "green": 5,  # F
    "blue": 7,  # G
    "purple": 9,  # A
    "black": 11,  # B
}


def distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def findclosest(input_color):
    mn = 999999
    for name, rgb in colors:
        d = distance(input_color, rgb)
        if d < mn:
            mn = d
            color = name
    return color




def create_pairs(shapes, num_bounding_boxes, shape_name):
    pygame.mixer.pre_init(channels=1, allowedchanges=0)
    pygame.init()
    pygame.mixer.init()

    # if shape_name == "Circle":
    #     num_bounding_boxes = 2
    #     shapes_steps = [0, 0]
    # else:
    shape_steps = [create_frequency(shape["size"])*12+color_mapping[shape['color']] for shape in shapes]

    num_of_shapes = len(shapes)

    print(f"Generating sound for {shape_name} with {num_of_shapes} shapes")
    y, sr = librosa.load(f"audio/{instrument_mapping[shape_name]}.mp3", sr=16000)

    pairs = {}
    
    if num_of_shapes < 2:
        pitch_shifted = librosa.effects.pitch_shift(
            y, sr=sr, n_steps=shape_steps[0]
        )
        time_stretched = librosa.effects.time_stretch(
            pitch_shifted, rate=0.7
        )
        a = (time_stretched * 32767).astype(np.int16)

        key = shapes[0]["center"] + shapes[0]["center"]
        pairs[key] = [pygame.sndarray.make_sound(a), pygame.sndarray.make_sound(a)]
        return pairs
        
        

    for i in range(num_of_shapes):
        for j in range(num_of_shapes):
            if (
                shapes[i]["center"] != shapes[j]["center"]
            ):  # Skip pairs like "AA", "BB", "CC"
                steps_per_interval = (shape_steps[i] - shape_steps[j]) / (
                    num_bounding_boxes - 1
                )

                rainbow = []
                for _ in range(num_bounding_boxes):
                    pitch_shifted = librosa.effects.pitch_shift(
                        y, sr=sr, n_steps=int(shape_steps[i])
                    )
                    time_stretched = librosa.effects.time_stretch(
                        pitch_shifted, rate=0.7
                    )
                    a = (time_stretched * 32767).astype(np.int16)
                    # sd.play(a, sr)
                    # time.sleep(1)
                    rainbow.append(pygame.sndarray.make_sound(a))

                key = shapes[i]["center"] + shapes[j]["center"]
                pairs[key] = rainbow
    return pairs



# testing data.
def main():
    # Pre Click Data, can be placed in calibration stage
    shapes_formatted = populate_shapes_formatted(shapes_objects)

    bounding_box_size = 50
    num_bounding_boxes = 8
    # shape_steps = [0 for i in range(4)]

    pygame.mixer.pre_init(frequency=16000, channels=1, allowedchanges=0)
    pygame.init()
    pygame.mixer.init()

    pairs_dict = {}
    for shape in shapes_formatted.keys():
        pairs = create_pairs(shapes_formatted[shape], num_bounding_boxes, shape)
        pairs_dict[shape] = pairs

    # after click

    starting_note_info = ShapeData(
        size=1000, color=(0, 0, 0), center=(50, 50), name="Rectangle"
    )
    finger_tip = (150, 120)

    points = []
    for i in shapes_formatted[starting_note_info.name]:
        points.append(i["center"])
    nearest_line_connection = find_nearest_line_to_finger_tip(
        finger_tip, points, starting_note_info
    )
    print(nearest_line_connection)
    bounding_box_centers = create_bounding_box_centers(
        starting_note_info.center, nearest_line_connection, num_bounding_boxes
    )

    selected_bounding_box_index = None
    for i, bounding_box in enumerate(bounding_box_centers):
        if check_if_finger_tip_is_in_bounding_box(
            finger_tip, bounding_box, bounding_box_size
        ):
            selected_bounding_box_index = i
            break

    if selected_bounding_box_index is None:
        print("Finger tip is not in bounding box")
    else:
        prev_channel = None

        for i in range(10):
            for pair in pairs_dict[starting_note_info.name][
                starting_note_info.center + nearest_line_connection
            ]:
                channel = pair.play()
                time.sleep(0.05)
                if prev_channel is not None:
                    prev_channel.stop()
                time.sleep(0.05)  # Adjust the sleep duration as needed

                prev_channel = channel

            for pair in pairs_dict[starting_note_info.name][
                starting_note_info.center + nearest_line_connection
            ][::-1]:
                channel = pair.play()
                time.sleep(0.05)
                if prev_channel is not None:
                    prev_channel.stop()
                time.sleep(0.05)  # Adjust the sleep duration as needed

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
