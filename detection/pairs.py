import librosa
import pygame
import numpy as np

instrument_mapping = {
    "Rectangle": "piano",
    "Circle": "recorder",
    "Triangle": "sax",
    "Pentagon": "piano",
    "Hexagon": "drums",
    "Unknown": "sax",
}

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


def create_frequency(area):
    threshold = 60
    factor = 10
    edge_approx = int(np.sqrt(area))
    return int((edge_approx - threshold) / factor)*-1


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

    print(shape_name)
    y, sr = librosa.load(f"audio/{instrument_mapping[shape_name]}.mp3", sr=16000)

    pairs = {}

    steps_per_interval = 0
    
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
                for k in range(num_bounding_boxes):
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


if __name__ == "__main__":
    pairs = create_pairs([(10, 10), (20, 20), (9, 40)], 8, [5, 12, 3, 8])
    print(pairs)

# print(stepsPerInterval)

# for i in range(10):
#     sd.play(pairs["CB"][i], sr)
#     time.sleep(0.1)
