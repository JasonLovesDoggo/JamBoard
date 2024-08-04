import librosa
import sounddevice as sd
import time
import pygame
import numpy as np

instrument_mapping = {
    "Rectangle": "piano",
    "Circle": "drums",
    "Triangle": "sax",
    "Pentagon": "piano",
    "Hexagon": "piano",
}
    
def create_frequency(area):
    threshold = 50
    factor = 2
    edge_approx = int(np.sqrt(area))
    return int((edge_approx - threshold) / factor)

def create_pairs(shapes, num_bounding_boxes, shape_name):
    pygame.mixer.pre_init(channels=1, allowedchanges=0)
    pygame.init()
    pygame.mixer.init()
    
    if shape_name == "Circle":
        num_bounding_boxes = 2
        shapes_steps = [0,0]
    else:
        shape_steps = [create_frequency(shape['size']) for shape in shapes]

    num_of_shapes = len(shapes)

    y, sr = librosa.load(f"audio/{instrument_mapping[shape_name]}.mp3", sr=16000)

    pairs = {}

    steps_per_interval = 0

    for i in range(num_of_shapes):
        for j in range(num_of_shapes):
            if shapes[i]['center'] != shapes[j]['center']:  # Skip pairs like "AA", "BB", "CC"
                steps_per_interval = (
                    shape_steps[i] - shape_steps[j]
                ) / (num_bounding_boxes-1)

                rainbow = []
                for k in range(num_bounding_boxes):
                    pitch_shifted = librosa.effects.pitch_shift(
                        y, sr=sr, n_steps=int(steps_per_interval * k)
                    )
                    time_stretched = librosa.effects.time_stretch(
                        pitch_shifted, rate=0.2
                    )
                    a = (time_stretched * 32767).astype(np.int16)
                    # sd.play(a, sr)
                    # time.sleep(1)
                    rainbow.append(pygame.sndarray.make_sound(a))

                key = shapes[i]['center'] + shapes[j]['center']
                pairs[key] = rainbow
    return pairs


if __name__ == "__main__":
    pairs = create_pairs([(10, 10), (20, 20), (9, 40)], 8, [5, 12, 3, 8])
    print(pairs)

# print(stepsPerInterval)

# for i in range(10):
#     sd.play(pairs["CB"][i], sr)
#     time.sleep(0.1)
