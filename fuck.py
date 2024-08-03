import librosa
import sounddevice as sd
import time
import pygame
import numpy as np


def create_pairs(shapes, num_bounding_boxes, shape_steps):
    pygame.mixer.pre_init(channels=1, allowedchanges=0)
    pygame.init()
    pygame.mixer.init()

    num_of_shapes = len(shapes)

    y, sr = librosa.load("short2.mp3", sr=16000)

    pairs = {}

    steps_per_interval = 0

    for i in range(num_of_shapes):
        for j in range(num_of_shapes):
            if shapes[i] != shapes[j]:  # Skip pairs like "AA", "BB", "CC"
                steps_per_interval = (
                    shape_steps[i] - shape_steps[j]
                ) / num_bounding_boxes

                rainbow = []
                for k in range(1, num_bounding_boxes + 1):
                    pitch_shifted = librosa.effects.pitch_shift(
                        y, sr=sr, n_steps=steps_per_interval * k
                    )
                    time_stretched = librosa.effects.time_stretch(pitch_shifted, rate=1)
                    a = (time_stretched * 32767).astype(np.int16)
                    # sd.play(a, sr)
                    # time.sleep(1)
                    rainbow.append(pygame.sndarray.make_sound(a))

                key = shapes[i] + shapes[j]
                pairs[key] = rainbow
    return pairs


if __name__ == "__main__":
    create_pairs(["A", "B", "C", "D"], 10, [5, 12, 3, 8])

# print(stepsPerInterval)

# for i in range(10):
#     sd.play(pairs["CB"][i], sr)
#     time.sleep(0.1)
