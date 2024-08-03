import librosa
import sounddevice as sd
import time


def create_pairs(shapes, num_bounding_boxes, shape_steps):
    num_of_shapes = len(shapes)

    y, sr = librosa.load("short.mp3", sr=16000)

    pairs = {}

    steps_per_interval = 0

    for i in range(num_of_shapes):
        for j in range(num_of_shapes):
            if shapes[i] != shapes[j]:  # Skip pairs like "AA", "BB", "CC"
                steps_per_interval = (
                    shape_steps[i] - shape_steps[j]
                ) / num_bounding_boxes

                rainbow = []
                for k in range(num_bounding_boxes):
                    rainbow.append(
                        librosa.effects.pitch_shift(
                            y, sr=sr, n_steps=steps_per_interval * k
                        )
                    )

                key = shapes[i] + shapes[j]
                pairs[key] = rainbow
    return pairs


if __name__ == "__main__":
    create_pairs(["A", "B", "C", "D"], 10, [5, 12, 3, 8])

# print(stepsPerInterval)

# for i in range(10):
#     sd.play(pairs["CB"][i], sr)
#     time.sleep(0.1)
