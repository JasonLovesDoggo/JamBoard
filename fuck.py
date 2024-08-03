import librosa
import sounddevice as sd
import time

numOfIntervals = 10
shapes = ["A", "B", "C"]

shapeSteps = [5, 12 ,3]

numOfShapes = len(shapes)

y, sr = librosa.load('short.mp3', sr=16000)

pairs = {}

stepsPerInterval = 0

for i in range(numOfShapes):
    for j in range(numOfShapes):
        if shapes[i] != shapes[j]:  # Skip pairs like "AA", "BB", "CC"
            

            stepsPerInterval = (shapeSteps[i]-shapeSteps[j])/numOfIntervals

            rainbow = []
            for k in range(numOfIntervals):
                rainbow.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=stepsPerInterval*k))
            

            key = shapes[i] + shapes[j]
            pairs[key] = rainbow

print(stepsPerInterval)

for i in range(10):
    sd.play(pairs["CB"][i], sr)
    time.sleep(0.2)


