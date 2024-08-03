from typing import Optional, Type
import time
import mediapipe as mp


class MediaPipeHands:
    _instance: Optional["MediaPipeHands"] = None
    hands: mp.solutions.hands.Hands
    drawing_utils: Type[mp.solutions.drawing_utils]
    drawing_styles: Type[mp.solutions.drawing_styles]

    def __new__(cls: Type["MediaPipeHands"]) -> "MediaPipeHands":
        if cls._instance is None:
            cls._instance = super(MediaPipeHands, cls).__new__(cls)
            start_time = time.time()
            cls._instance.hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
            )
            cls._instance.drawing_utils = mp.solutions.drawing_utils
            cls._instance.drawing_styles = mp.solutions.drawing_styles
            print(
                f"MediaPipe Hands initialized in {time.time() - start_time:.2f} seconds"
            )
        return cls._instance
