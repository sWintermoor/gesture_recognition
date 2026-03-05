import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from dotenv import load_dotenv

load_dotenv()

class MediaPipeProcessor:
    def __init__(self, model_path=f"{os.getenv("HAND_LANDMARKER")}"):
        base_options = python.BaseOptions(model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        self.landmarker = vision.HandLandmarker.create_from_options(options)


    def processing(self, image_path):
        image = mp.Image.create_from_file(image_path)
        keypoints = self._extracting_keypoints(image)

        return keypoints


    def processing_frame(self, frame):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) #TODO: Check whether format changing is ok
        keypoints = self._extracting_keypoints(image)

        return keypoints


    def process_video(self, frames):
        keypoint_sequence = []

        for frame in frames:
            keypoints = self.processing_frame(frame)

            #if keypoints is not None:
            #    print(keypoints.shape)

            if keypoints is None:
                keypoint_sequence.append(np.zeros(126)) # Assuption that mediapipe uses 126 landmarks
            else:
                keypoint_sequence.append(keypoints)

        keypoint_sequence = np.array(keypoint_sequence)

        return keypoint_sequence


    def _extracting_keypoints(self, image):
        landmarks = self.landmarker.detect(image)

        if (landmarks.hand_landmarks):
            keypoints_left_hand = []
            keypoints_right_hand = []

            cnt_hands = 0
            for i, hand in enumerate(landmarks.hand_landmarks):
                cnt_hands = cnt_hands + 1
                #print(f"Handside: {landmarks.handedness[i][0].category_name}")
                if(landmarks.handedness[i][0].category_name == "Left"):
                    #print("left")
                    for lm in hand:
                        keypoints_left_hand.append([lm.x, lm.y, lm.z])
                else:
                    #print("right")
                    for lm in hand:
                        keypoints_right_hand.append([lm.x, lm.y, lm.z])

            if(cnt_hands != 2):
                if(len(keypoints_left_hand) != 0):
                    keypoints_right_hand = [[0, 0, 0]] * 21 #assumption: hand has 63 keypoints
                else:
                    keypoints_left_hand = [[0, 0, 0]] * 21

            #print(f"left: {len(keypoints_left_hand)}")
            #print(f"right: {len(keypoints_right_hand)}")

            keypoints_left_hand.extend(keypoints_right_hand)
            keypoints = np.array(keypoints_left_hand).flatten()

            if(keypoints.shape != (126,)): print(keypoints.shape)

            return keypoints
        
        return None