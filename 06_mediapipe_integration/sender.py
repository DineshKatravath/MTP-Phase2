import cv2
import socket
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HOST = "127.0.0.1"
PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

# ---------------- MODEL ----------------
MODEL_PATH = "hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

# ---------------- MAP (IMPORTANT) ----------------
MAP = {
    0: "wrist",

    # palm anchors
    1: "palm_thumb",
    5: "palm_index",
    9: "palm_middle",
    13: "palm_ring",
    17: "palm_pinky",

    # thumb
    2: "thumb_01",
    3: "thumb_02",
    4: "thumb_02",

    # fingers
    6: "index_01", 7: "index_02", 8: "index_03",
    10: "middle_01",11: "middle_02",12: "middle_03",
    14: "ring_01",15: "ring_02",16: "ring_03",
    18: "pinky_01",19: "pinky_02",20: "pinky_03",
}

cap = cv2.VideoCapture(0)
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, frame_id)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        data = {}

        for i, lm in enumerate(hand):
            if i in MAP:
                name = MAP[i]
                data[name] = [lm.x, lm.y, lm.z]

        sock.sendall((json.dumps(data) + "\n").encode())

        # DEBUG
        print("SENT:", list(data.keys()))

    cv2.imshow("MediaPipe", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
sock.close()