# ---------- CONFIG ----------

# Bone names – CHANGE if your names differ
FINGERS = {
    "index":  ["index_01", "index_02", "index_03"],
    "middle": ["middle_01", "middle_02", "middle_03"],
    "ring":   ["ring_01", "ring_02", "ring_03"],
    "pinky":  ["pinky_01", "pinky_02", "pinky_03"],
    "thumb":  ["thumb_01", "thumb_02"]
}

# Curl distribution (base, mid, tip)
CURL_WEIGHTS = [1.0, 0.9, 0.8]

# Axis indices
AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

# Which axis curls fingers
CURL_AXIS = AXIS_X

# Slight inward arc bias per finger
BIAS = {
    "index": -4,
    "middle": -2,
    "ring": 2,
    "pinky": 4
}
