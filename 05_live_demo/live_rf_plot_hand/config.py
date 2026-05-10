BASE = "/Users/dinesh/Documents/mtp/hand_models/live_rf_plot_hand"

LIVE_DIR      = f"{BASE}/hand_frames_live"
CLEAN_DIR     = f"{BASE}/hand_frames_live_clean"
NORM_DIR      = f"{BASE}/hand_frames_live_normalized"
RF_DIR        = f"{BASE}/rf_output"
PROCESSED_DIR = f"{LIVE_DIR}/processed"

FREQUENCY       = 28e9
BANDWIDTH       = 400e6
NUM_SUBCARRIERS = 64
# the number of subcarriers determines the resolution of the channel impulse response (CIR) in the time domain. With 64 subcarriers, we can capture multipath components that are spaced at least 1/(400e6) = 2.5 ns apart, which corresponds to a path length difference of about 0.75 meters. 
# This should be sufficient for capturing the main multipath components in a hand gesture scenario, while keeping the computational load manageable for real-time processing.

MAX_DEPTH       = 2 
# later we can keep it as 6, as this represents the maximum number of interactions (reflections, diffractions, etc.) that a ray can undergo before reaching the receiver. 
# Setting it to 2 is a simplification for faster processing during development, but increasing it to 6 would allow for more complex and realistic scenarios to be simulated, albeit with increased computational time.
TARGET_SPAN_M   = 0.20
CLEARANCE_RATIO = 0.75
IDLE_TIMEOUT    = 30.0   # sionna ray-tracing takes several seconds per frame
POLL_INTERVAL   = 0.05
