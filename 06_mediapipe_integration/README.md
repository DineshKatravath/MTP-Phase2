# 06 MediaPipe Integration

This module bridges MediaPipe landmarks to Blender-compatible control signals for the live gesture pipeline.

## Structure

- `sender.py`: primary MediaPipe landmark sender script.
- `blender_receivers/`: Blender receiver variants that consume landmark streams and drive the hand rig.
- `mediaPipe/`: MediaPipe model assets and supporting reference material.

## Integration Notes

- The receiver variants document successive improvements in curl estimation and rig control.
- These scripts form the bridge from webcam landmarks to Blender bone motion through a local socket-based workflow.
