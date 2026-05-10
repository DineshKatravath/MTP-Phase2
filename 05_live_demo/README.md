# 05 Live Demo

This module contains the real-time demonstration pipeline linking MediaPipe-driven Blender frames, RF simulation, and online visualization.

## Structure

- `live_rf_plot_hand/`: runtime scripts for the live pipeline, plotting, frame handling, and Sionna worker variants.
- `live_rf_plot_hand/blender_receiver/`: Blender-side receiver used to export live hand frames into the demo pipeline.

## Runtime Flow

The live demo follows a three-stage runtime loop: MediaPipe sender to Blender-driven hand update, Blender live frame export, and Sionna-based RF computation with real-time plotting.
