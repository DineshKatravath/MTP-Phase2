# Live RF Plot Hand

This folder contains the runnable scripts for the real-time RF demo pipeline.

## Key Files

- `run_pipeline.py`: top-level launcher for the live workflow.
- `live_rf_plot.py`: real-time CSI/CIR visualization.
- `rf_scene.py`: RF scene construction helpers.
- `sionna_live*.py`: Sionna worker variants for live RF computation.
- `frame_utils.py`, `config.py`: shared runtime utilities and configuration.
- `blender_receiver/reciever_live_frames.py`: Blender-side receiver for exporting live mesh frames.

## Execution Summary

At runtime, this folder coordinates live frame generation, RF scene construction, parallel or optimized Sionna workers, and plotting of the resulting RF traces.
