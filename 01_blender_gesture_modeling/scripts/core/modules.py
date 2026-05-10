import bpy
import sys

config = bpy.data.texts["config.py"].as_module()
sys.modules["config"] = config

utils = bpy.data.texts["utils.py"].as_module()
sys.modules["utils"] = utils

gestures = bpy.data.texts["gestures.py"].as_module()
sys.modules["gestures"] = gestures
  