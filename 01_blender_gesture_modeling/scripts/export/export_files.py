import bpy
import os

# Set this to your Mac's target folder (e.g., your Desktop)
target_dir = os.path.expanduser("Users/Dinesh/Downloads/BlenderScripts")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for text in bpy.data.texts:
    # Use the script's name in Blender as the filename
    filename = text.name if text.name.endswith(".py") else text.name + ".py"
    filepath = os.path.join(target_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(text.as_string())
    print(f"Exported {text.name} to {target_dir}")
