import subprocess
import os

# this file generates the syhtetic scenes, the bounding boxes and creates the dataset in YOLO format 

#files_to_run = ['generate_scenes.py', 'generate_bboxes.py', 'yolo_conversion.py', 'yolo_conversion_v2.py', 'ycb_conversion_v2.py']
files_to_run = ['generate_scenes.py', 'generate_bboxes.py']
current_dir = os.path.dirname(__file__)

for file in files_to_run:
    path = os.path.join(current_dir, 'SceneGeneration', file)
    subprocess.run(['python', path], check=True)

print("All the scripts have been executed successfully!")
