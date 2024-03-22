# SyntheticVideoGeneration
This project has been inspired by the synthetic frame generation process provided by the [iros20-6d-pose-tracking](https://github.com/wenbowen123/iros20-6d-pose-tracking) project. This repository allows you to generate synthetic videos from your CAD models. The models are dynamically loaded into a Blender scene, and through a keyframe generation process, the camera moves along spherical points smoothly to create the final video. The whole project is designed to work with YCB-Video datasets like.

## Installation
We recomend the use of Anaconda (or Miniconda) to run the project since we used Python 3.10.

### Linux 
If you are using a Linux system you can simply run:
```
bash install.sh
```
The system will prompt you with some questions about files to be downloaded (`download_heavy_files.py`). To run the system you have to download at least one of the two `.blend` files (`1.blend` contains YCB-Video used to generate noise in the video, while `2.blend` is an empty scene) and the Blender version. The system will download all the files and delete the temporal files so do not delete anything manually! After this step a new folder `Blender-279b` will be created.

### Windows
As we claimed, our project is fully compatible with Windows, however, unlike Linux you have to perform some manual steps to install the needed components:

```
# Create the conda env and install the requirements
conda create --prefix=./venv python=3.10

conda activate ./venv

pip install -r requirements.txt

# Download files from the server
python Code/Scripts/download_heavy_files.py 

conda env config vars set BLENDER_PATH="/absolute/path/SyntheticVideoGeneration/Blender-2.79b/blender.exe"

# Restart the env to refresh the env vars
conda deactivate
```
After running the 
```
python Code/Scripts/download_heavy_files.py
```
You will be prompted with some questions. Please, refer to the Linux installation part for the instructions on what to do.

## How to run the project using your own dataset
In order to use the project with your dataset, you have to arrange the files such that they can be processed correctly by the scripts. We used the YCB-Dataset format as the system is designed to work with it and with similar custom datasets. 

1. Create a folder `Data/Datasets/`.
2. Inside `Data/Datasets/` create the folder `MyDataset/` (you can call it as you want).
3. Create a folder `MyDataset/ObjectModels/` and put here all the CAD model files (only .obj files are accepted, if your files are in a different format you can import them in Blender and export them as .obj)
4. Run the following command to generate the `models_id.yml` file:
```
python Code/Scripts/generate_models_id.py
```

## Run the project
The project consists in three steps:

### 1. Generate the video sequences
```
python Code/SceneGeneration/generate_scenes.py
```
This script will generate a new directory `Data/GeneratedScenes/`. Then for each blender file specified in `scene_generation.yml`, the script will generated a number of scenes (video sequences) based on the `num_scenes_to_generate` specified in the same config file as well. Each video sequence is identified by a 4 digit number (starting from 0000/). The script will generate 4 files for each frame:
 - `xxxx-color.png`: RBG Frame.
 - `xxxx-depth.png`: Depth map of the frame.
 - `xxxx-seg.png`: Segmentation map with the all you models segmented.
 - `xxxx-poses_in_world.npy`: Numpy file containing:
    1. `class_ids`: IDs of the objects in the scene (the models are picked randomly), the IDs are those specified in the `models_id.yml` file.
    2. `poses_in_world`: 4x4 matrix specifying the 6D pose of the objects ($i$-th matrix corresponds to the 6Dposes of the $i$-th model in `class_ids`).
    3. `blendercam_in_world`: 4x4 matrix specifying the 6D pose of the camera.
    4. `K`: Matrix specifying the camera intrinsic parameters.

### 2. Generated bounding boxes (Optional)
```
python Code/SceneGeneration/generate_bboxes.py
```
This script will loop over the generated sequences to generat an addition file per frame called `xxxx-box.txt`. Using the information provieded by the `.npy` files, the ground-truth boxes are computed for each object. The boxes are represented using the upper-right and lower-left corners.

### 3. Video creation (Optional)
```
python Code/SceneGeneration/visualize_video.py
```
This script is used for putting toghether all the frames of each sequence to create the video.

## Why Blender-2.79b?
We perfectly know that this Blender version is quite outdated, however most of the projects about synthetic data generation use this version. As far as we know, Blender 2.80 has completely changed the Python scripting APIs making them more complicated even for simpler tasks as the one performed by this project.

### Why should I download Blender using your script?
The Blender version we have provided in the installation script is read-to-go for the project. If you legitimatelly don't trust our files, you can freely download Blender-2.79b from the [official website](https://download.blender.org/release/Blender2.79/) and install the pip packages needed to run the project.