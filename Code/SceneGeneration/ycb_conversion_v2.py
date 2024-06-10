import os
import shutil
from time import time
import random
import json
import cv2

import yaml
from tqdm import tqdm
import numpy as np



def copy_with_progress(src, dst):
    """Copy a single file from src to dst and update the progress bar."""
    shutil.copy2(src, dst)
    pbar.update(1)  # Update the progress bar by one step



def compute_camera_matrices(cam_pose):
    """Computes the camera rotation and translation matrices from metadata.

    Args:
        metadata (dict): Dictionary containing the camera and object data.

    Returns:
        tuple: A tuple containing the rotation matrix (3x3) and the translation vector (3).
    """

    # Invert the camera pose matrix to get the camera-to-world transformation
    cam_pose_c2w = np.linalg.inv(cam_pose)

    # Extract the rotation matrix (top-left 3x3 submatrix) and translation vector (first three elements of the last column)
    cam_R_w2c = cam_pose_c2w[:3, :3]
    cam_t_w2c = cam_pose_c2w[:3, 3]

    return cam_R_w2c, cam_t_w2c



def compute_bbox_visibility(x_min, y_min, x_max, y_max, img_width, img_height):
    # Calculate the total number of pixels in the rectangle, ensuring positive values by taking absolute differences
    px_count_all = abs(x_max - x_min) * abs(y_max - y_min)

    # Adjust the rectangle coordinates to be within the image boundaries
    x_min_adj = max(min(x_min, img_width), 0)
    y_min_adj = max(min(y_min, img_height), 0)
    x_max_adj = max(min(x_max, img_width), 0)
    y_max_adj = max(min(y_max, img_height), 0)

    # Calculate the number of pixels that are visible within the image boundaries
    if x_min_adj < x_max_adj and y_min_adj < y_max_adj:
        px_count_visib = (x_max_adj - x_min_adj) * (y_max_adj - y_min_adj)
    else:
        px_count_visib = 0  # No visible area if the adjusted bounds do not form a valid rectangle

    # Calculate the visibility fraction, handle division by zero if the rectangle has no area
    visib_fract = px_count_visib / px_count_all if px_count_all > 0 else 0

    return px_count_all, px_count_visib, px_count_visib, visib_fract



def generate(source_dataset:str, destination_dataset:str, img_width:int, img_height:int):

    if os.path.exists(destination_dataset):
            shutil.rmtree(destination_dataset)

    # Count the number of files in the source directory
    num_files = sum(len(files) for _, _, files in os.walk(source_dataset))

    # Create a global progress bar
    global pbar
    pbar = tqdm(total=num_files, desc="Copying files")
    
    # Use shutil.copytree with the custom copy function
    shutil.copytree(source_dataset, destination_dataset, copy_function=copy_with_progress)
    
    # Close the progress bar
    pbar.close()

    scenes = os.listdir(destination_dataset)
    random.shuffle(scenes)
    
    # Split scene names into train and validation sets
    train_scenes = scenes[:int(len(scenes)*0.9)]

    train_synt_path = os.path.join(destination_dataset, 'train', 'train_synt')
    test_path = os.path.join(destination_dataset, 'test_all', 'test')
    os.makedirs(train_synt_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    start_time = time()
    print('Organizing files...')

    for scn in sorted(scenes):

        scn_path = os.path.join(destination_dataset, scn)

        os.makedirs(os.path.join(scn_path, 'rgb'), exist_ok=True) #copy rgb
        os.makedirs(os.path.join(scn_path, 'depth'), exist_ok=True) # copy depth
        os.makedirs(os.path.join(scn_path, 'mask_visib'), exist_ok=True) # copy seg file n times, with n number of objects in the scene
        scene_files = os.listdir(scn_path)
        for file in scene_files:

            file_path = os.path.join(scn_path, file)

            if file.endswith('-color.png'): 
                file = file.replace('-color', '')
                new_file_path = os.path.join(scn_path, 'rgb', file)
            elif file.endswith('-depth.png'): 
                file = file.replace('-depth', '')
                new_file_path = os.path.join(scn_path, 'depth', file)
            elif file.endswith('-seg.png'): 
                file = file.replace('-seg', '')
                new_file_path = os.path.join(scn_path, 'mask_visib', file)
            elif '-box-3d' in file:
                os.remove(file_path)
                continue
            else:
                continue

            shutil.move(file_path, new_file_path)

        photos = sorted(os.listdir(os.path.join(scn_path, 'rgb')))

        scene_camera_path = os.path.join(scn_path, 'scene_camera.json')
        scene_gt_path = os.path.join(scn_path, 'scene_gt.json')
        scene_gt_info_path = os.path.join(scn_path, 'scene_gt_info.json')
        with open(scene_camera_path, 'w') as scene_camera_f, open(scene_gt_path, 'w') as scene_gt_f, open(scene_gt_info_path, 'w') as scene_gt_info_f:
            
            scene_camera_json = {}
            scene_gt_info_json = {}
            scene_gt_json = {}

            for photo in photos:
            
                photo_id = int(photo.split('.')[0])

                scene_camera_json.setdefault(photo_id, {})
                scene_gt_info_json.setdefault(photo_id, [])
                scene_gt_json.setdefault(photo_id, [])
            
                ### scene_camera.json
                meta_file = photo.replace('.png', '-meta.npy')
                meta_path = os.path.join(scn_path, meta_file)
                meta_data = np.load(meta_path, allow_pickle=True).item()

                cam_pose = meta_data['blendercam_in_world'] # 4x4 6DoF matrix representing camera position in the world
                cam_R_w2c, cam_t_w2c = compute_camera_matrices(cam_pose)  # Compute camera matrices

                camera_info = {
                    'cam_K': meta_data['intrinsic_matrix'].flatten().tolist(), 
                    'depth_scale': 0.1, # TODO investigate on its value
                    'cam_R_w2c': cam_R_w2c.flatten().tolist(),
                    'cam_t_w2c': cam_t_w2c.tolist()
                }

                scene_camera_json[photo_id] = camera_info

                ### scene_gt_info.json
                box_file = photo.replace('.png', '-box-2d.txt')
                box_path = os.path.join(scn_path, box_file)
                with open(box_path, 'r') as box_f:
                    for line in box_f:
                        object_bbox_info = {}
                        
                        portions = line.strip().split()
                        object_name, x_min, y_min, x_max, y_max = portions[0], float(portions[1]), float(portions[2]), float(portions[3]), float(portions[4])
                        
                        bbox_infos = compute_bbox_visibility(x_min, y_min, x_max, y_max, img_width, img_height)

                        x_min_adj = max(min(x_min, img_width), 0)
                        y_min_adj = max(min(y_min, img_height), 0)
                        x_max_adj = max(min(x_max, img_width), 0)
                        y_max_adj = max(min(y_max, img_height), 0)

                        new_dict_obj = {
                            'bbox_obj': [x_min_adj, y_min_adj, x_max_adj-x_min_adj, y_max_adj-y_min_adj], 
                            'bbox_visib': [x_min_adj, y_min_adj, x_max_adj, y_max_adj], 
                            'px_count_all': bbox_infos[0], 
                            'px_count_valid': bbox_infos[1], 
                            'px_count_visib': bbox_infos[2],  
                            'visib_fract': bbox_infos[3]
                            }
                        
                        scene_gt_info_json[photo_id].append(new_dict_obj)


                ### scene_gt.json
                objects_id = meta_data['cls_indexes']
                for i, id in enumerate(objects_id):
                    object_info = {}
                    object_pose = meta_data['poses'][i]  # This is the model-to-world transformation matrix

                    # Compute the object-to-camera transformation
                    # T_m2c = T_w2c * T_m2w
                    # Transform from model to world then to camera coordinates
                    T_m2c = np.dot(np.linalg.inv(cam_pose), object_pose)
                    cam_R_m2c = T_m2c[:3, :3]
                    cam_t_m2c = T_m2c[:3, 3]

                    object_info = {
                        'obj_id': int(id),
                        'cam_R_m2c': cam_R_m2c.flatten().tolist(),
                        'cam_t_m2c': cam_t_m2c.tolist()
                    }

                    # Append object_info to the appropriate JSON structure
                    scene_gt_json[photo_id].append(object_info)


                os.remove(meta_path)
                os.remove(box_path)


            json.dump(scene_camera_json, scene_camera_f, indent=2) 
            json.dump(scene_gt_info_json, scene_gt_info_f, indent=2) 
            json.dump(scene_gt_json, scene_gt_f, indent=2)     

            scene_camera_f.close()
            scene_gt_info_f.close()
            scene_gt_f.close() 
            

        if scn in train_scenes:
            new_scn_path = os.path.join(train_synt_path, scn)
        else:
            new_scn_path = os.path.join(test_path, scn)
            

        shutil.move(scn_path, new_scn_path)

    
    delta_time = time() - start_time
    print(f'File organized in {int(delta_time)} s')

def apply_motion_blur(image, kernel_size=15):
    # Generate a motion blur kernel
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size

    # Apply the kernel to the image
    output = cv2.filter2D(image, -1, kernel_motion_blur)
    return output

def apply_salt_and_pepper(image, salt_prob=0.01, pepper_prob=0.01):
    # Add salt and pepper noise to the image
    noisy = np.copy(image)
    num_salt = np.ceil(salt_prob * image.size)
    num_pepper = np.ceil(pepper_prob * image.size)

    # Add Salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords[0], coords[1], :] = 1

    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[coords[0], coords[1], :] = 0

    return noisy


def modify_image(dir_path):
    '''
    dir_pat: str
        The path to the directory containing the images and file annotations (multiple annotaions in ycb dataset)
    dataset: str
        The dataset to augment
    '''

    for dir in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, dir)):
            full_path_dir = os.path.join(dir_path, dir)
            for dir_scene in os.listdir(full_path_dir): # id scenes
                full_path_scene = os.path.join(full_path_dir, dir_scene, 'rgb')
                for file_name in os.listdir(full_path_scene): # images
                    full_path_file = os.path.join(full_path_scene, file_name)
                    if os.path.isfile(full_path_file):
                        # Read the image
                        image = cv2.imread(full_path_file)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        rn = random.randint(1, 4)

                        if rn == 1:
                            motion_blurred = apply_motion_blur(image)
                            cv2.imwrite(full_path_file, cv2.cvtColor(motion_blurred, cv2.COLOR_RGB2BGR))
                        
                        elif rn == 2:
                            salt_pepper = apply_salt_and_pepper(image)
                            cv2.imwrite(full_path_file, cv2.cvtColor(salt_pepper, cv2.COLOR_RGB2BGR))

                        elif rn == 3:
                            both_effects = apply_salt_and_pepper(apply_motion_blur(image))
                            cv2.imwrite(full_path_file, cv2.cvtColor(both_effects, cv2.COLOR_RGB2BGR))
                        
                        else:
                            pass

    return "Succesfully modified the dataset!"

if __name__ == '__main__':


    # Get the current directory path
    CURRENT_DIR_PATH = os.path.dirname(__file__)

    # Load configuration file
    CONFIG_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'scene_generation.yml')
    with open(CONFIG_PATH, 'r') as f:
        config_file = yaml.safe_load(f)

    # Extract relevant information from the configuration file
    dataset_name = config_file['dataset_name']
    GENERATED_SCENES_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets', dataset_name, 'GeneratedScenes')
    GENERATED_SCENES_BOP_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets', dataset_name, 'GeneratedScenesBop')

    generate(GENERATED_SCENES_PATH, GENERATED_SCENES_BOP_PATH, config_file['camera_settings']['width'], config_file['camera_settings']['height'])

    #modify_image(os.path.join(GENERATED_SCENES_BOP_PATH, 'test_all'))
    #modify_image(os.path.join(GENERATED_SCENES_BOP_PATH, 'train'))

    