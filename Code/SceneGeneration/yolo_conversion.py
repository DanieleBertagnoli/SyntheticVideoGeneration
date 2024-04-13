import os
import yaml
import shutil
import random
from time import time

def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamps a value within a specified range.

    Args:
        value (float): The value to be clamped.
        min_value (float): The minimum allowed value.
        max_value (float): The maximum allowed value.

    Returns:
        float: The clamped value.
    """

    # Clamp the value within the specified range
    return max(min_value, min(value, max_value))



def extract_data(generated_scenes_path: str, yolo_dataset_path: str) -> None:
    """
    Extracts data from generated scenes to a YOLO dataset directory.

    Args:
        generated_scenes_path (str): Path to the directory containing generated scenes.
        yolo_dataset_path (str): Path to the directory where YOLO dataset will be stored.

    Returns:
        None
    """

    print(f'Copying {generated_scenes_path} into {yolo_dataset_path}')
    shutil.copytree(generated_scenes_path, yolo_dataset_path)

    print('Extracting files...')    
    
    # Move files within folders and rename them
    file_list = sorted(os.listdir(yolo_dataset_path))
    for folder_name in file_list:
        folder_path = os.path.join(yolo_dataset_path, folder_name)

        for file_name in os.listdir(folder_path):

            file_path = os.path.join(folder_path, file_name)

            if 'color' in file_name:
                file_name = folder_name + '-' + file_name.replace('-color', '')

            elif 'box-2d' in file_name:
                file_name = folder_name + '-' + file_name

            else:
                continue

            new_file_path = os.path.join(yolo_dataset_path, file_name)
            shutil.move(file_path, new_file_path) # Move files and rename them

        shutil.rmtree(folder_path) # Remove empty folder after moving its contents



def to_yolo(yolo_dataset_path: str, data: dict, width_img: int = 640, height_img: int = 480) -> None:
    """
    Converts bounding box annotations to YOLO format.

    Args:
        yolo_dataset_path (str): Path to the YOLO dataset directory.
        data (dict): Dictionary containing object names and corresponding object IDs.
        width_img (int): Width of the image (default is 640).
        height_img (int): Height of the image (default is 480).

    Returns:
        None
    """

    file_list = os.listdir(yolo_dataset_path)
    for box_txt_file in file_list:

        box_txt_file = os.path.join(yolo_dataset_path, box_txt_file)

        if not box_txt_file.endswith('.txt'):
            continue

        base_name, extension = os.path.splitext(box_txt_file)
        base_name = base_name.replace('-box-2d', '')
        yolo_bbox_file_name = f"{base_name}{extension}"
        
        with open(box_txt_file, 'r') as f_in, open(yolo_bbox_file_name, 'w') as f_out:
            for object_line in f_in:
                portions = object_line.strip().split()
                object_name, x_min, y_min, x_max, y_max = portions[0], float(portions[1]), float(portions[2]), float(portions[3]), float(portions[4])

                # Clamp coordinates to image dimensions
                x_min = clamp(x_min, 0, width_img)
                x_max = clamp(x_max, 0, width_img)
                y_min = clamp(y_min, 0, height_img)
                y_max = clamp(y_max, 0, height_img)

                # Calculate bounding box center and dimensions relative to image size
                x_centre = ((x_min + x_max) / 2) / width_img
                y_centre = ((y_max + y_min) / 2) / height_img
                width = (x_max - x_min) / width_img
                height = (y_max - y_min) / height_img

                # Convert object name to object ID
                object_id = data.get(object_name + '.obj')

                # Write new coordinates in YOLO format
                f_out.write(f"{object_id} {x_centre:.2f} {y_centre:.2f} {width:.2f} {height:.2f}\n")
        
        # Close files
        f_in.close()
        f_out.close()

        # Remove the original box_txt_file
        os.remove(box_txt_file)



def check_and_create_folder(folder_path: str) -> None:
    """
    Checks if a folder exists, creates it if it doesn't, and clears it if it does.

    Args:
        folder_path (str): Path to the folder to be checked or created.

    Returns:
        None
    """
    # Check if folder_path exists
    if os.path.exists(folder_path):
        # Remove existing folder and its contents
        shutil.rmtree(folder_path)

    os.makedirs(folder_path)  # Create a new empty folder



def organize_yolo_dataset(yolo_dataset_path: str) -> None:
    """
    Organizes images and labels from a YOLO dataset into separate train and validation sets.

    Args:
        yolo_dataset_path (str): Path to the YOLO dataset directory.

    Returns:
        None
    """

    # Get scene names from image files
    scenes_names = [os.path.splitext(filename)[0] for filename in os.listdir(yolo_dataset_path) if filename.endswith('.png')]
    random.shuffle(scenes_names)
    
    # Split scene names into train and validation sets
    train_files = scenes_names[:int(len(scenes_names)*0.8)]
    val_files = scenes_names[int(len(scenes_names)*0.8):]

    # Define paths for organizing the dataset
    YOLO_DATASET_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets','YoloDataset')
    YOLO_DATASET_PATH_images = os.path.join(YOLO_DATASET_PATH, 'images')
    YOLO_DATASET_PATH_labels = os.path.join(YOLO_DATASET_PATH, 'labels')

    # Create necessary folders
    check_and_create_folder(YOLO_DATASET_PATH_images)
    check_and_create_folder(YOLO_DATASET_PATH_labels)

    YOLO_DATASET_PATH_images_train = os.path.join(YOLO_DATASET_PATH, 'images', 'train')
    YOLO_DATASET_PATH_images_val = os.path.join(YOLO_DATASET_PATH, 'images', 'val')
    YOLO_DATASET_PATH_labels_train = os.path.join(YOLO_DATASET_PATH, 'labels', 'train')
    YOLO_DATASET_PATH_labels_val = os.path.join(YOLO_DATASET_PATH, 'labels', 'val')

    check_and_create_folder(YOLO_DATASET_PATH_images_train)
    check_and_create_folder(YOLO_DATASET_PATH_images_val)
    check_and_create_folder(YOLO_DATASET_PATH_labels_train)
    check_and_create_folder(YOLO_DATASET_PATH_labels_val)

    # Move image and label files to appropriate train or validation folders
    for file_name in os.listdir(yolo_dataset_path):
        n, ext = os.path.splitext(file_name)

        if file_name.endswith('.png'):
            full_path = os.path.join(yolo_dataset_path, file_name)
            if os.path.isfile(full_path):
                if n in train_files:
                    shutil.move(full_path, os.path.join(YOLO_DATASET_PATH_images_train, file_name))
                elif n in val_files:
                    shutil.move(full_path, os.path.join(YOLO_DATASET_PATH_images_val, file_name))

        elif file_name.endswith('.txt'):
            full_path = os.path.join(yolo_dataset_path, file_name)
            if os.path.isfile(full_path):
                if n in train_files:
                    shutil.move(full_path, os.path.join(YOLO_DATASET_PATH_labels_train, file_name))
                elif n in val_files:
                    shutil.move(full_path, os.path.join(YOLO_DATASET_PATH_labels_val, file_name))



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

    # Define paths for YOLO dataset and model data
    YML_DATA_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'models_id.yml')
    YOLO_DATASET_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets','YoloDataset')

    # If YOLO dataset path exists, remove it
    if os.path.exists(YOLO_DATASET_PATH):
        shutil.rmtree(YOLO_DATASET_PATH)

    start_time = time()

    # Extract data from GENERATED_SCENES_PATH to YOLO_DATASET_PATH
    extract_data(GENERATED_SCENES_PATH, YOLO_DATASET_PATH) # extract all the files from the video folders

    delta_time = time() - start_time
    print(f'\nFiles copied and extracted in {int(delta_time)}s') 

    # Load model data from YML file
    with open(YML_DATA_PATH, 'r') as file:
        data = yaml.safe_load(file)

    start_time = time()

    # Convert extracted data to YOLO format
    to_yolo(YOLO_DATASET_PATH, data)

    delta_time = time() - start_time
    print(f'\nBounding boxes refactored in {int(delta_time)}s') 

    start_time = time()

    # Organize YOLO dataset into train and val folders
    organize_yolo_dataset(YOLO_DATASET_PATH)     

    delta_time = time() - start_time
    print(f'\nFile organized in training and validation set in {int(delta_time)}s') 