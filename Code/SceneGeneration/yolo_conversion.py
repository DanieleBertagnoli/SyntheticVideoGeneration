import os
import yaml
import shutil
import random


def extract_data(generated_scenes_path, generated_videos_path):
    # copy all the files from the generated scenes to the generated videos
    if not os.path.exists(generated_videos_path):
        os.makedirs(generated_videos_path)
    
    for item in os.listdir(generated_scenes_path):
        s_path = os.path.join(generated_scenes_path, item)
        d_path = os.path.join(generated_videos_path, item)
    
        if os.path.isdir(s_path):
            shutil.copytree(s_path, d_path)
        else:
            shutil.copy(s_path, generated_videos_path)


    for folder_name in os.listdir(generated_scenes_path):
        full_folder_path = os.path.join(generated_scenes_path, folder_name)

        if os.path.isdir(full_folder_path):
            for file_name in os.listdir(full_folder_path):
                full_file_path = os.path.join(full_folder_path, file_name)

                new_file_name = folder_name + '-' + file_name
                full_new_file_path = os.path.join(generated_scenes_path, new_file_name)

                shutil.move(full_file_path, full_new_file_path)

        os.rmdir(full_folder_path)

def create_yolo_dataset(source_folder, destination_folder, generated_videos_path):

    for file_name in os.listdir(source_folder):
        if file_name.endswith('-color.png') or 'yolo' in file_name:
            full_file_name = os.path.join(source_folder, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, destination_folder)
            if 'yolo' in file_name:
                os.remove(full_file_name)

    shutil.rmtree(source_folder)

    base_folder = os.path.dirname(generated_videos_path)

    new_path = os.path.join(base_folder, 'GeneratedScenes')
    os.rename(generated_videos_path, new_path)



def to_yolo(generated_scenes_path, data, width_img=640, height_img=480):
    # for each .txt file in GENERATED_SCENES_PATH
    for box_txt_file in os.listdir(generated_scenes_path):
        if not box_txt_file.endswith('.txt') or 'yolo' in box_txt_file:
            continue
        base_name, extension = os.path.splitext(box_txt_file)
        yolo_txt_scene = f"{base_name}-yolo{extension}"
        with open(os.path.join(generated_scenes_path, box_txt_file), 'r') as f_in, open(os.path.join(generated_scenes_path, yolo_txt_scene), 'w') as f_out:
            for object_line in f_in:
                portions = object_line.strip().split()
                object_name, x_min, y_min, x_max, y_max = portions[0], float(portions[1]), float(portions[2]), float(portions[3]), float(portions[4])

                if x_min < 0:
                    x_min = 0
                elif x_min > width_img:
                    x_min = width_img
                if x_max < 0:
                    x_max = 0
                elif x_max > width_img:
                    x_max = width_img
                if y_max < 0:
                    y_max = 0
                elif y_max > height_img:
                    y_max = height_img
                if y_min < 0:
                    y_min = 0
                elif y_min > height_img:
                    y_min = height_img

                # calculate the center and dimensions of the bounding box
                x_centre = ((x_min + x_max) / 2) / width_img
                y_centre = ((y_max + y_min) / 2) / height_img
                width = (x_max - x_min) / width_img
                height = (y_max - y_min) / height_img

                # objcet_id conversion
                object_id = data.get(object_name)

                # write new coordinates
                f_out.write(f"{object_id} {x_centre:.2f} {y_centre:.2f} {width:.2f} {height:.2f}\n")

def check_and_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    elif os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)


def organize_yolo_dataset(yolo_dataset_path):

    # txt and png must be the same name
    for file_name in os.listdir(yolo_dataset_path):
        full_path = os.path.join(yolo_dataset_path, file_name)
        n, ext = os.path.splitext(file_name)
        if ext == '.png':
            new_file_name = file_name.replace('-color', '')
        elif ext == '.txt':
            new_file_name = file_name.replace('-box-yolo', '')
        else:
            continue
        new_file_path = os.path.join(yolo_dataset_path, new_file_name)
        os.rename(full_path, new_file_path)

    scenes_names = [os.path.splitext(filename)[0] for filename in os.listdir(yolo_dataset_path)]
    random.shuffle(scenes_names)    
    scenes_names = list(set(scenes_names))
    
    train_files = scenes_names[:int(len(scenes_names)*0.8)]
    val_files = scenes_names[int(len(scenes_names)*0.8):]

    YOLO_DATASET_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets','YoloDataset')
    YOLO_DATASET_PATH_images = os.path.join(YOLO_DATASET_PATH, 'images')
    YOLO_DATASET_PATH_labels = os.path.join(YOLO_DATASET_PATH, 'labels')

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


    CURRENT_DIR_PATH = os.path.dirname(__file__)
    CONFIG_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'scene_generation.yml')

    with open(CONFIG_PATH, 'r') as f:
        config_file = yaml.safe_load(f)

    dataset_name = config_file['dataset_name']

    GENERATED_SCENES_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets', dataset_name, 'GeneratedScenes')
    GENERATED_VIDEOS_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets', dataset_name, 'GeneratedVideos')
    YML_DATA_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'models_id.yml')
    YOLO_DATASET_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets','YoloDataset')

    if not os.path.exists(YOLO_DATASET_PATH):
        os.makedirs(YOLO_DATASET_PATH)
    elif os.path.exists(YOLO_DATASET_PATH):
        shutil.rmtree(YOLO_DATASET_PATH)
        os.makedirs(YOLO_DATASET_PATH)

    # copiare GENERATED_SCENES_PATH in GENERATED_VIDEO_PATH    
    extract_data(GENERATED_SCENES_PATH, GENERATED_VIDEOS_PATH) # extract all the files from the video folders

    with open(YML_DATA_PATH, 'r') as file:
        data = yaml.safe_load(file)

    to_yolo(GENERATED_SCENES_PATH, data) # create a new box.txt file in yolo format

    create_yolo_dataset(GENERATED_SCENES_PATH, YOLO_DATASET_PATH, GENERATED_VIDEOS_PATH) # move the images and the box labels to a new folder for the YOLO dataset

    organize_yolo_dataset(YOLO_DATASET_PATH) # organize the YOLO dataset in train and val folders

        