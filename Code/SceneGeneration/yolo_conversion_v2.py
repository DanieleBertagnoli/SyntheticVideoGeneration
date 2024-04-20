import os
import shutil
import yaml

def move_files(src_dir, dest_dir):
    for filename in os.listdir(src_dir):
        if filename.endswith('.txt'):
            os.rename(os.path.join(src_dir, filename), os.path.join(src_dir, filename))
            shutil.copy(os.path.join(src_dir, filename), dest_dir)
        else:
            shutil.copy(os.path.join(src_dir, filename), dest_dir)

def new_yolo(src_dir, dest_dir):

    # create new folder YoloDatasetV2 in dest_dir
    yolo_dir = os.path.join(dest_dir, 'YoloDatasetV2')
    os.makedirs(yolo_dir, exist_ok=True)

    new_join_path = os.path.join(dest_dir, 'YoloDatasetV2')
    train_images = os.path.join(new_join_path, 'train', 'images')
    train_labels = os.path.join(new_join_path, 'train', 'labels')
    test_images = os.path.join(new_join_path, 'test', 'images')
    test_labels = os.path.join(new_join_path, 'test', 'labels')
    os.makedirs(train_images, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)
    os.makedirs(test_images, exist_ok=True)
    os.makedirs(test_labels, exist_ok=True)

    path_train_images = os.path.join(src_dir, 'images', 'train')
    path_train_labels = os.path.join(src_dir, 'labels', 'train')
    path_test_images = os.path.join(src_dir, 'images', 'val')
    path_test_labels = os.path.join(src_dir, 'labels', 'val')
    move_files(path_train_images, train_images)
    move_files(path_train_labels, train_labels)
    move_files(path_test_images, test_images)
    move_files(path_test_labels, test_labels)

def create_txt_list(yolo_dataset_path, yolo_dataset_path_test, yolo_dataset_path_train):
    # create train.txt file in yolo_dataset_path
    with open(os.path.join(yolo_dataset_path, 'train', 'train.txt'), 'w') as f:
        for filename in os.listdir(yolo_dataset_path_train):
            f.write("./images/" + filename + "\n")
    with open(os.path.join(yolo_dataset_path, 'test', 'val.txt'), 'w') as f:
        for filename in os.listdir(yolo_dataset_path_test):
            f.write("./images/" + filename + "\n")


if __name__ == '__main__':

    CURRENT_DIR_PATH = os.path.dirname(__file__)
    CONFIG_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'scene_generation.yml')

    with open(CONFIG_PATH, 'r') as f:
        config_file = yaml.safe_load(f)

    dataset_name = config_file['dataset_name']

    GENERATED_SCENES_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets', dataset_name, 'GeneratedScenes')
    GENERATED_VIDEOS_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets', dataset_name, 'GeneratedVideos')
    YML_DATA_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'models_id.yml')

    DEST_DATASET_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets')
    YOLO_DATASET_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets','YoloDataset')

    new_yolo(YOLO_DATASET_PATH, DEST_DATASET_PATH)

    YOLOV2_DATASET_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets','YoloDatasetV2')
    YOLOV2_DATASET_PATH_TEST = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets','YoloDatasetV2', 'test', 'images')
    YOLOV2_DATASET_PATH_TRAIN = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets','YoloDatasetV2', 'train', 'images')


    create_txt_list(YOLOV2_DATASET_PATH, YOLOV2_DATASET_PATH_TEST, YOLOV2_DATASET_PATH_TRAIN)