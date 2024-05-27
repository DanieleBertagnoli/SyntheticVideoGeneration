import os
import yaml


def rename(path, no_from):
    current_no = int(no_from)
    folders = sorted([d for d in os.listdir(path) if d.isdigit() and os.path.isdir(os.path.join(path, d))])

    for folder in folders:
        new_name = f"{current_no:04d}"
        os.rename(os.path.join(path, folder), os.path.join(path, new_name))
        current_no += 1


if __name__ == '__main__':
    # Define paths
    CURRENT_DIR_PATH = os.path.dirname(__file__)
    CONFIG_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'scene_generation.yml')

    with open(CONFIG_PATH, 'r') as f:
        config_file = yaml.safe_load(f)

    dataset_name = config_file['dataset_name']
    GENERATED_SCENES_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets', dataset_name, 'Tmp')

    # Rename folders
    no_from = "0010"
    rename(GENERATED_SCENES_PATH, no_from)
