import os
import yaml
import argparse
import sys
import trimesh

def generate(dataset_name:str) -> None:

    CURRENT_DIR_PATH = os.path.dirname(__file__)
    dataset_path = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets', dataset_name)

    if not os.path.exists(dataset_path):
        print('\n\n!!! The dataset you specified does not exists !!!\n\n')
        sys.exit()

    models_id_yaml_file = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'models_id.yml')
    with open(models_id_yaml_file, 'r') as f:
        loaded_models = yaml.safe_load(f)

    classes_file = os.path.join(dataset_path, 'classes.txt')
    if os.path.exists(classes_file):
        os.remove(classes_file)

    all_models_loaded = ''
    for model in loaded_models:
        all_models_loaded += f'{model[:-4]}\n'

    with open(classes_file, 'w') as f:
        f.write(all_models_loaded)
    
    print(f'\n\n --- classes.txt created for {dataset_name} ---\n\n')

    model_folders = os.listdir(os.path.join(dataset_path, 'Models'))
    for model_folder in model_folders:
        model_folder = os.path.join(dataset_path, 'Models', model_folder)

        model_name = [f for f in os.listdir(model_folder) if f.endswith('.obj')][0]
        mesh = trimesh.load(os.path.join(model_folder, model_name))

        points = ''
        for vertex in mesh.vertices:
            points += f'{vertex[0]} {vertex[1]} {vertex[2]}\n'
        
        with open(os.path.join(model_folder, 'points.xyz'), 'w') as f:
            f.write(points)

        print(f'\n\n --- points.xyz created for {model_name} model ---\n\n')

    file_list_path = os.path.join(dataset_path, 'file_list.txt')
    if os.path.exists(file_list_path):
        os.remove(classes_file)

    files = ''
    generated_scenes_path = os.path.join(dataset_path, 'GeneratedScenes')
    for sequence in os.listdir(generated_scenes_path):
        sequence_path = os.path.join(generated_scenes_path, sequence)
        for file in os.listdir(sequence_path):
            if not file.endswith('-color.png'):
                continue
            file_id = file.split('-')[0]
            files += f'{sequence}/{file_id}\n'

    with open(file_list_path, 'w') as f:
        f.write(files)

    print(f'\n\n --- files_list.txt created ---\n\n')

if __name__ == '__main__':

    # Define paths
    CURRENT_DIR_PATH = os.path.dirname(__file__)
    CONFIG_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'scene_generation.yml')

    with open(CONFIG_PATH, 'r') as f:
        config_file = yaml.safe_load(f)

    dataset_name = config_file['dataset_name']

    generate(dataset_name)