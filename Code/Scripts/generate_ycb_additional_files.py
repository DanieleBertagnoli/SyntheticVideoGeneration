import os
import yaml
import open3d as o3d
import sys
import numpy as np
import trimesh
import json


def create_model_info(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    vertices = np.asarray(mesh.vertices)
    diameter = np.linalg.norm(np.max(vertices, axis=0) - np.min(vertices, axis=0))
    min_x, min_y, min_z = np.min(vertices, axis=0)
    max_x, max_y, max_z = np.max(vertices, axis=0)
    com = np.mean(vertices, axis=0)
    
    return {
        "diameter": diameter,
        "min_x": min_x,
        "min_y": min_y,
        "min_z": min_z,
        "max_x": max_x,
        "max_y": max_y,
        "max_z": max_z,
        "com": com.tolist(),
        "number_of_points": len(vertices)
    }


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
    models_info = {}
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

        model_name = model_name.replace('.obj', '.ply')
        model_path = os.path.join(model_folder, model_name)

        print(f'Creating model info for {model_name}')
        models_info[model_name[:-4]] = create_model_info(model_path)

    output_path = os.path.join(dataset_path, 'models_info.json')
    with open(output_path, 'w') as f:
        json.dump(models_info, f, indent=4)

    file_list_path = os.path.join(dataset_path, 'file_list.txt')
    if os.path.exists(file_list_path):
        os.remove(file_list_path)

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