import os
import argparse
import yaml
import sys

def generate_yaml(dataset_name):
    """
    Generates a YAML file containing model file names and their corresponding indices.

    Args:
        dataset_name (str): The name of the dataset containing the model files.
    """

    # Directory containing the model files (PLY and OBJ)
    CURRENT_DIR_PATH = os.path.dirname(__file__)
    yaml_file = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'models_id.yml')

    dataset_path = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets', dataset_name)
    if not os.path.exists(dataset_path):
        print('\n\n!!! The dataset you specified does not exists !!!\n\n')
        sys.exit()

    # Get a list of all files in the directory
    model_files_path = os.path.join(dataset_path, 'Models')
    if not os.path.exists(model_files_path):
        print(f"Error: Dataset directory '{model_files_path}' not found.")
        return
    
    model_files = []
    # Iterate though the model folders and select all the .obj files
    for model_folder in sorted(os.listdir(model_files_path)):
        model_folder = os.path.join(model_files_path, model_folder)

        model_name = [f for f in os.listdir(model_folder) if f.endswith((".obj"))][0]
        model_files.append(model_name)

    # Generate the YAML dictionary
    yaml_data = {}
    for i, model_file in enumerate(model_files):
        yaml_data[model_file] = i

    try:
        # Output the YAML data to a file
        with open(yaml_file, 'w') as file:
            yaml.dump(yaml_data, file, default_flow_style=False)
        print("YAML file generated successfully:", yaml_file)

        yaml_file = os.path.join(dataset_path, 'models_id.yml')
        with open(yaml_file, 'w') as file:
            yaml.dump(yaml_data, file, default_flow_style=False)
        print("YAML file generated successfully:", yaml_file)

    except Exception as e:
        print("Error:", e)



if __name__ == '__main__':
    
    # Define paths
    CURRENT_DIR_PATH = os.path.dirname(__file__)
    CONFIG_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'scene_generation.yml')

    with open(CONFIG_PATH, 'r') as f:
        config_file = yaml.safe_load(f)

    dataset_name = config_file['dataset_name']

    generate_yaml(dataset_name)
