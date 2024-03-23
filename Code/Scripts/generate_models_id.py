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

    # Get a list of all files in the directory
    model_files_path = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets', dataset_name, 'ObjectModels')
    if not os.path.exists(model_files_path):
        print(f"Error: Dataset directory '{model_files_path}' not found.")
        return
    
    model_files = os.listdir(model_files_path)

    # Filter out only PLY and OBJ files
    model_files = [f for f in model_files if f.endswith((".obj"))]

    # Generate the YAML dictionary
    yaml_data = {}
    for i, model_file in enumerate(model_files):
        yaml_data[model_file] = i

    try:
        # Output the YAML data to a file
        with open(yaml_file, 'w') as file:
            yaml.dump(yaml_data, file, default_flow_style=False)
        print("YAML file generated successfully:", yaml_file)
    except Exception as e:
        print("Error:", e)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="Name of the folder placed in Data/Datasets/ containing the model files")
    args = parser.parse_args()

    if not args.dataset_name:
        print('\n\n!!! You have to specify the dataset name using --dataset_name=DatasetFolder !!!\n\n')
        sys.exit()

    generate_yaml(args.dataset_name)
