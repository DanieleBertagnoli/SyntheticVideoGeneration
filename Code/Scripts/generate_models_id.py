import os
import yaml

if __name__ == '__main__':

    
    # Directory containing the model files (PLY and OBJ)
    CURRENT_DIR_PATH = os.path.dirname(__file__)
    yaml_file = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'models_id.yml')

    # Get a list of all files in the directory
    model_files = os.listdir(os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets', 'ThalesDataset', 'ObjectModels'))

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
