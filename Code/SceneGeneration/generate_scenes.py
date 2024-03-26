import os
import yaml
import sys
import shutil
import argparse

def start_rendering(config_file:dict, reset_folder:bool) -> None:

    """
    Function to start rendering Blender scenes based on the configuration file.

    Args:
        - config_file (dict): Configuration data loaded from YAML file.
    """

    blender_script_path = os.path.join(CURRENT_DIR_PATH, 'blender_script.py')
    data_path = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data')

    OUTPUT_DIRECTORY = os.path.join(data_path, 'Datasets', config_file['dataset_name'], 'GeneratedScenes')
    if os.path.exists(OUTPUT_DIRECTORY) and reset_folder:
        shutil.rmtree(OUTPUT_DIRECTORY)

    for blender_scene_file in config_file['blender_files'].values():
        print(f'\n --- Start rendering using {blender_scene_file} --- \n')
        blender_start_file_path = os.path.join(data_path, 'BlenderScenes', blender_scene_file)

        try:
            render_cmd = f'{BLENDER_EXECUTABLE_PATH} {blender_start_file_path} -b -P {blender_script_path}'
            os.system(render_cmd)
        except:
            print(f'\n !!! Render failed. render_cmd: {render_cmd} !!! \n')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset_folder", type=bool, default=False, help="Name of the folder placed in Data/Datasets/ containing the model files")
    args = parser.parse_args()

    # Get Blender executable path from environment variable
    BLENDER_EXECUTABLE_PATH = os.environ.get('BLENDER_PATH')

    if BLENDER_EXECUTABLE_PATH is None:
        print("\n\n !!! Please set BLENDER_PATH as environment variable !!! \n\n")
        sys.exit()

    CURRENT_DIR_PATH = os.path.dirname(__file__)
    CONFIG_FILE_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'scene_generation.yml')

    with open(CONFIG_FILE_PATH, 'r') as f:
        config_file = yaml.safe_load(f)

    print('\n --- Scene generation config loaded --- \n')

    # Start rendering based on the configuration file 
    start_rendering(config_file, args.reset_folder)
