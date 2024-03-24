import os
import shutil




def check_and_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    elif os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)



def create_models_folder(models_path):
    list_model = []
    for model_name in os.listdir(models_path):
        full_path = os.path.join(models_path, model_name)
        n, ext = os.path.splitext(model_name)
        list_model.append(n)
    
    list_model = list(set(list_model))

    # Create the folder for each model
    for model_name in list_model:
        model_folder = os.path.join(models_path, model_name)
        check_and_create_folder(model_folder)

    # for each file and not folder in the models_path
    for file_name in os.listdir(models_path):
        full_path = os.path.join(models_path, file_name)
        n, ext = os.path.splitext(file_name)
        if os.path.isfile(full_path):
            if n in list_model:
                shutil.move(full_path, os.path.join(models_path, n, file_name))


if __name__ == '__main__':
    CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    MODELS_THALES_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets', 'ThalesDataset', 'Models')

    create_models_folder(MODELS_THALES_PATH)