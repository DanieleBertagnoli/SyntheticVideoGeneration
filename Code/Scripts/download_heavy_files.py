import os
import requests
from tqdm import tqdm
import shutil

def download_file(url, save_path):
    """
    Downloads a file from the given URL and saves it to the specified path.

    Args:
    - url (str): The URL from which to download the file.
    - save_path (str): The path where the downloaded file will be saved.
    """

    print('\nStarting the download\n')
    
    # Request the file from the URL
    response = requests.get(url, stream=True)
    
    # Get the total size of the file
    total_size = int(response.headers.get('content-length', 0))
    
    # Set the block size for downloading
    block_size = 1024  # 1 Kibibyte

    # Initialize tqdm progress bar
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    # Open the file and start downloading
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    # Close the progress bar
    progress_bar.close()

    # Check if download was successful
    if total_size != 0 and progress_bar.n != total_size:
        print("Download failed. Please try again.")
    else:
        print(f"File downloaded successfully and saved at {save_path}")



if __name__ == "__main__":

    # Define the main URL from which files will be downloaded
    MAIN_URL = 'https://bertagnoli.ddns.net/static/PublicDrive'

    CURRENT_DIR_PATH = os.path.dirname(__file__)

    # Dictionary containing files to be downloaded along with their save paths
    files_to_download = {
        '1.blend': os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'BlenderScenes'),
        '2.blend': os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'BlenderScenes'),
        'Blender-2.79b-Linux.tar.xz': os.path.join(CURRENT_DIR_PATH, '..', '..'),
        'Blender-2.79b-Windows.zip': os.path.join(CURRENT_DIR_PATH, '..', '..')
    }

    # Loop through the files to download
    for file, save_path in files_to_download.items():
        while True:
            choice = input(f'\nDo you want to download {file}? [y]/n\n')
            if choice == 'n' or choice == 'y' or choice == '':
                break
            else:
                print('\nType y for yes, n for no!')
        
        choice = str(choice).lower()
        if choice == 'y' or choice == '':

            # Create directories if they don't exist
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            # Remove old file if exists
            if os.path.exists(os.path.join(save_path, file)):
                print(f'Removing old {os.path.join(save_path, file)}')
                os.remove(os.path.join(save_path, file))

            # Download the file
            download_file(MAIN_URL + f'/{file}', os.path.join(save_path, file))

            # For Blender files, remove old Blender installation and unpack the new one
            if file == 'Blender-2.79b-Linux.tar.xz' or file == 'Blender-2.79b-Windows.zip':
                old_blender_folder = os.path.join(save_path, file[:-7])
                if os.path.exists(old_blender_folder):
                    print('Removing old blender')
                    shutil.rmtree(old_blender_folder)

                print('Unpacking blender')

                # Unpack the Blender file and remove the downloaded archive
                command = f'tar -xf {os.path.join(save_path, file)} -C {save_path}'
                os.system(command)
                os.remove(os.path.join(save_path, file))