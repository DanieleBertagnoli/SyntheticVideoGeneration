#!/bin/bash

eval "$(conda shell.bash hook)"

# Define the possible anaconda/miniconda paths
miniconda_path=~/miniconda3/etc/profile.d/conda.sh
anaconda_path=~/anaconda3/etc/profile.d/conda.sh

if [ -e "$miniconda_path" ]; then # Check if miniconda path exists
    source "$miniconda_path"

elif [ -e "$anaconda_path" ]; then # Check if anaconda path exists
    source "$anaconda_path"

else # If neither path exists, print an error and exit
    echo "Error: conda.sh path not found"
    exit 1
fi

# Create the conda env and install the requirements
conda create --prefix=./venv python=3.10

conda activate ./venv

conda info # Print env infos

pip install -r requirements.txt

python Code/Scripts/download_heavy_files.py # Download files from the server

blender_path="$(pwd)/Blender-2.79b/blender" # Define the default Blender path

# Set the Blender path as an environment variable
if [ -e "$blender_path" ]; then # Check if the Blender binary exists
    conda env config vars set BLENDER_PATH="$blender_path"
    echo "BLENDER_PATH set to: $blender_path"

elif [ -e "${blender_path}.exe" ]; then # Check if the Blender executable (for Windows) exists
    conda env config vars set BLENDER_PATH="${blender_path}.exe"
    echo "BLENDER_PATH set to: ${blender_path}.exe"
    
else
    echo "Error: Blender path not found"
    exit 1
fi

echo -e "\n\n!!! Installation completed! Now you can activate your conda env. !!!"