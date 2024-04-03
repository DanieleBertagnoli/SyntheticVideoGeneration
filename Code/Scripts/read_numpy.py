import os
import numpy as np

file_path = '/home/daniele/Documents/University/ComputerScience/Thesis/SyntheticVideoGeneration/Code/SceneGeneration/../../Data/Datasets/ThalesDataset/GeneratedScenes/0003/000041-meta.npy'
data = np.load(file_path, allow_pickle=True).item()
print(f"\n {data}\n\n")