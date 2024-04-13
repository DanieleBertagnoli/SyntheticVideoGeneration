# given a .npy file path, read and print it
import numpy as np
import sys

fp = "/home/daniele/Documents/University/ComputerScience/Thesis/SyntheticVideoGeneration/Data/Datasets/ThalesDataset/GeneratedScenes/0000/000448-meta.npy"

data = np.load(fp, allow_pickle=True).item()
print(data)