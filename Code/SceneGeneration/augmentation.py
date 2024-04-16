import cv2
import numpy as np
import os
import random
import shutil

def apply_motion_blur(image, kernel_size=15):
    # Generate a motion blur kernel
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size

    # Apply the kernel to the image
    output = cv2.filter2D(image, -1, kernel_motion_blur)
    return output

def apply_salt_and_pepper(image, salt_prob=0.01, pepper_prob=0.01):
    # Add salt and pepper noise to the image
    noisy = np.copy(image)
    num_salt = np.ceil(salt_prob * image.size)
    num_pepper = np.ceil(pepper_prob * image.size)

    # Add Salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords[0], coords[1], :] = 1

    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[coords[0], coords[1], :] = 0

    return noisy


def augment(dir_path):
    '''
    dir_pat: str
        The path to the directory containing the images and file annotations (multiple annotaions in ycb dataset)
    dataset: str
        The dataset to augment
    '''

    # mb: motion blur, sp: salt and pepper, be: both effects

    images_path = [(os.path.splitext(filename)[0]).split("-")[0] for filename in os.listdir(dir_path) if filename.endswith('.png')]

    # iterate the file in dir_path
    for file_name in os.listdir(dir_path):

        if "color" in file_name and "mb" not in file_name and "sp" not in file_name and "be" not in file_name:
            full_path = os.path.join(dir_path, file_name)
            if os.path.isfile(full_path):
                # Read the image
                image = cv2.imread(full_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                id_img = (os.path.splitext(file_name)[0]).split("-")[0]

                rn = np.random.rand()
                if rn > 0.5:
                    motion_blurred = apply_motion_blur(image)
                    full_path = os.path.join(dir_path, file_name.split(".")[0] + "-mb.png")
                    cv2.imwrite(full_path, cv2.cvtColor(motion_blurred, cv2.COLOR_RGB2BGR))
                    for fn in os.listdir(dir_path):
                        if "color" not in fn and "mb" not in fn and "sp" not in fn and "be" not in fn:
                            fp = os.path.join(dir_path, fn)
                            if os.path.isfile(fp):
                                if id_img in fn:
                                    shutil.copy(fp, os.path.join(dir_path, fn.split(".")[0] + "-mb." + fn.split(".")[1]))

                rn = np.random.rand()
                if rn > 0.5:
                    salt_pepper = apply_salt_and_pepper(image)
                    full_path = os.path.join(dir_path, file_name.split(".")[0] + "-sp.png")
                    cv2.imwrite(full_path, cv2.cvtColor(salt_pepper, cv2.COLOR_RGB2BGR))
                    for fn in os.listdir(dir_path):
                        if "color" not in fn and "mb" not in fn and "sp" not in fn and "be" not in fn:
                            fp = os.path.join(dir_path, fn)
                            if os.path.isfile(fp):
                                if id_img in fn:
                                    shutil.copy(fp, os.path.join(dir_path, fn.split(".")[0] + "-sp." + fn.split(".")[1]))

                rn = np.random.rand()
                if rn > 0.5:
                    both_effects = apply_salt_and_pepper(apply_motion_blur(image))
                    full_path = os.path.join(dir_path, file_name.split(".")[0] + "-be.png")
                    cv2.imwrite(full_path, cv2.cvtColor(both_effects, cv2.COLOR_RGB2BGR))
                    for fn in os.listdir(dir_path):
                        if "color" not in fn and "mb" not in fn and "sp" not in fn and "be" not in fn:
                            fp = os.path.join(dir_path, fn)
                            if os.path.isfile(fp):
                                if id_img in fn:
                                    shutil.copy(fp, os.path.join(dir_path, fn.split(".")[0] + "-be." + fn.split(".")[1]))

                
                if np.random.rand() > 0.5:
                    pass
                else:
                    for fn in os.listdir(dir_path):
                        fp = os.path.join(dir_path, fn)
                        if os.path.isfile(fp):
                            if id_img in fn and "mb" not in fn and "sp" not in fn and "be" not in fn:
                                os.remove(fp)
                

    return "Succesfully augmented the dataset!"
    




if __name__ == '__main__':
    # Example usage
    path = r"C:\Users\marco\Documents\master_thesis\SyntheticVideoGeneration\Data\Datasets\ThalesDataset\GeneratedScenes\0000"
    augment(path)
