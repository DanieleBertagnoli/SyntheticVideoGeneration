import os 
import numpy as np
from typing import *

import trimesh
import trimesh.transformations as tf
import cv2
import yaml


def get_model_name_from_id(id) -> str:
    """
    Retrieve the name of a model given its ID.

    Args:
        id (str): The ID of the model to find the name for.

    Returns:
        str: The name of the model corresponding to the given ID, if found.
             None if no matching model ID is found in the YAML file.
    """

    # Construct the path to the YAML file containing model IDs and names
    yaml_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'Configs', 'models_id.yml')
    
    # Open the YAML file
    with open(yaml_path, 'r') as file:
        # Load YAML content into a dictionary
        models = yaml.load(file, Loader=yaml.FullLoader)
        
        # Iterate through each key-value pair in the dictionary
        for key, value in models.items():
            # If the value (model ID) matches the input ID
            if value == id:
                # Return the corresponding key (model name)
                return key



def project_points(points_3d: np.ndarray, blendercam_in_world: np.ndarray, K: np.ndarray, img_width: int) -> List[Tuple[float, float]]:
    """
    Projects 3D points onto a 2D image plane.

    Args:
        points_3d (np.ndarray): Array of 3D points with shape (N, 3).
        blendercam_in_world (np.ndarray): Transformation matrix from Blender camera coordinates to world coordinates (4x4).
        K (np.ndarray): Camera intrinsic matrix (3x3).
        img_width (int): Width of the output image.

    Returns:
        List[Tuple[float, float]]: List of 2D points (x, y) projected onto the image plane.
    """

    # Convert points to homogeneous coordinates
    points_3d_homogeneous = np.hstack((points_3d, np.ones((len(points_3d), 1))))

    # Transform points from world to camera coordinates
    blendercam_in_world_inv = np.linalg.inv(blendercam_in_world)
    points_3d_camera = np.dot(blendercam_in_world_inv, points_3d_homogeneous.T).T[:, :3]

    # Project points onto image plane
    points_2d_homogeneous = np.dot(K, points_3d_camera.T).T
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]

    points = []
    for p in points_2d:
        x = img_width/2 + (img_width/2 - p[0])
        points.append([x, p[1]])

    return points



def get_object_dimensions(model_path, pose_in_world, blendercam_in_world, K, img_width, bbox_adjustment) -> Tuple[float, float]:
    """
    Get the dimensions and projected points of an object in an image.

    Args:
        - model_path (str): The file path to the 3D model of the object.
        - pose_in_world (np.ndarray): The 4x4 transformation matrix representing the pose of the object in world coordinates.
        - blendercam_in_world (np.ndarray): The 4x4 transformation matrix representing the pose of the camera in world coordinates.
        - K (np.ndarray): The camera intrinsic matrix.
        - img_width (int): The width of the image.
        - bbox_adjustment (int): The adjustment value for the bounding box.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the vertices of the bounding box and the projected points of the object.
    """

    # Load the 3D model of the object
    mesh = trimesh.load(model_path)
    
    # Apply the 6D pose of the object relative to the world
    transformed_mesh = mesh.copy()
    transformed_mesh.apply_transform(pose_in_world)
    
    # Project the vertices of the 3D model onto the image
    projected_points = project_points(transformed_mesh.vertices, blendercam_in_world, K, img_width)
    
    # Calculate the bounding box around the projected points
    bbox_x_min, bbox_y_min = np.min(projected_points, axis=0)
    bbox_x_max, bbox_y_max = np.max(projected_points, axis=0)
    
    # Calculate the coordinates of the bounding box vertices
    top_left = [bbox_x_min - bbox_adjustment, bbox_y_min - bbox_adjustment]
    bottom_left = [bbox_x_min - bbox_adjustment, bbox_y_max + bbox_adjustment]
    top_right = [bbox_x_max + bbox_adjustment, bbox_y_min - bbox_adjustment]
    bottom_right = [bbox_x_max + bbox_adjustment, bbox_y_max + bbox_adjustment]
    
    # Return the bounding box vertices as a numpy array
    bounding_box_vertices = np.array([top_left, bottom_left, bottom_right, top_right], dtype=np.int32)

    return bounding_box_vertices, projected_points



def generated_bboxes(blendercam_in_world, pose_in_world, image_path, model_id, K, img_width, bbox_adjustment, show_image=False) -> np.ndarray:
    """
    Generate bounding boxes on an image for a given object.

    Args:
        blendercam_in_world (numpy.ndarray): Pose transformation matrix of the Blender camera in world coordinates.
        pose_in_world (numpy.ndarray): Pose transformation matrix of the object in world coordinates.
        image_path (str): Path to the input image.
        model_id (str): ID of the object model.
        K (numpy.ndarray): Camera intrinsic matrix.
        show_image (bool, optional): Whether to display the image with bounding boxes. Defaults to False.

    Returns:
        numpy.ndarray: Bounding box coordinates.
    """

    # Define paths
    CURRENT_DIR_PATH = os.path.dirname(__file__)
    OBJECT_MODELS_DIR_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets', 'ThalesDataset', 'ObjectModels')

    # Get the name of the object model
    model_name = get_model_name_from_id(model_id)

    # Get the bbox
    bbox, points = get_object_dimensions(os.path.join(OBJECT_MODELS_DIR_PATH, model_name), pose_in_world, blendercam_in_world, K, img_width, bbox_adjustment)

    if show_image:
        # Read the input image
        image = cv2.imread(image_path)

        # Draw the point and bounding box on the image
        color = (0, 255, 0)  # Green color
        thickness = 2
        for p in points:
            image = cv2.circle(image, (int(p[0]), int(p[1])), 4, color, thickness)

        image = cv2.polylines(image, [bbox], True, color, thickness)

        # Display the image
        cv2.imshow('Image', image)
        
        # Wait for a 'q' key press to close the image window
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    return bbox



def is_box_inside(box, threshold=10) -> bool:
    """
    Checks if the bounding box is inside the image for at least a specified threshold percentage.

    Parameters:
        box (tuple): A tuple representing the coordinates of the bounding box in the format (x1, y1, x2, y2).
        threshold (int, optional): The threshold percentage for the box to be considered inside the image. 

    Returns:
        bool: True if the box is inside the image for at least the threshold percentage, False otherwise.
    """

    # Define image dimensions
    image_width, image_height = (640, 480)
    
    # Extract coordinates of the bounding box
    x1, y1, x2, y2 = box

    # Calculate the area of the bounding box
    box_area = (x2 - x1) * (y2 - y1)

    # Calculate the area of intersection with the image
    inside_width = min(x2, image_width) - max(x1, 0)
    inside_height = min(y2, image_height) - max(y1, 0)
    inside_area = max(inside_width, 0) * max(inside_height, 0)

    # Calculate the percentage of the box inside the image
    inside_percentage = (inside_area / box_area) * 100

    # Check if the box is inside the image for at least the threshold percentage
    return inside_percentage >= threshold



if __name__ == '__main__':

    # Define paths
    CURRENT_DIR_PATH = os.path.dirname(__file__)
    GENERATED_SCENES_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'GeneratedScenes')
    CONFIG_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'scene_generation.yml')

    with open(CONFIG_PATH, 'r') as f:
        config_file = yaml.safe_load(f)

    folder_list = os.listdir(GENERATED_SCENES_PATH)
    folder_list.sort()
    for folder_name in folder_list:
        folder_name_path = os.path.join(GENERATED_SCENES_PATH, folder_name)

        print(f'\n\n --- Generating bboxes for sequence {folder_name} ---\n\n')

        # Iterate through files in the directory
        for file_name in os.listdir(folder_name_path):

            # Check if the file is an npy file
            if not file_name.endswith('.npy'):
                continue

            # Extract scene ID from file name
            scene_id = file_name.split('_')[0]

            # Load metadata from the file
            metadata = np.load(os.path.join(folder_name_path, file_name), allow_pickle=True).item()

            bboxes = {}
            count_object_id = 0
            
            # Iterate through class IDs in the metadata
            new_class_ids = []
            new_poses = []
            for class_id in metadata['class_ids']:
                # Generate bounding box for the object

                model_name = get_model_name_from_id(class_id)
                bboxes[model_name] = []

                bbox = generated_bboxes(metadata['blendercam_in_world'],
                                        metadata['poses_in_world'][count_object_id],
                                        os.path.join(folder_name_path, f'{scene_id}-color.png'),
                                        class_id,
                                        metadata['K'], 
                                        config_file['camera_settings']['width'],
                                        config_file['bbox_adjustment'],
                                        False)
                
                (x1, y1, x2, y2) = (bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1])

                if is_box_inside((x1, y1, x2, y2)):
                    bboxes[model_name].append(bbox)
                    new_class_ids.append(class_id)
                    new_poses.append(metadata['poses_in_world'][count_object_id])
                
                count_object_id += 1


            with open(os.path.join(folder_name_path, f'{scene_id}-box.txt'), 'w') as f:
                for model in bboxes:
                    for model_instance in bboxes[model]:
                        bbox_coords = f'{model_instance[0][0]} {model_instance[0][1]} {model_instance[2][0]} {model_instance[2][1]}'
                        f.write(f"{model} {bbox_coords}\n")
            
            
            data_dict = {
                'class_ids': new_class_ids,
                'poses_in_world':new_poses,
                'blendercam_in_world': metadata['blendercam_in_world'],
                'K': metadata['K']
            }

            # Save the dictionary as a .npy file
            npy_filename = os.path.join(folder_name_path, file_name)
            np.save(npy_filename, data_dict)