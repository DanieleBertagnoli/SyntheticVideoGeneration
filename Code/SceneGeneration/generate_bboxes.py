import os 
from time import time
from typing import *
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
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



def project_points(points_3d: np.ndarray, blendercam_in_world: np.ndarray, intrinsic_matrix: np.ndarray, img_width: int, img_height: int) -> List[Tuple[float, float]]:
    """
    Projects 3D points onto the 2D image plane.

    Parameters:
        points_3d (numpy.ndarray): Array of 3D points to be projected.
        blendercam_in_world (numpy.ndarray): Transformation matrix representing the camera position relative to the world.
        intrinsic_matrix (numpy.ndarray): Camera intrinsic matrix.
        img_width (int): Width of the image in pixels.
        img_height (int): Height of the image in pixels.

    Returns:
        List[Tuple[float, float]]: List of 2D points representing the projected points on the image plane.
    """

    # Convert points to homogeneous coordinates
    points_3d_homogeneous = np.hstack((points_3d, np.ones((len(points_3d), 1))))

    # Transform points from world to camera coordinates
    blendercam_in_world_inv = np.linalg.inv(blendercam_in_world)
    points_3d_camera = np.dot(blendercam_in_world_inv, points_3d_homogeneous.T).T[:, :3]

    # Project points onto image plane
    points_2d_homogeneous = np.dot(intrinsic_matrix, points_3d_camera.T).T
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]

    points = []
    for p in points_2d:
        x = round(img_width/2 + (img_width/2 - p[0]))
        y = round(p[1])

        # If the object is not in camera at all, the projected points could assume too negative or too positive values, 
        # therefore we put a cap to avoid overflows in the next operations
        if x < -300:
            x = -300
        if y < -300:
            y = -300

        if x > img_width + 300:
            x = img_width + 300
        if y > img_height + 300:
            y = img_height + 300

        points.append([x, y])

    return points



def get_bbox_2d(model_path: str,
                poses: np.ndarray,
                blendercam_in_world: np.ndarray,
                intrinsic_matrix: np.ndarray,
                img_width: int,
                img_height: int,
                bbox_adjustment: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the 2D bounding box around a 3D object projected onto an image.

    Parameters:
        model_path (str): Path to the 3D model file.
        poses (numpy.ndarray): Transformation matrix representing the 6D pose of the object relative to the world.
        blendercam_in_world (numpy.ndarray): Transformation matrix representing the camera position relative to the world.
        intrinsic_matrix (numpy.ndarray): Camera intrinsic matrix.
        img_width (int): Width of the image in pixels.
        img_height (int): Height of the image in pixels.
        bbox_adjustment (float): Percentage adjustment to scale the bounding box.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - bounding_box_vertices (numpy.ndarray): Vertices of the 2D bounding box.
            - projected_points (numpy.ndarray): Projected vertices of the 3D model onto the image.
    """

    # Load the 3D model of the object
    mesh = trimesh.load(model_path)
    
    # Apply the 6D pose of the object relative to the world
    transformed_mesh = mesh.copy()
    transformed_mesh.apply_transform(poses)

    # Project the vertices of the 3D model onto the image
    projected_points = project_points(transformed_mesh.vertices, blendercam_in_world, intrinsic_matrix, img_width, img_heigth)

    # Calculate the bounding box around the projected points
    bbox_x_min, bbox_y_min = np.min(projected_points, axis=0)
    bbox_x_max, bbox_y_max = np.max(projected_points, axis=0)
    
    # Calculate the adjustment values
    width_adjustment = (bbox_x_max - bbox_x_min) * bbox_adjustment / 100
    height_adjustment = (bbox_y_max - bbox_y_min) * bbox_adjustment / 100
    
    # Calculate the coordinates of the bounding box vertices with adjustment
    top_left = [int(bbox_x_min - width_adjustment), int(bbox_y_min - height_adjustment)]
    bottom_left = [int(bbox_x_min - width_adjustment), int(bbox_y_max + height_adjustment)]
    top_right = [int(bbox_x_max + width_adjustment), int(bbox_y_min - height_adjustment)]
    bottom_right = [int(bbox_x_max + width_adjustment), int(bbox_y_max + height_adjustment)]
    
    # Return the bounding box vertices as a numpy array
    bounding_box_vertices = np.array([top_left, bottom_left, bottom_right, top_right], dtype=np.int32)

    return bounding_box_vertices, projected_points



def get_bbox_3d(model_path: str,
                poses: np.ndarray,
                blendercam_in_world: np.ndarray,
                intrinsic_matrix: np.ndarray,
                img_width: int,
                img_height: int,
                bbox_adjustment: float) -> tuple:
    """
    Computes the 3D oriented bounding box of a transformed 3D model and projects it onto the image plane.

    Parameters:
        model_path (str): Path to the 3D model file.
        poses (numpy.ndarray): Transformation matrix representing the 6D pose of the object relative to the world.
        blendercam_in_world (numpy.ndarray): Transformation matrix representing the camera position relative to the world.
        intrinsic_matrix (numpy.ndarray): Camera intrinsic matrix.
        img_width (int): Width of the image in pixels.
        img_height (int): Height of the image in pixels.
        bbox_adjustment (float): Percentage adjustment to scale the bounding box.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - bbox_vertices (numpy.ndarray): Vertices of the scaled bounding box in 3D space.
            - projected_bbox_vertices (numpy.ndarray): Projected vertices of the bounding box onto the image plane.
    """

    # Load the 3D model of the object
    mesh = trimesh.load(model_path)
    
    # Apply the 6D pose of the object relative to the world
    transformed_mesh = mesh.copy()
    transformed_mesh.apply_transform(poses)

    # Compute the oriented bounding box of the transformed mesh
    bounding_box = transformed_mesh.bounding_box_oriented
    
    # Get the vertices of the bounding box
    bbox_vertices = bounding_box.vertices

    # Calculate scale factor from bbox_adjustment
    scale_factor = 1 + (bbox_adjustment / 100)
    # Calculate the centroid of the bounding box to scale vertices around it
    centroid = np.mean(bbox_vertices, axis=0)
    # Scale the vertices relative to the centroid
    bbox_vertices = centroid + scale_factor * (bbox_vertices - centroid)

    # Project the scaled vertices of the bounding box onto the image plane
    projected_bbox_vertices = np.array(project_points(bbox_vertices, blendercam_in_world, intrinsic_matrix, img_width, img_height), dtype=np.int32)

    return bbox_vertices, projected_bbox_vertices



def draw_3d_bbox(image: np.ndarray, points: list, color: tuple) -> np.ndarray:
    """
    Draws a 3D bounding box on the input image.

    Parameters:
        image (numpy.ndarray): The input image.
        points (list): List of 3D points representing the vertices of the bounding box.
        color (tuple): Color of the bounding box in BGR format.

    Returns:
        numpy.ndarray: The image with the 3D bounding box drawn on it.
    """

    points = [tuple(map(int, point)) for point in points]

    # Draw bottom square
    points_order = [0,1,3,2]
    for i in range(len(points_order)):
        start_point = points[points_order[i]]
        end_point = points[points_order[(i+1) % len(points_order)]]  # Loop back to the first point
        image = cv2.line(image, start_point, end_point, color, 2)
    
    # Draw top square
    points_order = [4,5,7,6]
    for i in range(len(points_order)):
        start_point = points[points_order[i]]
        end_point = points[points_order[(i+1) % len(points_order)]]  # Loop back to the first point
        image = cv2.line(image, start_point, end_point, color, 2)

    # Draw vertical lines (edges)
    for i in range(4):
        bottom_point = points[i]
        top_point = points[i+4]
        image = cv2.line(image, bottom_point, top_point, color, 2)

    return image



def generated_bboxes(model_paths: list,
                    blendercam_in_world: np.ndarray,
                    poses: np.ndarray,
                    image_path: str,
                    model_name: str,
                    intrinsic_matrix: np.ndarray,
                    img_width: int,
                    img_height: int,
                    bbox_adjustment_2d: float,
                    bbox_adjustment_3d: float,
                    show_image: bool = False) -> tuple:
    """
    Generates 2D and 3D bounding boxes for a given model and image.

    Parameters:
        model_paths (list): List of paths to 3D model files.
        blendercam_in_world (numpy.ndarray): Transformation matrix representing the camera position relative to the world.
        poses (numpy.ndarray): Transformation matrix representing the 6D pose of the object relative to the world.
        image_path (str): Path to the input image.
        model_name (str): Name of the model to be processed.
        intrinsic_matrix (numpy.ndarray): Camera intrinsic matrix.
        img_width (int): Width of the image in pixels.
        img_height (int): Height of the image in pixels.
        bbox_adjustment_2d (float): Percentage adjustment to scale the 2D bounding box.
        bbox_adjustment_3d (float): Percentage adjustment to scale the 3D bounding box.
        show_image (bool, optional): Whether to display the image with bounding boxes. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - bbox_2d (numpy.ndarray): 2D bounding box vertices.
            - bbox_3d (numpy.ndarray): 3D bounding box vertices.
            - projected_bbox_3d_vertices (numpy.ndarray): Projected 3D bounding box vertices onto the image plane.
    """

    # Get the model abs path
    for path in model_paths:
        if model_name in path:
            model_path = path
            break

    # Get the bbox
    bbox_2d, model_projected_vertices = get_bbox_2d(model_path, poses, blendercam_in_world, intrinsic_matrix, img_width, img_height, bbox_adjustment_2d)
    bbox_3d, projected_bbox_3d_vertices = get_bbox_3d(model_path, poses, blendercam_in_world, intrinsic_matrix, img_width, img_height, bbox_adjustment_3d)
    
    if show_image:

        # Read the input image
        image = cv2.imread(image_path)

        # Draw the point and bounding box on the image
        thickness = 2
        for p in model_projected_vertices:
            image = cv2.circle(image, (int(p[0]), int(p[1])), 4, (0, 255, 0), thickness)

        image = cv2.polylines(image, [bbox_2d], True, (255, 255, 0), thickness)
        image = draw_3d_bbox(image, projected_bbox_3d_vertices, (0, 0, 255))
        
        # Display the image
        cv2.imshow('Image', image)
        
        # Wait for a 'q' key press to close the image window
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    return bbox_2d, bbox_3d, projected_bbox_3d_vertices



def process_folder(folder_name:str) -> None:
    """
    Processes a folder containing scene data by generating bounding boxes for each scene.

    Parameters:
        folder_name (str): Name of the folder containing scene data.

    Returns:
        None
    """

    thread_id = threading.get_ident()
    folder_name_path = os.path.join(GENERATED_SCENES_PATH, folder_name)

    if not os.path.isdir(folder_name_path):
        return

    print(f'\n\n --- Generating bboxes for sequence {folder_name} by thread {thread_id} ---\n\n')

    start = time()

    # Iterate through files in the directory
    for file_name in os.listdir(folder_name_path):

        # Check if the file is an npy file
        if not file_name.endswith('.npy') or '79' not in file_name:
            continue

        # Extract scene ID from file name
        scene_id = file_name.split('-')[0]

        # Load metadata from the file
        metadata = np.load(os.path.join(folder_name_path, file_name), allow_pickle=True).item()

        bboxes_2d = {}
        bboxes_3d = {}
        bboxes_3d_proj = {}
        count_object_id = 0
            
        # Iterate through class IDs in the metadata
        new_class_ids = []
        new_poses = []
        for class_id in metadata['cls_indexes']:

            # Generate bounding box for the object
            model_name = get_model_name_from_id(class_id)

            bboxes_2d[model_name] = []
            bboxes_3d[model_name] = []
            bboxes_3d_proj[model_name] = []

            bbox_2d, bbox_3d, projected_bbox_3d_vertices = generated_bboxes(model_paths,
                                    metadata['blendercam_in_world'],
                                    metadata['poses'][count_object_id],
                                    os.path.join(folder_name_path, f'{scene_id}-color.png'),
                                    model_name,
                                    metadata['intrinsic_matrix'], 
                                    config_file['camera_settings']['width']-15, # This -15 is chosen in an empirical way, it may depend on the rounding or conversion that are not precise.
                                    config_file['camera_settings']['height'],
                                    config_file['bbox_adjustment_2d'],
                                    config_file['bbox_adjustment_3d'],
                                    True)

            #if is_box_inside((x1, y1, x2, y2)):
            bboxes_2d[model_name].append(bbox_2d)
            bboxes_3d[model_name].append(bbox_3d)
            bboxes_3d_proj[model_name].append(projected_bbox_3d_vertices)
            new_class_ids.append(class_id)
            new_poses.append(metadata['poses'][count_object_id])
                
            count_object_id += 1


        with open(os.path.join(folder_name_path, f'{scene_id}-box-2d.txt'), 'w') as f:
            for model in bboxes_2d:
                for model_instance in bboxes_2d[model]:
                    bbox_coords = f'{model_instance[0][0]} {model_instance[0][1]} {model_instance[2][0]} {model_instance[2][1]}'
                    f.write(f"{model[:-4]} {bbox_coords}\n")

        with open(os.path.join(folder_name_path, f'{scene_id}-box-3d.txt'), 'w') as f:
            for model in bboxes_3d:
                for model_instance in bboxes_3d[model]:
                    bbox_coords = ''
                    for c in model_instance:
                        bbox_coords += f'{c} '
                    f.write(f"{model[:-4]} {bbox_coords}\n")

        with open(os.path.join(folder_name_path, f'{scene_id}-box-3d-proj.txt'), 'w') as f:
            for model in bboxes_3d:
                for model_instance in bboxes_3d_proj[model]:
                    bbox_coords = ''
                    for c in model_instance:
                        bbox_coords += f'[{c[0]} {c[1]}] '
                    f.write(f"{model[:-4]} {bbox_coords}\n")
            
            
        data_dict = {
            'cls_indexes': new_class_ids,
            'poses':new_poses,
            'blendercam_in_world': metadata['blendercam_in_world'],
            'intrinsic_matrix': metadata['intrinsic_matrix']
        }

        # Save the dictionary as a .npy file
        npy_filename = os.path.join(folder_name_path, file_name)
        np.save(npy_filename, data_dict)

    delta = int(time() - start)
    print(f'Box generation for sequence {folder_name} done in {delta} seconds by thread {thread_id}')



if __name__ == '__main__':

    # Define paths
    CURRENT_DIR_PATH = os.path.dirname(__file__)
    CONFIG_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'scene_generation.yml')

    with open(CONFIG_PATH, 'r') as f:
        config_file = yaml.safe_load(f)

    dataset_name = config_file['dataset_name']

    GENERATED_SCENES_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets', dataset_name, 'GeneratedScenes')
    
    OBJECT_MODELS_DIR_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets', dataset_name, 'Models')

    model_folders = os.listdir(OBJECT_MODELS_DIR_PATH)
    model_paths = []
    for model_folder in model_folders:
        model_folder = os.path.join(OBJECT_MODELS_DIR_PATH, model_folder)
        model_name = [f for f in os.listdir(model_folder) if f.endswith('.obj')][0]
        model_paths.append(os.path.join(model_folder, model_name))

    folder_list = os.listdir(GENERATED_SCENES_PATH)[-1:]
    folder_list.sort()

    # Use ThreadPoolExecutor to process each folder in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Map the process_folder function to each folder in the folder_list
        results = list(executor.map(process_folder, folder_list))