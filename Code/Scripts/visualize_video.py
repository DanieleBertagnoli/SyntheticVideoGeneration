import cv2
import os
import yaml
import numpy as np


def draw_bboxes_2d(img, filename: str, models_id:dict):

    # Construct the path to the bounding box file
    bbox_file = filename.replace('-color.png', '-box-2d.txt')

    meta_filename = filename.replace('-color.png', '-meta.npy')
    metadata = np.load(meta_filename, allow_pickle=True).item()

    # Read bounding box coordinates from the file
    with open(bbox_file, 'r') as f:

        lines = f.readlines()
        for line in lines:
            # Split the line into individual components
            components = line.strip().split()
            if len(components) != 5:
                # Skip invalid lines
                continue
            
            # Extract bounding box coordinates
            model_name, x_min, y_min, x_max, y_max = components

            # Convert coordinates to integers
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            
            # Draw bounding box on the image
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Put model name as label
            cv2.putText(img, model_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
            cx = (x_max + x_min)//2
            cy = (y_max + y_min)//2

            cls_index = list(metadata['cls_indexes']).index(models_id[model_name])
            abs_pose = metadata['poses'][cls_index]
            blendercam = metadata['blendercam_in_world']

            inv_blendercam = np.linalg.inv(blendercam)
            cam_pose = np.dot(inv_blendercam, abs_pose)

            R = cam_pose[0:3, 0:3]
            draw_axes(img, (cx, cy), R)  # Assuming draw_axes is defined elsewhere

    return img



def draw_bboxes_3d(image, filename:str):
    # Points are expected in the order:
    # 0-3: Bottom square in clockwise order starting from the top left corner
    # 4-7: Top square in clockwise order starting from the top left corner

    # Construct the path to the bounding box file
    bbox_file = filename.replace('-color.png', '-box-3d-proj.txt')

    # Read bounding box coordinates from the file
    with open(bbox_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # Split the line into individual components
            components = line.strip().split()
            
            # Extract bounding box coordinates
            model_name = components[0]
            string_points = components[1:]

            points = []
            for i in range(0, len(string_points), 2):
                x = int(string_points[i][1:])
                y = int(string_points[i+1][:-1])
                points.append([x,y])
                
            points = [tuple(map(int, point)) for point in points]

            # Draw bottom square
            points_order = [0,1,3,2]
            for i in range(len(points_order)):
                start_point = points[points_order[i]]
                end_point = points[points_order[(i+1) % len(points_order)]]  # Loop back to the first point
                image = cv2.line(image, start_point, end_point, (0, 0, 255), 2)
            
            # Draw top square
            points_order = [4,5,7,6]
            for i in range(len(points_order)):
                start_point = points[points_order[i]]
                end_point = points[points_order[(i+1) % len(points_order)]]  # Loop back to the first point
                image = cv2.line(image, start_point, end_point, (0, 0, 255), 2)

            # Draw vertical lines (edges)
            for i in range(4):
                bottom_point = points[i]
                top_point = points[i+4]
                image = cv2.line(image, bottom_point, top_point, (0, 0, 255), 2)

    return image
    


def draw_axes(img, center, rot, scale=50):
    # Define unit vectors for axes in 3D
    axes = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])
    point_center = np.array([center[0], center[1], 0])  # Add a dummy zero for the 3D center

    # Colors for the axes: Red for X, Green for Y, Blue for Z
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    # Transform axes using the rotation matrix
    transformed_axes = rot.dot(axes.T).T

    for i, color in enumerate(colors):
        # Calculate the end point of each axis in 3D, then drop the z-component to project it to 2D
        axis_end = point_center + transformed_axes[i]
        cv2.line(img, tuple(point_center[:2].astype(int)), tuple(axis_end[:2].astype(int)), color, 2)



def generate_video_from_frames(input_dir:str, output_dir:str, id_scene:int, models_id:dict, fps=24, draw_boxes_2d=False, draw_boxes_3d=False) -> None:
    
    output_video_path = os.path.join(output_dir, f'{id_scene}.mp4')

    input_dir = os.path.join(input_dir, id_scene)

    frame_files = sorted(os.listdir(input_dir))
    frame_files = [f for f in frame_files if f.endswith('-color.png')]  # Filter out only PNG files
    frame_files = sorted(frame_files, key=lambda x: int(x.split('-')[0]))  # Sort files numerically
    frame = cv2.imread(os.path.join(input_dir, frame_files[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for filename in frame_files:

        img = cv2.imread(os.path.join(input_dir, filename))
        if draw_boxes_2d:
            img = draw_bboxes_2d(img, os.path.join(input_dir, filename), models_id)

        if draw_boxes_3d:    
            img = draw_bboxes_3d(img, os.path.join(input_dir, filename))

        video.write(img)

    cv2.destroyAllWindows()
    video.release()



if __name__ == '__main__':

    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'Data')

    with open(os.path.join(DATA_PATH, 'Configs', 'scene_generation.yml')) as f:
        config_file = yaml.safe_load(f)

    input_directory = os.path.join(DATA_PATH, 'Datasets', config_file['dataset_name'], 'GeneratedScenes')
    output_directory = os.path.join(DATA_PATH, 'Datasets', config_file['dataset_name'], 'Video')
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    models_id_path = os.path.join(DATA_PATH, 'Configs', 'models_id.yml')
    with open(models_id_path, 'r') as f:
        tmp = yaml.safe_load(f)

    models_id = {}
    for k, v in tmp.items():
        models_id[k.split('.')[0]] = v

    for id_scene in sorted(os.listdir(input_directory)):

        print(f'\n\n--- Generating video for {id_scene} scene ---\n\n')

        if not os.path.isdir(os.path.join(input_directory, id_scene)):
            continue
        
        generate_video_from_frames(input_directory, output_directory, id_scene, models_id, config_file['fps'], True, False)
