import cv2
import os
import yaml

def draw_bboxes(img, filename: str):

    # Construct the path to the bounding box file
    bbox_file = filename.replace('-color.png', '-box.txt')

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
    
    return img



def generate_video_from_frames(directory:str, id_scene:int, fps=24, draw_boxes=False) -> None:
    
    output_video_path = os.path.join(directory, f'{id_scene}.mp4')

    directory = os.path.join(directory, id_scene)

    frame_files = sorted(os.listdir(directory))
    frame_files = [f for f in frame_files if f.endswith('-color.png')]  # Filter out only PNG files
    frame_files = sorted(frame_files, key=lambda x: int(x.split('-')[0]))  # Sort files numerically
    frame = cv2.imread(os.path.join(directory, frame_files[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for filename in frame_files:

        img = cv2.imread(os.path.join(directory, filename))
        if draw_boxes:
            img = draw_bboxes(img, os.path.join(directory, filename))

        video.write(img)

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':

    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'Data')

    with open(os.path.join(DATA_PATH, 'Configs', 'scene_generation.yml')) as f:
        config_file = yaml.safe_load(f)

    input_directory = os.path.join(DATA_PATH, 'GeneratedScenes')
    for id_scene in os.listdir(input_directory):
        
        if not os.path.isdir(os.path.join(input_directory, id_scene)):
            continue
        
        generate_video_from_frames(input_directory, id_scene, config_file['fps'], True)
