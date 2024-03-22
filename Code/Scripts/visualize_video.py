import cv2
import os

def generate_video_from_frames(directory, id_scene, fps=24):
    
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
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':

    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'Data')

    input_directory = os.path.join(DATA_PATH, 'GeneratedScenes')
    for id_scene in os.listdir(input_directory):

        generate_video_from_frames(input_directory, id_scene)
