import os
import sys
import string
import shutil
import re
import glob
import random
import math
from typing import *

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)

import bpy
import cv2
from PIL import Image
import yaml
import numpy as np
import transformations as T
from mathutils import Matrix, Vector



def read_exr(exr_dir):
    return cv2.imread(exr_dir, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)



def matrix_to_numpy_array(mat):
    """
    Convert a 4x4 matrix represented as a list of lists into a NumPy array.

    Args:
    - mat (list of lists): A 4x4 matrix represented as a list of lists.

    Returns:
    - new_mat (numpy.ndarray): A NumPy array representing the input matrix.

    Note:
    The input matrix should be a list of lists with dimensions 4x4.
    """

    new_mat = np.array(
        [
            [mat[0][0], mat[0][1], mat[0][2], mat[0][3]],
            [mat[1][0], mat[1][1], mat[1][2], mat[1][3]],
            [mat[2][0], mat[2][1], mat[2][2], mat[2][3]],
            [mat[3][0], mat[3][1], mat[3][2], mat[3][3]],
        ]
    )

    return new_mat



def numpy_array_to_matrix(array):
    """
    Convert a NumPy array to a 4x4 matrix.

    Args:
    - array (numpy.ndarray): A NumPy array representing a 4x4 matrix.

    Returns:
    - mat (Matrix): A 4x4 matrix object.

    Note:
    The input array should be a NumPy array with dimensions 4x4.
    """

    mat = Matrix(
        (
            (array[0, 0], array[0, 1], array[0, 2], array[0, 3]),
            (array[1, 0], array[1, 1], array[1, 2], array[1, 3]),
            (array[2, 0], array[2, 1], array[2, 2], array[2, 3]),
            (array[3, 0], array[3, 1], array[3, 2], array[3, 3]),
        )
    )

    return mat



def change_environment_light(dataset_info) -> None:
    """
    Change the environment light (sun) in Blender according to the provided dataset information.

    Args:
    - dataset_info (dict): A dictionary containing information about the dataset and Blender settings.
    """

    sun_energy = dataset_info['blender_settings']['sun_energy']
    sun_lamp = bpy.data.objects.get("Sun")
    sun_lamp.data.energy = np.random.uniform(sun_energy[0], sun_energy[1])
   
    sun_color_r = np.random.uniform(
        dataset_info['blender_settings']['sun_color'][0][0],
        dataset_info['blender_settings']['sun_color'][0][1],
    )
    sun_color_g = np.random.uniform(
        dataset_info['blender_settings']['sun_color'][1][0],
        dataset_info['blender_settings']['sun_color'][1][1],
    )
    sun_color_b = np.random.uniform(
        dataset_info['blender_settings']['sun_color'][2][0],
        dataset_info['blender_settings']['sun_color'][2][1],
    )
    sun_lamp.color = (sun_color_r, sun_color_g, sun_color_b, 1)



def reset(dataset_info) -> None:
    """
    Reset the Blender scene based on the provided dataset information.

    Args:
    - dataset_info (dict): A dictionary containing information about the dataset and Blender settings.
    """

    change_environment_light(dataset_info)

    # Deselect all objects
    for ob in bpy.data.objects:
        ob.select = False
    
    # Delete all lamps except the Sun lamp
    for ob in bpy.data.objects:
        if ob.type == 'LAMP' and 'Sun' not in ob.name:
            ob.select = True
            bpy.ops.object.delete()
        else:
            ob.select = False

    # TODO: Reset locations of objects named 'ob' to (9999, y, z)
    # for ob in bpy.data.objects:
    #     if 'ob' in ob.name.lower():
    #         ob.location[0] = 9999



def setup_camera(H, W, K) -> None:
    """
    Setup camera parameters in Blender.

    Args:
    - H (int): Height of the render resolution.
    - W (int): Width of the render resolution.
    - K (numpy.ndarray): Camera intrinsic matrix.
    """

    # Set render resolution
    bpy.context.scene.render.resolution_x = W
    bpy.context.scene.render.resolution_y = H

    # Access camera data
    cam_data = bpy.data.objects['Camera'].data

    # Retrieve sensor width in millimeters
    sensor_width_in_mm = cam_data.sensor_width

    # Set camera shift based on the principal point
    cam_data.shift_x = -(K[0, 2] - 0.5 * W) / W
    cam_data.shift_y = (K[1, 2] - 0.5 * H) / W

    # Set camera lens based on the focal length and sensor width
    cam_data.lens = K[0, 0] / W * sensor_width_in_mm

    # Calculate pixel aspect ratio
    pixel_aspect = K[1, 1] / K[0, 0]

    # Set pixel aspect ratio for rendering
    bpy.context.scene.render.pixel_aspect_x = 1.0
    bpy.context.scene.render.pixel_aspect_y = pixel_aspect

    # Set the camera as the active camera in the scene
    bpy.context.scene.camera = bpy.data.objects['Camera']

    # Update the scene
    bpy.context.scene.update()



def place_object(ob_name: str, pose: np.ndarray) -> None:
    """
    Places an object in the scene at a specified pose.

    Args:
        - ob_name (str): The name of the object to be placed.
        - pose (numpy.ndarray): The pose matrix defining the object's position and orientation.
    """

    # Get the object from the Blender data
    ob = bpy.data.objects[ob_name]

    # Convert the pose numpy array to a Blender matrix
    pose_mat = numpy_array_to_matrix(pose)

    # Set the object's world matrix to the pose matrix
    ob.matrix_world = pose_mat

    # Update the scene to reflect the changes
    bpy.context.scene.update()



def add_light_and_place(dataset_info: dict, num: int) -> None:
    """
    Adds point lights to the scene and places them randomly within specified ranges.

    Args:
        - dataset_info (dict): A dictionary containing dataset information including Blender settings.
        - num (int): The number of lights to add to the scene.
    """

    # Add the specified number of point lights to the scene
    for _ in range(num):
        bpy.ops.object.lamp_add(type='POINT', view_align=False)

    # Iterate through all objects in the scene
    for ob in bpy.data.objects:

        # Check if the object is a point light
        if 'Point' in ob.name:

            # Retrieve lamp brightness and position range from dataset info
            lamp_brightness = dataset_info['blender_settings']['lamp_brightness']
            pos_ranges = dataset_info['blender_settings']['lamp_pos_range']

            # Generate random positions within the specified ranges
            lx = np.random.uniform(pos_ranges[0][0], pos_ranges[0][1])
            ly = np.random.uniform(pos_ranges[1][0], pos_ranges[1][1])
            lz = np.random.uniform(pos_ranges[2][0], pos_ranges[2][1])

            # Generate random strength and colors for the light
            strength = np.random.uniform(lamp_brightness[0], lamp_brightness[1])
            light_color_ranges = dataset_info['blender_settings']['lamp_colors']
            r = np.random.uniform(light_color_ranges[0][0], light_color_ranges[0][1])
            g = np.random.uniform(light_color_ranges[1][0], light_color_ranges[1][1])
            b = np.random.uniform(light_color_ranges[2][0], light_color_ranges[2][1])

            # Set light properties
            ob.location = [lx, ly, lz]
            ob.data.use_specular = False
            ob.data.shadow_method = 'RAY_SHADOW'
            ob.data.energy = strength
            ob.data.color = (r, g, b)
            ob.data.shadow_ray_samples = 6
            ob.data.shadow_ray_sample_method = 'ADAPTIVE_QMC'



def load_object_model(model_path: str, index: int, name: str) -> None:
    """
    Loads an object model from an OBJ file and assigns a material to it.

    Args:
        - model_path (str): The file path to the OBJ model.
        - index (int): Index to assign to the object.
        - name (str): Name to assign to the object.
    """

    print('Loading object ', model_path)

    # Import the OBJ model
    bpy.ops.import_scene.obj(filepath=model_path)
    
    # Get the imported object
    ob = bpy.context.selected_objects[0]
    
    # Check if the object has no materials assigned
    if len(ob.data.materials) == 0:
        # Create a new material
        mat_name = 'Material'
        mat = bpy.data.materials.new(name=mat_name)
        # Assign the material to the object
        ob.data.materials.append(mat)

    # Get the material of the object
    mat = ob.data.materials[0]
    
    # Add a texture slot to the material
    slot = mat.texture_slots.add()
    
    # Set the pass index of the object
    imported = bpy.context.selected_objects[0]
    imported.pass_index = index
    
    # Set the initial location of the object to be far from the scene
    imported.location[0] = 9999
    
    # Set the name of the object
    imported.name = str(name)



def change_object_texture(ob_name: str, image_dir: str) -> None:
    """
    Changes the texture of an object.

    Args:
        - ob_name (str): The name of the object.
        - image_dir (str): The directory path to the image texture.
    """

    # Get the object from Blender data
    ob = bpy.data.objects[ob_name]

    # Check if the object has no materials assigned
    if len(ob.data.materials) == 0:
        # Create a new material
        mat_name = 'Material'
        mat = bpy.data.materials.new(name=mat_name)
        # Assign the material to the object
        ob.data.materials.append(mat)

    # Get the material of the object
    mat = ob.data.materials[0]

    # Disable use of nodes for the material
    mat.use_nodes = False
    
    # Load the image texture
    img = bpy.data.images.load(image_dir)
    
    # Create a new texture
    tex_name = 'Texture'
    tex = bpy.data.textures.new(tex_name, 'IMAGE')
    tex.image = img

    # Get the texture slot of the material
    slot = mat.texture_slots[0]

    # Assign the texture to the texture slot
    slot.texture = tex

    # Update the scene
    bpy.context.scene.update()

    # Set texture coordinates to 'OBJECT'
    ob.active_material.texture_slots[0].texture_coords = 'OBJECT'

    # Set texture scale
    ob.active_material.texture_slots[0].scale[0] = 4
    ob.active_material.texture_slots[0].scale[1] = 4



def random_string(size: int) -> str:
    """
    Generates a random string of specified size using uppercase letters and digits.

    Args:
        - size (int): The length of the random string to generate.

    Returns:
        str: The randomly generated string.
    """

    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(np.random.choice(chars) for _ in range(size))



def render(id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Renders images (RGB, depth, segmentation) for the given scene ID.

    Args:
        - id (str): The identifier for the scene.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing RGB, depth, and segmentation images.
    """

    # Set up the output directory for rendered images
    current_dir = os.path.dirname(__file__)
    out_dir = os.path.join(current_dir, 'tmp', random_string(20)) + os.path.sep
    os.makedirs(out_dir, exist_ok=True)

    # Disable node usage for objects named 'ob'
    for ob in bpy.data.objects:
        if 'ob' in ob.name:
            ob.active_material.use_nodes = False

    # Set render quality and use OpenCL for rendering
    tree = bpy.context.scene.node_tree
    tree.render_quality = 'HIGH'
    tree.edit_quality = 'HIGH'
    tree.use_opencl = True

    links = tree.links

    # Clear all existing nodes in the scene
    for n in tree.nodes:
        tree.nodes.remove(n)

    # Add nodes for rendering RGB, depth, and segmentation images
    render_node = tree.nodes.new('CompositorNodeRLayers')
    rgb_node = tree.nodes.new('CompositorNodeOutputFile')
    rgb_node.format.file_format = 'PNG'
    rgb_node.base_path = out_dir
    rgb_node.file_slots[0].path = '{}rgbB'.format(id)
    links.new(render_node.outputs['Image'], rgb_node.inputs[0])

    depth_node = tree.nodes.new('CompositorNodeOutputFile')
    depth_node.format.file_format = 'OPEN_EXR'
    depth_node.base_path = out_dir
    depth_node.file_slots[0].path = '{}depthB'.format(id)
    links.new(render_node.outputs['Depth'], depth_node.inputs[0])

    seg_node = tree.nodes.new('CompositorNodeOutputFile')
    seg_node.format.file_format = 'OPEN_EXR'
    seg_node.base_path = out_dir
    seg_node.file_slots[0].path = '{}segB'.format(id)
    links.new(render_node.outputs['IndexOB'], seg_node.inputs[0])

    # Render the scene
    bpy.ops.render.render(write_still=False)

    # Determine the index of the rendered images
    index = re.findall(r'depthB\d{4}', glob.glob(out_dir + '*depthB*.exr')[0])[0].replace('depthB', '')

    # Read the rendered images into numpy arrays
    rgbB = np.array(Image.open(out_dir + '{}rgbB{}.png'.format(id, index)))[:, :, :3]
    depth_meter = read_exr(out_dir + '{}depthB{}.exr'.format(id, index))[:, :, 0]
    depth_meter[depth_meter < 0.1] = 0
    depth_meter[depth_meter > 2.0] = 0
    depthB = (depth_meter * 1000).astype(np.uint16)
    segB = read_exr(out_dir + '{}segB{}.exr'.format(id, index)).astype(np.uint8)

    # Clean up the temporary directory
    shutil.rmtree(os.path.join(current_dir, 'tmp'))

    return rgbB, depthB, segB



def get_dynamic_objects() -> List[bpy.types.Object]:
    """
    Retrieves dynamic objects from the scene.

    Returns:
        list: A list of dynamic objects in the scene.
    """

    # Initialize an empty list to store dynamic objects
    obs = []
    # Iterate through all objects in the scene
    for ob in bpy.data.objects:
        # Check if the object is not a camera, light, plane, or sun
        if (
            'Camera' not in ob.name
            and 'Point' not in ob.name
            and 'box_plane' not in ob.name
            and 'Sun' not in ob.name
            and 'Bezier' not in ob.name
        ):
            # Append the object to the list of dynamic objects
            obs.append(ob)

    return obs



def set_objects_parameters() -> Tuple[Dict[int, bpy.types.Object], np.ndarray]: 
    """
    Sets parameters for dynamic objects in the scene.

    Returns:
        Tuple[Dict[int, bpy.types.Object], np.ndarray]: A tuple containing a dictionary mapping object IDs to objects
        and an array of class IDs.
    """

    # Initialize a dictionary to map object IDs to objects
    id2ob = {}

    # Get dynamic objects in the scene
    obs = get_dynamic_objects()
    
    # Iterate through each dynamic object
    for ob in obs:
        # Set the object as the active object in the scene
        bpy.context.scene.objects.active = ob
        # Add rigid body physics to the object
        bpy.ops.rigidbody.object_add(type='ACTIVE')
        
        # Check if the object already has a 'COLLISION' modifier
        collision_modifier = None
        for modifier in ob.modifiers:
            if modifier.type == 'COLLISION':
                collision_modifier = modifier
                break
        
        # If the object does not have a 'COLLISION' modifier, add one
        if collision_modifier is None:
            bpy.ops.object.modifier_add(type='COLLISION')
        
        # Set rigid body parameters for the object
        ob.rigid_body.mass = 10.0
        ob.rigid_body.use_margin = True
        ob.rigid_body.collision_margin = 1e-4
        ob.rigid_body.linear_damping = 0.01
        ob.rigid_body.angular_damping = 0.01
        ob.rigid_body.friction = 0.01
        ob.collision.absorption = 0.01
        ob.collision.friction_factor = 0.01
        ob.rigid_body.restitution = 0.99
        ob.data.materials[0].ambient = 0.2
        ob.layers[0] = True

        # Get the class ID of the object
        class_id = int(ob.pass_index)

        # Map the class ID to the object in the dictionary (255 as all the objects are insert with an ID < 255, 
        #all the others are lights and blender additional stuffs)
        if class_id < 255:
            id2ob[class_id] = ob
        
    # Get the list of class IDs
    class_ids = np.array(list(id2ob.keys()))
    print('class_ids',class_ids)

    return id2ob, class_ids



def remove_objects(class_ids: list) -> None:
    """
    Removes objects from the scene based on their class IDs.

    Args:
        class_ids (list): A list of class IDs of objects to be removed.
    """

    # Iterate through all objects in the scene
    for ob in bpy.data.objects:
        # Check if the object's class ID is in the list of class IDs to be removed
        if ob.pass_index in class_ids:
            # Print a message indicating the object is being removed
            print('Removed ', ob.pass_index, ' ', ob.name)
            # Remove the object from the scene
            bpy.data.objects.remove(ob, do_unlink=True)



def generate_random_point_in_sphere(center: Vector, min_radius: float, max_radius: float) -> Vector:
    """
    Generates a random point within a sphere defined by a center and minimum and maximum radii.

    Args:
        - center (Vector): The center of the sphere.
        - min_radius (float): The minimum radius of the sphere.
        - max_radius (float): The maximum radius of the sphere.

    Returns:
        Vector: A random point within the sphere.
    """

    # Generate random spherical coordinates
    radius = random.uniform(min_radius, max_radius)
    theta = random.uniform(0, math.pi)
    phi = random.uniform(0, 2 * math.pi)

    # Convert spherical coordinates to Cartesian coordinates
    x = center[0] + radius * math.sin(theta) * math.cos(phi)
    y = center[1] + radius * math.sin(theta) * math.sin(phi)
    z = center[2] + radius * math.cos(theta)

    # Ensure that the object never goes below the horizontal plane (z >= 0)
    z = max(z, 0)

    return Vector((x, y, z))



def move_camera_smoothly(starting_frame, ending_frame) -> None:
    """
    Moves the camera smoothly from one position to another over a range of frames.

    Args:
        starting_frame (int): The starting frame of the animation.
        ending_frame (int): The ending frame of the animation.
    """

    # Get the camera object
    camera = bpy.data.objects['Camera']

    # Generate a random target point within a sphere
    target_point = generate_random_point_in_sphere((0, 0, 0), 1, 2.8)

    # Get the initial position of the camera
    initial_position = Vector(camera.location)

    # Define the point the camera should be looking at
    looking_point = Vector((0, 0, 0.5))
    
    # Calculate the number of frames in the animation
    num_frames = ending_frame - starting_frame
    
    # Create a Bezier curve to interpolate camera movement
    bezier_curve = bpy.data.curves.new(name="BezierCurve", type='CURVE')
    bezier_curve.dimensions = '3D'
    bezier_curve.resolution_u = 2
    spline = bezier_curve.splines.new('BEZIER')
    spline.bezier_points.add(1)
    spline.bezier_points[0].co = initial_position
    spline.bezier_points[1].co = target_point
    bezier_obj = bpy.data.objects.new("BezierCurveObj", bezier_curve)
    bpy.context.scene.objects.link(bezier_obj)

    # Iterate over each frame in the specified range
    for frame in range(starting_frame, ending_frame + 1):
        
        print('Generating keyframe for frame: {}'.format(frame))
        
        # Calculate the position along the Bezier curve for this frame
        t = (frame - starting_frame) / num_frames
        new_position = interpolate_bezier(bezier_obj, t)
        camera.location = new_position
        
        # Calculate the direction the camera should be pointing
        direction = (looking_point - new_position).normalized()
        rotation = direction.to_track_quat('-Z', 'Y')
        
        # Apply the rotation to the camera
        camera.rotation_quaternion = rotation

        # Set keyframes for position and rotation
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # Set the current frame
        bpy.context.scene.frame_set(frame)



def interpolate_bezier(bezier_obj: bpy.types.Object, t: float) -> Vector:
    """
    Interpolates points along a Bezier curve.

    Args:
        - bezier_obj (bpy.types.Object): The Blender object representing the Bezier curve.
        - t (float): The parameter value ranging from 0 to 1 indicating the position along the curve.

    Returns:
        Vector: The interpolated point along the Bezier curve.
    """

    # Get the spline from the Bezier curve object
    spline = bezier_obj.data.splines[0]
    # Interpolate between the two control points of the Bezier curve
    point = spline.bezier_points[0].co.lerp(spline.bezier_points[1].co, t)
    # Get the translation vector of the Bezier curve object
    translation_vector = bezier_obj.matrix_world.to_translation()
    # Add the translation vector to the interpolated point to get the final position
    return translation_vector + point



def generate() -> None:
    """
    Generates scenes using Blender based on provided configurations.
    """
    
    # Set up all the paths and configurations vars
    CURRENT_DIR_PATH = os.path.dirname(__file__)
    CONFIG_FILE_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'scene_generation.yml')

    with open(CONFIG_FILE_PATH, 'r') as f:
        config_file = yaml.safe_load(f)

    dataset_name = config_file['dataset_name']

    MODELS_ID_FILE_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'models_id.yml')
    with open(MODELS_ID_FILE_PATH, 'r') as f:
        models_id = yaml.safe_load(f)

    OUTPUT_DIRECTORY = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'GeneratedScenes')
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    BACKGROUND_FOLDER = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'BackgroundTextures')
    OBJECT_MODELS_DIR_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets', dataset_name, 'Models')

    num_scenes = config_file['num_scenes_to_generate']
    fps = config_file['fps']
    cam_movements_per_scene = config_file['cam_movements_per_scene']
    cam_movements_duration_in_seconds = config_file['cam_movements_duration_in_seconds']

    xmin = config_file['blender_settings']['range_x'][0]
    xmax = config_file['blender_settings']['range_x'][1]
    ymin = config_file['blender_settings']['range_y'][0]
    ymax = config_file['blender_settings']['range_y'][1]
    zmin = config_file['blender_settings']['range_z'][0]
    zmax = config_file['blender_settings']['range_z'][1]

    camera_height = config_file['camera_settings']['height']
    camera_width = config_file['camera_settings']['width']
    K = np.eye(3)
    K[0, 0] = config_file['camera_settings']['focalX']
    K[1, 1] = config_file['camera_settings']['focalY']
    K[0, 2] = config_file['camera_settings']['centerX']
    K[1, 2] = config_file['camera_settings']['centerY']
    K[1, 1] = K[0, 0]


    print('\n --- Setting the camera values --- \n')
    setup_camera(W=camera_width, H=camera_height, K=K)
    

    print('\n --- Collecting texture files --- \n')
    background_files = []
    for background_file_name in os.listdir(BACKGROUND_FOLDER):
        background_files.append(os.path.join(BACKGROUND_FOLDER, background_file_name))


    if len(background_files) == 0:
        print('\n !!! No background textures found, quitting !!! \n')
        sys.exit()



    print('\n --- Loading models in the scene --- \n')

    # Get the abs paths of the object models
    object_models = models_id.keys()

    model_folders = os.listdir(OBJECT_MODELS_DIR_PATH)
    object_model_files = []
    for model_folder in model_folders:
        model_folder = os.path.join(OBJECT_MODELS_DIR_PATH, model_folder)
        model_name = [f for f in os.listdir(model_folder) if f.endswith('.obj')][0]
        if model_name in object_models:
            object_model_files.append(os.path.join(model_folder, model_name))

    # List all directories in the given directory
    folders = [folder for folder in os.listdir(OUTPUT_DIRECTORY)]

    if folders == []:
        starting_num_scenes = 0
    else:
        # Find the starting numerical folder
        folders.sort()
        starting_num_scenes = int(folders[-1]) + 1

    ending_num_scenes = starting_num_scenes + num_scenes

    for i in range(starting_num_scenes, ending_num_scenes):

        os.makedirs(os.path.join(OUTPUT_DIRECTORY, '{:04d}'.format(i)))

        # Pick a random number of elements from 1 to the length of object_model_files
        num_elements = np.random.randint(1, len(object_model_files) + 1)
            
        # Randomly select num_elements from object_model_files
        random_models = np.random.choice(object_model_files, num_elements, replace=False)

        # Add the selected models in the blender scene
        for model_file_path in random_models:
            file_name = os.path.basename(model_file_path)
            load_object_model(model_file_path, models_id[file_name], models_id[file_name])
        
        # Reset the blender scene lights
        reset(config_file)

        # Set the models parameters
        idx_to_object, class_ids = set_objects_parameters()

        # Add new ligths
        light_num = np.random.randint(0, config_file['blender_settings']['max_lamp_num'] + 1)
        add_light_and_place(config_file, light_num)

        # Add the background textures to the planes in the blender scene
        for ob in bpy.data.objects:
            if 'box_plane' in ob.name:
                texture_file = np.random.choice(background_files)
                print('Box plane: {} using texture file\n {}'.format(ob.name, texture_file))
                change_object_texture(ob.name, texture_file)

        
        # Randomize the object position in the scene
        obs = get_dynamic_objects()
        for ob in obs:
            pose = np.eye(4)
            pose[0, 3] = np.random.uniform(xmin, xmax)
            pose[1, 3] = np.random.uniform(ymin, ymax)
            pose[2, 3] = np.random.uniform(zmin, zmax)
            pose[:3, :3] = T.random_rotation_matrix()[:3, :3]
            place_object(ob.name, pose)


        print('Start gravity simulation')
        bpy.context.scene.gravity = np.random.uniform(-2, 2, size=3)

        # Simulate the graviti for 6 frames
        for ii in range(1, 6):
            bpy.context.scene.frame_set(ii)
        
        # Turn off the graviti
        bpy.context.scene.gravity = (0, 0, 0)

        # Iterate through all selected objects
        for obj in obs:
            # Lock location, rotation, and scale
            obj.lock_location[0] = obj.lock_location[1] = obj.lock_location[2] = True
            obj.lock_rotation[0] = obj.lock_rotation[1] = obj.lock_rotation[2] = True
            obj.lock_scale[0] = obj.lock_scale[1] = obj.lock_scale[2] = True

        # Update the scene configuration
        bpy.context.scene.update()

        # Define the number of camera movements in the currect scene
        num_movements = random.randint(cam_movements_per_scene[0], cam_movements_per_scene[1])
        print('\n\n --- Camera movements for the scene: {} ---\n\n'.format(num_movements))
        total_frame_num = 0

        for _ in range(num_movements):
            # Define the movement duration
            movement_duration_in_frames = random.randint(cam_movements_duration_in_seconds[0]*fps, cam_movements_duration_in_seconds[1]*fps)
            
            starting_frame = total_frame_num
            total_frame_num += movement_duration_in_frames
            ending_frame = total_frame_num

            # Create the keyframes for the movement
            move_camera_smoothly(starting_frame, ending_frame)


        # Generate the scene
        for count in range(total_frame_num):
    
            print('\n\n\n\n --- Generated ' + str(count)+'/'+str(total_frame_num)+' frames --- \n\n\n\n')
            
            # Set the frame
            bpy.context.scene.frame_set(count)

            # Get the 6D matrix pose of the camera
            blendercam_in_world = matrix_to_numpy_array(bpy.data.objects['Camera'].matrix_world)

            # Render the images
            rgbB, depthB, segB = render(count)

            # Save the images
            rgb_filename = os.path.join(OUTPUT_DIRECTORY, '{:04d}'.format(i), '{:07d}-color.png'.format(count))
            depth_filename = os.path.join(OUTPUT_DIRECTORY, '{:04d}'.format(i), '{:07d}-depth.png'.format(count))
            seg_filename = os.path.join(OUTPUT_DIRECTORY, '{:04d}'.format(i), '{:07d}-seg.png'.format(count))

            Image.fromarray(rgbB).save(rgb_filename, optimize=True)
            cv2.imwrite(depth_filename, depthB.astype(np.uint16))
            cv2.imwrite(seg_filename, segB.astype(np.uint8))

            # Update the scene
            bpy.context.scene.update()

            # Get the 6D poses of the custom models in the scene
            poses_in_world = []
            for class_id in class_ids:
                ob = idx_to_object[class_id]
                ob_in_world = matrix_to_numpy_array(ob.matrix_world)
                poses_in_world.append(ob_in_world)
            poses_in_world = np.array(poses_in_world)


            # Create a dictionary containing the arrays
            data_dict = {
                'class_ids': class_ids,
                'poses_in_world': poses_in_world,
                'blendercam_in_world': blendercam_in_world,
                'K': K
            }

            # Save the dictionary as a .npy file
            npy_filename = os.path.join(OUTPUT_DIRECTORY, '{:04d}'.format(i), '{:07d}_poses_in_world.npy'.format(count))
            np.save(npy_filename, data_dict)
        

        # Remove custom models from the scene 
        remove_objects(class_ids)


    print('Finished {}'.format(OUTPUT_DIRECTORY))


if __name__ == '__main__':
    generate()