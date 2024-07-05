import os
import yaml
import open3d as o3d
import sys
import numpy as np
import trimesh
import json

import math
import struct

def load_ply(path):
    """Loads a 3D mesh model from a PLY file.

    :param path: Path to a PLY file.
    :return: The loaded model given by a dictionary with items:
     - 'pts' (nx3 ndarray)
     - 'normals' (nx3 ndarray), optional
     - 'colors' (nx3 ndarray), optional
     - 'faces' (mx3 ndarray), optional
     - 'texture_uv' (nx2 ndarray), optional
     - 'texture_uv_face' (mx6 ndarray), optional
     - 'texture_file' (string), optional
    """
    f = open(path, "rb")

    # Only triangular faces are supported.
    face_n_corners = 3

    n_pts = 0
    n_faces = 0
    pt_props = []
    face_props = []
    is_binary = False
    header_vertex_section = False
    header_face_section = False
    texture_file = None

    # Read the header.
    while True:
        # Strip the newline character(s).
        line = f.readline().decode("utf8").rstrip("\n").rstrip("\r")

        if line.startswith("comment TextureFile"):
            texture_file = line.split()[-1]
        elif line.startswith("element vertex"):
            n_pts = int(line.split()[-1])
            header_vertex_section = True
            header_face_section = False
        elif line.startswith("element face"):
            n_faces = int(line.split()[-1])
            header_vertex_section = False
            header_face_section = True
        elif line.startswith("element"):  # Some other element.
            header_vertex_section = False
            header_face_section = False
        elif line.startswith("property") and header_vertex_section:
            # (name of the property, data type)
            pt_props.append((line.split()[-1], line.split()[-2]))
        elif line.startswith("property list") and header_face_section:
            elems = line.split()
            if elems[-1] == "vertex_indices" or elems[-1] == "vertex_index":
                # (name of the property, data type)
                face_props.append(("n_corners", elems[2]))
                for i in range(face_n_corners):
                    face_props.append(("ind_" + str(i), elems[3]))
            elif elems[-1] == "texcoord":
                # (name of the property, data type)
                face_props.append(("texcoord", elems[2]))
                for i in range(face_n_corners * 2):
                    face_props.append(("texcoord_ind_" + str(i), elems[3]))
            else:
                print("Warning: Not supported face property: " + elems[-1])
        elif line.startswith("format"):
            if "binary" in line:
                is_binary = True
        elif line.startswith("end_header"):
            break

    # Prepare data structures.
    model = {}
    if texture_file is not None:
        model["texture_file"] = texture_file
    model["pts"] = np.zeros((n_pts, 3), np.float64)
    if n_faces > 0:
        model["faces"] = np.zeros((n_faces, face_n_corners), np.float64)

    pt_props_names = [p[0] for p in pt_props]
    face_props_names = [p[0] for p in face_props]

    is_normal = False
    if {"nx", "ny", "nz"}.issubset(set(pt_props_names)):
        is_normal = True
        model["normals"] = np.zeros((n_pts, 3), np.float64)

    is_color = False
    if {"red", "green", "blue"}.issubset(set(pt_props_names)):
        is_color = True
        model["colors"] = np.zeros((n_pts, 3), np.float64)

    is_texture_pt = False
    if {"texture_u", "texture_v"}.issubset(set(pt_props_names)):
        is_texture_pt = True
        model["texture_uv"] = np.zeros((n_pts, 2), np.float64)

    is_texture_face = False
    if {"texcoord"}.issubset(set(face_props_names)):
        is_texture_face = True
        model["texture_uv_face"] = np.zeros((n_faces, 6), np.float64)

    # Formats for the binary case.
    formats = {
        "float32": ("f", 4),
        "double": ("d", 8),
        "int": ("i", 4),
        "uchar": ("B", 1),
    }

    # Load vertices.
    for pt_id in range(n_pts):
        prop_vals = {}
        load_props = [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "red",
            "green",
            "blue",
            "texture_u",
            "texture_v",
        ]
        if is_binary:
            for prop in pt_props:
                format = formats[prop[1]]
                read_data = f.read(format[1])
                val = struct.unpack(format[0], read_data)[0]
                if prop[0] in load_props:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().decode("utf8").rstrip("\n").rstrip("\r").split()
            for prop_id, prop in enumerate(pt_props):
                if prop[0] in load_props:
                    prop_vals[prop[0]] = elems[prop_id]

        model["pts"][pt_id, 0] = float(prop_vals["x"])
        model["pts"][pt_id, 1] = float(prop_vals["y"])
        model["pts"][pt_id, 2] = float(prop_vals["z"])

        if is_normal:
            model["normals"][pt_id, 0] = float(prop_vals["nx"])
            model["normals"][pt_id, 1] = float(prop_vals["ny"])
            model["normals"][pt_id, 2] = float(prop_vals["nz"])

        if is_color:
            model["colors"][pt_id, 0] = float(prop_vals["red"])
            model["colors"][pt_id, 1] = float(prop_vals["green"])
            model["colors"][pt_id, 2] = float(prop_vals["blue"])

        if is_texture_pt:
            model["texture_uv"][pt_id, 0] = float(prop_vals["texture_u"])
            model["texture_uv"][pt_id, 1] = float(prop_vals["texture_v"])

    # Load faces.
    for face_id in range(n_faces):
        prop_vals = {}
        if is_binary:
            for prop in face_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] == "n_corners":
                    if val != face_n_corners:
                        raise ValueError("Only triangular faces are supported.")
                elif prop[0] == "texcoord":
                    if val != face_n_corners * 2:
                        raise ValueError("Wrong number of UV face coordinates.")
                else:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().decode("utf8").rstrip("\n").rstrip("\r").split()
            for prop_id, prop in enumerate(face_props):
                if prop[0] == "n_corners":
                    if int(elems[prop_id]) != face_n_corners:
                        raise ValueError("Only triangular faces are supported.")
                elif prop[0] == "texcoord":
                    if int(elems[prop_id]) != face_n_corners * 2:
                        raise ValueError("Wrong number of UV face coordinates.")
                else:
                    prop_vals[prop[0]] = elems[prop_id]

        model["faces"][face_id, 0] = int(prop_vals["ind_0"])
        model["faces"][face_id, 1] = int(prop_vals["ind_1"])
        model["faces"][face_id, 2] = int(prop_vals["ind_2"])

        if is_texture_face:
            for i in range(6):
                model["texture_uv_face"][face_id, i] = float(
                    prop_vals["texcoord_ind_{}".format(i)]
                )

    f.close()

    return model


def calc_pts_diameter(pts):
    """Calculates the diameter of a set of 3D points (i.e. the maximum distance
    between any two points in the set).

    :param pts: nx3 ndarray with 3D points.
    :return: The calculated diameter.
    """
    diameter = -1.0
    for pt_id in range(pts.shape[0]):
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter

def create_model_info(file_path):

    model = load_ply(file_path)

    # Calculated diameter.
    diameter = calc_pts_diameter(model["pts"])

    # Calculate 3D bounding box.
    ref_pt = list(map(float, model["pts"].min(axis=0).flatten()))
    size = list(map(float, (model["pts"].max(axis=0) - ref_pt).flatten()))

    return {
        "min_x": ref_pt[0],
        "min_y": ref_pt[1],
        "min_z": ref_pt[2],
        "size_x": size[0],
        "size_y": size[1],
        "size_z": size[2],
        "diameter": diameter,
    }

def generate(dataset_name:str) -> None:

    CURRENT_DIR_PATH = os.path.dirname(__file__)
    dataset_path = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Datasets', dataset_name)

    if not os.path.exists(dataset_path):
        print('\n\n!!! The dataset you specified does not exists !!!\n\n')
        sys.exit()

    models_id_yaml_file = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'models_id.yml')
    with open(models_id_yaml_file, 'r') as f:
        loaded_models = yaml.safe_load(f)

    classes_file = os.path.join(dataset_path, 'classes.txt')
    if os.path.exists(classes_file):
        os.remove(classes_file)

    all_models_loaded = ''
    for model in loaded_models:
        all_models_loaded += f'{model[:-4]}\n'

    with open(classes_file, 'w') as f:
        f.write(all_models_loaded)
    
    print(f'\n\n --- classes.txt created for {dataset_name} ---\n\n')

    model_folders = os.listdir(os.path.join(dataset_path, 'Models'))
    models_info = {}
    for model_folder in model_folders:
        model_folder = os.path.join(dataset_path, 'Models', model_folder)

        model_name = [f for f in os.listdir(model_folder) if f.endswith('.obj')][0]
        mesh = trimesh.load(os.path.join(model_folder, model_name))

        points = ''
        for vertex in mesh.vertices:
            points += f'{vertex[0]} {vertex[1]} {vertex[2]}\n'
        
        with open(os.path.join(model_folder, 'points.xyz'), 'w') as f:
            f.write(points)

        print(f'\n\n --- points.xyz created for {model_name} model ---\n\n')

        model_name = model_name.replace('.obj', '.ply')
        model_path = os.path.join(model_folder, model_name)

        print(f'Creating model info for {model_name}')
        models_info[model_name[:-4]] = create_model_info(model_path)

    output_path = os.path.join(dataset_path, 'models_info.json')
    with open(output_path, 'w') as f:
        json.dump(models_info, f, indent=4)

    file_list_path = os.path.join(dataset_path, 'file_list.txt')
    if os.path.exists(file_list_path):
        os.remove(file_list_path)

    files = ''
    generated_scenes_path = os.path.join(dataset_path, 'GeneratedScenes')
    for sequence in os.listdir(generated_scenes_path):
        sequence_path = os.path.join(generated_scenes_path, sequence)
        for file in os.listdir(sequence_path):
            if not file.endswith('-color.png'):
                continue
            file_id = file.split('-')[0]
            files += f'{sequence}/{file_id}\n'

    with open(file_list_path, 'w') as f:
        f.write(files)

    print(f'\n\n --- files_list.txt created ---\n\n')

if __name__ == '__main__':

    # Define paths
    CURRENT_DIR_PATH = os.path.dirname(__file__)
    CONFIG_PATH = os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'Configs', 'scene_generation.yml')

    with open(CONFIG_PATH, 'r') as f:
        config_file = yaml.safe_load(f)

    dataset_name = config_file['dataset_name']

    generate(dataset_name)