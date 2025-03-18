import trimesh
import os
import numpy as np
from utils import utils
import cv2


def load_objects_HMDO():
    with open("/HMDO/template_models/obj_list.txt", "r") as file:
        object_names = file.read().splitlines()
        #print(object_names)
    root= '/HMDO/template_models/'
    obj_pc, obj_face, obj_scale, obj_pc_resampled, obj_resampled_faceid = {}, {}, {}, {}, {}
    for obj_name in object_names:
        obj_mesh = trimesh.load_mesh(root+obj_name) 
        obj_mesh.vertices=obj_mesh.vertices/1000
        obj_pc[obj_name] = obj_mesh.vertices
        obj_face[obj_name] = obj_mesh.faces
        obj_scale[obj_name] = get_diameter(obj_mesh.vertices)
        resample_obj_xyz(obj_mesh.vertices, obj_mesh.faces, root+obj_name)
        obj_pc_resampled[obj_name] = np.load((root+obj_name).replace('.ply', 'resampled.npy'),allow_pickle=True)
        obj_resampled_faceid[obj_name] = np.load((root+obj_name).replace('.ply', 'resample_face_id.npy'),allow_pickle=True)
    return obj_pc, obj_face, obj_scale, obj_pc_resampled, obj_resampled_faceid


def resample_obj_xyz(verts, faces, path):
    obj_mesh = trimesh.Trimesh(vertices=verts,
                               faces=faces)
    obj_xyz_resampled, face_id = trimesh.sample.sample_surface(obj_mesh, 3000)
    np.save(path.replace('.stl', 'resampled.npy'), obj_xyz_resampled)
    np.save(path.replace('.stl', 'resample_face_id.npy'), obj_xyz_resampled)

def resample_obj_xyz(verts, faces, path):
    obj_mesh = trimesh.Trimesh(vertices=verts,
                               faces=faces)
    obj_xyz_resampled, face_id = trimesh.sample.sample_surface(obj_mesh, 3000)
    np.save(path.replace('.ply', 'resampled.npy'), obj_xyz_resampled)
    np.save(path.replace('.ply', 'resample_face_id.npy'), obj_xyz_resampled)

def get_diameter(vp):
    x = vp[:, 0].reshape((1, -1))
    y = vp[:, 1].reshape((1, -1))
    z = vp[:, 2].reshape((1, -1))
    x_max, x_min, y_max, y_min, z_max, z_min = np.max(x), np.min(x), np.max(y), np.min(y), np.max(z), np.min(z)
    diameter_x = abs(x_max - x_min)
    diameter_y = abs(y_max - y_min)
    diameter_z = abs(z_max - z_min)
    diameter = np.sqrt(diameter_x**2 + diameter_y**2 + diameter_z**2)
    return diameter

def readTxt(file_path):
    img_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip()
            img_list.append(item)
    file_to_read.close()
    return img_list

def pose_from_RT_HO3D(R, T):
    pose = np.zeros((4,4))
    pose[:3,3] = T
    pose[3,3] = 1
    R33, _ = cv2.Rodrigues(R)
    pose[:3, :3] = R33
    return pose

#_, _ = load_objects_HO3D('../models/HO3D_Object_models')
if __name__ == '__main__':
    load_objects_FPHA()