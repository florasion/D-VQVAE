from torch.utils.data import Dataset
import torch
import os
import pickle
from torchvision import transforms
import numpy as np
from utils import utils
import time
import mano
from pytorch3d.structures import Meshes
from utils import utils_loss
from PIL import Image
import json
import trimesh
import torch.nn.functional as F
from scipy.linalg import orthogonal_procrustes
import scipy
import open3d as o3d
from scipy.spatial import KDTree
def find_point_distances(meshA, meshB):

    verticesA = torch.tensor(meshA.vertices, dtype=torch.float32)
    normalsA = torch.tensor(meshA.vertex_normals, dtype=torch.float32)*-1
    verticesB = torch.tensor(meshB.vertices, dtype=torch.float32)

    ray_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(meshB)
    

    distances = torch.full((verticesA.shape[0],), float(0), dtype=torch.float32)

    for i, (point, normal) in enumerate(zip(verticesA, normalsA)):

        ray_origins = point.unsqueeze(0).numpy()
        ray_directions = normal.unsqueeze(0).numpy()
        

        locations, index_ray, index_tri = ray_intersector.intersects_location(ray_origins, ray_directions)
        

        if len(locations) > 0:

            intersection_point = torch.tensor(locations[0], dtype=torch.float32)
            distance = torch.norm(intersection_point - point)

            distances[i] = distance
    
    return distances


def batched_index_select(input, index, dim=1):
    '''
    :param input: [B, N1, *]
    :param dim: the dim to be selected
    :param index: [B, N2]
    :return: [B, N2, *] selected result
    '''
    views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim=dim, index=index)

def get_faces_xyz(faces_idx, xyz):
    '''
    :param faces_idx: [B, N1, 3]. N1 is number of faces (1538 for MANO), index of face vertices in N2
    :param xyz: [B, N2, 3]. N2 is number of points.
    :return: faces_xyz: [B, N1, 3, 3] faces vertices coordinate
    '''
    B, N1, D = faces_idx.size()
    N2 = xyz.size(1)
    xyz_replicated = xyz.cpu().unsqueeze(1).repeat(1,N1,1,1)  # use cpu to save CUDA memory
    faces_idx_replicated = faces_idx.unsqueeze(-1).repeat(1,1,1,D).type(torch.LongTensor)
    return torch.gather(xyz_replicated, dim=2, index=faces_idx_replicated).to(faces_idx.device)

def batch_mesh_contains_points(
    ray_origins, # point cloud as origin of rays
    obj_triangles,
    direction=torch.Tensor([0.4395064455, 0.617598629942, 0.652231566745]),
):
    """Times efficient but memory greedy !
    Computes ALL ray/triangle intersections and then counts them to determine
    if point inside mesh
    Args:
    ray_origins: (batch_size x point_nb x 3)
    obj_triangles: (batch_size, triangle_nb, vertex_nb=3, vertex_coords=3)
    tol_thresh: To determine if ray and triangle are //
    Returns:
    exterior: (batch_size, point_nb) 1 if the point is outside mesh, 0 else
    """
    tol_thresh = 0.0000001
    batch_size = obj_triangles.shape[0]
    triangle_nb = obj_triangles.shape[1]
    point_nb = ray_origins.shape[1]

    # Batch dim and triangle dim will flattened together
    batch_points_size = batch_size * triangle_nb
    # Direction is random but shared
    v0, v1, v2 = obj_triangles[:, :, 0], obj_triangles[:, :, 1], obj_triangles[:, :, 2]
    # Get edges
    v0v1 = v1 - v0
    v0v2 = v2 - v0

    direction = direction.to(ray_origins.device)
    # Expand needed vectors
    batch_direction = direction.view(1, 1, 3).expand(batch_size, triangle_nb, 3)

    # Compute ray/triangle intersections
    pvec = torch.cross(batch_direction, v0v2, dim=2)
    dets = torch.bmm(
        v0v1.view(batch_points_size, 1, 3), pvec.view(batch_points_size, 3, 1)
    ).view(batch_size, triangle_nb)

    # Check if ray and triangle are parallel
    parallel = abs(dets) < tol_thresh
    invdet = 1 / (dets + 0.1 * tol_thresh)

    # Repeat mesh info as many times as there are rays
    triangle_nb = v0.shape[1]
    v0 = v0.repeat(1, point_nb, 1)
    v0v1 = v0v1.repeat(1, point_nb, 1)
    v0v2 = v0v2.repeat(1, point_nb, 1)
    hand_verts_repeated = (
        ray_origins.view(batch_size, point_nb, 1, 3)
        .repeat(1, 1, triangle_nb, 1)
        .view(ray_origins.shape[0], triangle_nb * point_nb, 3)
    )
    pvec = pvec.repeat(1, point_nb, 1)
    invdet = invdet.repeat(1, point_nb)
    tvec = hand_verts_repeated - v0
    u_val = (
        torch.bmm(
            tvec.view(batch_size * tvec.shape[1], 1, 3),
            pvec.view(batch_size * tvec.shape[1], 3, 1),
        ).view(batch_size, tvec.shape[1])
        * invdet
    )
    # Check ray intersects inside triangle
    u_correct = (u_val > 0) * (u_val < 1)
    qvec = torch.cross(tvec, v0v1, dim=2)

    batch_direction = batch_direction.repeat(1, point_nb, 1)
    v_val = (
        torch.bmm(
            batch_direction.view(batch_size * qvec.shape[1], 1, 3),
            qvec.view(batch_size * qvec.shape[1], 3, 1),
        ).view(batch_size, qvec.shape[1])
        * invdet
    )
    v_correct = (v_val > 0) * (u_val + v_val < 1)
    t = (
        torch.bmm(
            v0v2.view(batch_size * qvec.shape[1], 1, 3),
            qvec.view(batch_size * qvec.shape[1], 3, 1),
        ).view(batch_size, qvec.shape[1])
        * invdet
    )
    # Check triangle is in front of ray_origin along ray direction
    t_pos = t >= tol_thresh
    parallel = parallel.repeat(1, point_nb)
    # # Check that all intersection conditions are met
    try:
        not_parallel = 1 - parallel
    except:
        not_parallel = parallel==False
    final_inter = v_correct * u_correct * not_parallel * t_pos
    # Reshape batch point/vertices intersection matrix
    # final_intersections[batch_idx, point_idx, triangle_idx] == 1 means ray
    # intersects triangle
    final_intersections = final_inter.view(batch_size, point_nb, triangle_nb)
    # Check if intersection number accross mesh is odd to determine if point is
    # outside of mesh
    exterior = final_intersections.sum(2) % 2 == 0
    return exterior
def get_NN(src_xyz, trg_xyz, k=1):
    '''
    :param src_xyz: [B, N1, 3]
    :param trg_xyz: [B, N2, 3]
    :return: nn_dists, nn_dix: all [B, 3000] tensor for NN distance and index in N2
    '''
    B = src_xyz.size(0)
    src_lengths = torch.full(
        (src_xyz.shape[0],), src_xyz.shape[1], dtype=torch.int64, device=src_xyz.device
    )  # [B], N for each num
    trg_lengths = torch.full(
        (trg_xyz.shape[0],), trg_xyz.shape[1], dtype=torch.int64, device=trg_xyz.device
    )
    src_nn = knn_points(src_xyz, trg_xyz, lengths1=src_lengths, lengths2=trg_lengths, K=k)  # [dists, idx]
    nn_dists = src_nn.dists[..., 0]
    nn_idx = src_nn.idx[..., 0]
 
    return nn_dists, nn_idx


def random_rotation_matrix():
    return scipy.spatial.transform.Rotation.random().as_matrix()

# 应用随机旋转到点云
def apply_random_rotation(point_cloud,rotation_matrix):
    rotated_point_cloud = point_cloud @ rotation_matrix.T
    return rotated_point_cloud


def align_w_scale(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) 
    mtx2_t = mtx2_t + t1
    if return_trafo:
        #return R, s, s1, t1 - t2
        return R,  t1 
    else:
        return mtx2_t
class HMDO(Dataset):
    def __init__(self, 
                 obj_root='/',
                 mode="train", vis=False, batch_size=160):

        self.mode = mode
        self.batch_size = 3
        self.obj_pc_path = 'obj_path/object.npy'

        self.hand_v_path = 'hand_path/hand.npy'

        self.transformer = False
        self.gentransformer=True
        self.base_path = 'root_path'
        self.hand_folder = 'hand_mesh'
        self.object_folder = 'object_mesh'
        self.hand_annotation ='hand_annotation'
        self.all_data = []
        self.all_contact_map = []
        self.all_contact_map_bool = []
        self.all_interior_d = []
        self.all_object_vertices = []
        self.all_object_vertices_before = []
        self.all_object_vertices_org = []
        self.all_mano_p = []
        self.all_mano_p_before = [] 
        self.all_movement_gt = []
        self.all_face = []
        self.all_timestep = []
        self.all_interior_f = []
        self.all_normal= []
        self.all_distance = []
        self.all_mask_num = []
        self.all_id = []

        self.all_contact_map_bool_l1= []
        self.all_object_vertices_l1= []
        self.all_object_vertices_org_l1= []
        self.all_distance_l1= []
        self.all_normal_l1= []
        self.all_mask_num_l1= []
        self.all_index_l1 = []
        self.all_smp_distances_l1 = []
        

        self.all_contact_map_bool_l2 = []
        self.all_object_vertices_l2 = []
        self.all_object_vertices_org_l2 = []
        self.all_distance_l2 = []
        self.all_normal_l2 = []
        self.all_mask_num_l2 = []
        self.all_index_l2 = []
        self.all_smp_distances_l2 = []



        self.__load_dataset__()
        #self.save()
        self.dataset_size = int(len(self.all_contact_map_bool))

        self.transform = transforms.ToTensor()
        self.sample_nPoint = 3000
        #self.batch_size = batch_size
    
    def save(self):
        output_dir = '/root/autodl-tmp/processed'

        file_path = os.path.join(output_dir, 'all_contact_map_bool.npy')
        np.save(file_path, np.array(self.all_contact_map_bool,dtype=object))
        file_path = os.path.join(output_dir, 'all_object_vertices.npy')
        np.save(file_path, np.array(self.all_object_vertices,dtype=object))
        file_path = os.path.join(output_dir, 'all_object_vertices_org.npy')
        np.save(file_path, np.array(self.all_object_vertices_org,dtype=object))
        file_path = os.path.join(output_dir, 'all_distance.npy')
        np.save(file_path, np.array(self.all_distance,dtype=object))
        file_path = os.path.join(output_dir, 'all_normal.npy')
        np.save(file_path, np.array(self.all_normal,dtype=object))
        file_path = os.path.join(output_dir, 'all_movement_gt.npy')
        np.save(file_path, np.array(self.all_movement_gt,dtype=object))
        file_path = os.path.join(output_dir, 'all_mask_num.npy')
        np.save(file_path, np.array(self.all_mask_num,dtype=object))
        file_path = os.path.join(output_dir, 'all_id.npy')
        np.save(file_path, np.array(self.all_id,dtype=object))
        
        file_path = os.path.join(output_dir, 'all_contact_map_bool_l1.npy')
        np.save(file_path, np.array(self.all_contact_map_bool_l1,dtype=object))
        file_path = os.path.join(output_dir, 'all_object_vertices_org_l1.npy')
        np.save(file_path, np.array(self.all_object_vertices_org_l1,dtype=object))
        file_path = os.path.join(output_dir, 'all_distance_l1.npy')
        np.save(file_path, np.array(self.all_distance_l1,dtype=object))
        file_path = os.path.join(output_dir, 'all_normal_l1.npy')
        np.save(file_path, np.array(self.all_normal_l1,dtype=object))
        file_path = os.path.join(output_dir, 'all_mask_num_l1.npy')
        np.save(file_path, np.array(self.all_mask_num_l1,dtype=object))
        file_path = os.path.join(output_dir, 'all_index_l1.npy')
        np.save(file_path, np.array(self.all_index_l1,dtype=object))
        file_path = os.path.join(output_dir, 'all_smp_distances_l1.npy')
        np.save(file_path, np.array(self.all_smp_distances_l1,dtype=object))
        
        file_path = os.path.join(output_dir, 'all_contact_map_bool_l2.npy')
        np.save(file_path, np.array(self.all_contact_map_bool_l2,dtype=object))
        file_path = os.path.join(output_dir, 'all_object_vertices_org_l2.npy')
        np.save(file_path, np.array(self.all_object_vertices_org_l2,dtype=object))
        file_path = os.path.join(output_dir, 'all_distance_l2.npy')
        np.save(file_path, np.array(self.all_distance_l2,dtype=object))
        file_path = os.path.join(output_dir, 'all_normal_l2.npy')
        np.save(file_path, np.array(self.all_normal_l2,dtype=object))
        file_path = os.path.join(output_dir, 'all_mask_num_l2.npy')
        np.save(file_path, np.array(self.all_mask_num_l2,dtype=object))
        file_path = os.path.join(output_dir, 'all_index_l2.npy')
        np.save(file_path, np.array(self.all_index_l2,dtype=object))
        file_path = os.path.join(output_dir, 'all_smp_distances_l2.npy')
        np.save(file_path, np.array(self.all_smp_distances_l2,dtype=object))
        print('save finish')

    def load(self):
        self.all_contact_map_bool = np.load('/root/autodl-tmp/processed/all_contact_map_bool.npy',allow_pickle=True)
        self.all_object_vertices = np.load('/root/autodl-tmp/processed/all_object_vertices.npy',allow_pickle=True)
        self.all_object_vertices_org = np.load('/root/autodl-tmp/processed/all_object_vertices_org.npy',allow_pickle=True)
        self.all_distance = np.load('/root/autodl-tmp/processed/all_distance.npy',allow_pickle=True)
        self.all_normal = np.load('/root/autodl-tmp/processed/all_normal.npy',allow_pickle=True)
        self.all_movement_gt = np.load('/root/autodl-tmp/processed/all_movement_gt.npy',allow_pickle=True)
        self.all_mask_num = np.load('/root/autodl-tmp/processed/all_mask_num.npy',allow_pickle=True)
        self.all_hand_xyz = np.load('/root/autodl-tmp/processed/all_hand_xyz.npy',allow_pickle=True)
        self.all_id = np.load('/root/autodl-tmp/processed/all_id.npy',allow_pickle=True)

        self.all_contact_map_bool_l1 = np.load('/root/autodl-tmp/processed/all_contact_map_bool_l1.npy',allow_pickle=True)
        self.all_object_vertices_org_l1 = np.load('/root/autodl-tmp/processed/all_object_vertices_org_l1.npy',allow_pickle=True)
        self.all_distance_l1 = np.load('/root/autodl-tmp/processed/all_distance_l1.npy',allow_pickle=True)
        self.all_normal_l1 = np.load('/root/autodl-tmp/processed/all_normal_l1.npy',allow_pickle=True)
        self.all_mask_num_l1 = np.load('/root/autodl-tmp/processed/all_mask_num_l1.npy',allow_pickle=True)
        self.all_index_l1 = np.load('/root/autodl-tmp/processed/all_index_l1.npy',allow_pickle=True)
        self.all_smp_distances_l1 = np.load('/root/autodl-tmp/processed/all_smp_distances_l1.npy',allow_pickle=True)

        self.all_contact_map_bool_l2 = np.load('/root/autodl-tmp/processed/all_contact_map_bool_l2.npy',allow_pickle=True)
        self.all_object_vertices_org_l2 = np.load('/root/autodl-tmp/processed/all_object_vertices_org_l2.npy',allow_pickle=True)
        self.all_distance_l2 = np.load('/root/autodl-tmp/processed/all_distance_l2.npy',allow_pickle=True)
        self.all_normal_l2 = np.load('/root/autodl-tmp/processed/all_normal_l2.npy',allow_pickle=True)
        self.all_mask_num_l2 = np.load('/root/autodl-tmp/processed/all_mask_num_l2.npy',allow_pickle=True)
        self.all_index_l2 = np.load('/root/autodl-tmp/processed/all_index_l2.npy',allow_pickle=True)
        self.all_smp_distances_l2 = np.load('/root/autodl-tmp/processed/all_smp_distances_l2.npy',allow_pickle=True)
        print('load finish')
        

    def __load_dataset__(self):
        print('loading dataset start')
        self.load()
        #if self.transformer:
        #    self.all_z_q = np.load(self.z_q_path)
        print('loading dataset finish')

    # def process_sequences(self):

        
    
    def __len__(self):
        return self.dataset_size# - (self.dataset_size % self.batch_size)  # in case of unmatched mano batch size
    

    def pad_point_cloud_with_zeros(self,x, target_num=12):

        n, f = x.shape

        if n == target_num:
            return x, index  
        if n < target_num:

            num_to_pad = target_num - n
            

            zero_padding = torch.zeros(num_to_pad, f, device=x.device)


            x_padded = torch.cat([x, zero_padding], dim=0)

        return x_padded
    def pad_point_cloud_with_zeros_1d(self,x, target_num=12):
        
        n= x.shape
        f =1
        if n == target_num:
            return x, index  
        
        if n < target_num:

            num_to_pad = target_num - n

            zero_padding = torch.zeros(num_to_pad, f, device=x.device)

            x_padded = torch.cat([x, zero_padding], dim=0)

        return x_padded


    def pad_point_cloud_with_zeros_index(self,x, index, target_num=12):
        n, f = x.shape

        if n == target_num:
            return x, index  
        
        if n < target_num:

            num_to_pad = target_num - n
            

            zero_padding = torch.zeros(num_to_pad, f, device=x.device)


            x_padded = torch.cat([x, zero_padding], dim=0)

            index_padding = torch.full((1,num_to_pad), -1, dtype=index.dtype, device=index.device)
            index_padded = torch.cat([index.unsqueeze(0), index_padding], dim=1)


        return x_padded, index_padded.squeeze(0)

    def __getitem__(self, idx):

        contact_map_bool_batch = self.all_contact_map_bool[idx]
        distance_batch = self.all_distance[idx]
        object_vertices_batch = self.all_object_vertices[idx]
        object_vertices_org_batch = self.all_object_vertices_org[idx]#.unsqueeze(0)
        normal_batch = self.all_normal[idx]
        movement_gt_batch = self.all_movement_gt[idx]#object_vertices_batch-object_vertices_org_batch
        mask_num_batch = self.all_mask_num[idx]
        hand_xyz = self.all_hand_xyz[idx]
        face_id = self.all_id[idx]


        contact_map_bool_batch_l1 = self.all_contact_map_bool_l1[idx]
        distance_batch_l1 = self.all_distance_l1[idx]
        object_vertices_org_batch_l1 = self.all_object_vertices_org_l1[idx]
        normal_batch_l1 = self.all_normal_l1[idx]
        mask_num_batch_l1 = self.all_mask_num_l1[idx]
        index_l1 = self.all_index_l1[idx]
        smp_distances_l1 = self.all_smp_distances_l1[idx]

        contact_map_bool_batch_l2 = self.all_contact_map_bool_l2[idx]
        distance_batch_l2 = self.all_distance_l2[idx]
        object_vertices_org_batch_l2 = self.all_object_vertices_org_l2[idx]
        normal_batch_l2 = self.all_normal_l2[idx]
        mask_num_batch_l2 = self.all_mask_num_l2[idx]
        index_l2 = self.all_index_l2[idx]
        smp_distances_l2 = self.all_smp_distances_l2[idx]
        hand_xyz = self.all_hand_xyz[idx]

        data = {
            'contact_map_bool': contact_map_bool_batch.squeeze(0),
            'object_vertices': object_vertices_batch.squeeze(0).float(),
            'object_vertices_org': object_vertices_org_batch.squeeze(0).float(),
            #'offset': offset_batch.squeeze(0).squeeze(0),
            'movement_gt':movement_gt_batch.squeeze(0).float(),
            'distance':distance_batch.squeeze(0).float().detach(),
            'normal':normal_batch.squeeze(0).float(),
            'mask_num':mask_num_batch,
            'face_id':face_id,

            
            'contact_map_bool_l1': contact_map_bool_batch_l1.squeeze(0),
            'hand_xyz': hand_xyz.squeeze(0).float(),

            'object_vertices_org_l1': object_vertices_org_batch_l1.squeeze(0).float(),
            'distance_l1':distance_batch_l1.squeeze(0).float().detach(),
            'normal_l1':normal_batch_l1.squeeze(0).float(),
            'mask_num_l1':mask_num_batch_l1,
            'index_l1':index_l1.squeeze(0),
            'smp_distances_l1':smp_distances_l1.squeeze(0).float(),
            'contact_map_bool_l2': contact_map_bool_batch_l2.squeeze(0),

            'object_vertices_org_l2': object_vertices_org_batch_l2.squeeze(0).float(),
            'distance_l2':distance_batch_l2.squeeze(0).float().detach(),
            'normal_l2':normal_batch_l2.squeeze(0).float(),
            'mask_num_l2':mask_num_batch_l2,
            'index_l2':index_l2.squeeze(0),
            'smp_distances_l2':smp_distances_l2.squeeze(0).float(),

        }

        return data

