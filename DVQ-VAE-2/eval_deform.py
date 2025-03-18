import numpy as np

import time
import pickle
import os
import time
import torch
import argparse
from torch.utils.data import DataLoader

import numpy as np
import random
from utils import  utils_loss
from utils import utils as utilss
import mano
import json
from utils.loss import TTT_loss
import trimesh
from metric.simulate import run_simulation
from pytorch3d.structures import Meshes
import os
from pytorch3d.structures import Meshes
import open3d as o3d
import numpy as np
from PIL import Image
import os
from torch import optim, nn, utils, Tensor
import lightning as L
from pytorch3d.structures import Meshes
import mano
from torch.utils.data import DataLoader
import torch
import trimesh
import numpy as np
#from metric.simulate import run_simulation

from pytorch3d.loss import mesh_normal_consistency
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.loss import chamfer_distance
import os



from utils import utils_loss
import math


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
def seal(mesh_to_seal):
    '''
    Seal MANO hand wrist to make it wathertight.
    An average of wrist vertices is added along with its faces to other wrist vertices.
    '''
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
    center = (mesh_to_seal.vertices[circle_v_id, :]).mean(0)

    mesh_to_seal.vertices = np.vstack([mesh_to_seal.vertices, center])
    center_v_id = mesh_to_seal.vertices.shape[0] - 1

    # pylint: disable=unsubscriptable-object # pylint/issues/3139
    for i in range(circle_v_id.shape[0]):
        new_faces = [circle_v_id[i-1], circle_v_id[i], center_v_id] 
        mesh_to_seal.faces = np.vstack([mesh_to_seal.faces, new_faces])
    return mesh_to_seal

def intersect_vox_soft(obj_mesh, hand_mesh):
    '''
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    '''
    hand_vertices = hand_mesh.vertices

    mesh_obj = Meshes(verts= torch.tensor( obj_mesh.vertices).unsqueeze(0), faces=torch.tensor(obj_mesh.faces).unsqueeze(0))  
    obj_normal = mesh_obj.verts_normals_packed().view(1,-1, 3)

    hand_nn_dist_recon, hand_nn_idx_recon = utils_loss.get_NN( torch.tensor(hand_vertices).unsqueeze(0).float(),torch.tensor(obj_mesh.vertices).unsqueeze(0).float())
    contact_map_bool_hand = (hand_nn_dist_recon<1e-4).float()
    interior_hand = utils_loss.get_interior(obj_normal, torch.tensor(obj_mesh.vertices).unsqueeze(0).float(), torch.tensor(hand_vertices).unsqueeze(0).float(), hand_nn_idx_recon).type(torch.bool)
    contact_map_bool_hand[interior_hand] =  (contact_map_bool_hand[interior_hand]*-1).float()
    interior_f = torch.zeros(torch.tensor(obj_mesh.vertices).size(0)) 

    interior_f[hand_nn_idx_recon.squeeze(0)] = hand_nn_dist_recon.squeeze(1)

    mesh_ = Meshes(verts= torch.tensor(hand_vertices).unsqueeze(0), faces=torch.tensor(hand_mesh.faces).unsqueeze(0))  
    hand_normal = mesh_.verts_normals_packed().view(-1, 779, 3)
    hand_normal_prior = hand_normal[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)].unsqueeze(0)
    hand_vertices_prior = torch.tensor(hand_vertices[(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)]).float().unsqueeze(0)
    obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(torch.tensor(obj_mesh.vertices).unsqueeze(0).float(), hand_vertices_prior)
    if hand_normal_prior.size(1)>0:
        interior = utils_loss.get_interior(hand_normal_prior, hand_vertices_prior, torch.tensor(obj_mesh.vertices).unsqueeze(0).float(), obj_nn_idx_recon).type(torch.bool)
        NN_src_xyz = batched_index_select(hand_vertices_prior, obj_nn_idx_recon)  # [B, 3000, 3]
        NN_vector = NN_src_xyz - torch.tensor(obj_mesh.vertices).unsqueeze(0).float()  # [B, 3000, 3]
        # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
        NN_src_normal = batched_index_select(hand_normal_prior, obj_nn_idx_recon)
        #interior = interior。float
        interior_d = (NN_vector * NN_src_normal).sum(dim=-1)
        #interior = interior。float


        
        obj_nn_dist_recon[interior]= obj_nn_dist_recon[interior]*-1
        contact_map_bool = ((obj_nn_dist_recon<1e-4)&(obj_nn_dist_recon > -2e-4)).float()
        contact_map = obj_nn_dist_recon
        contact_map_bool[interior] = (contact_map_bool[interior]*-1).float()
        contact_map = contact_map *abs(contact_map_bool)
        interior_d = interior_d * contact_map_bool

        distance = find_point_distances(obj_mesh,hand_mesh)
        distance = distance * (contact_map_bool==-1).float().squeeze(0)
        return distance.sum()
    else:
        return 0

def find_point_distances(meshA, meshB):
    # 将mesh转换为点云和法线
    verticesA = torch.tensor(meshA.vertices, dtype=torch.float32)
    normalsA = torch.tensor(meshA.vertex_normals, dtype=torch.float32)*-1
    verticesB = torch.tensor(meshB.vertices, dtype=torch.float32)
    
    # 创建一个空的三角形光线拦截器
    ray_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(meshB)
    
    # 初始化距离张量
    distances = torch.full((verticesA.shape[0],), float(0), dtype=torch.float32)
    
    # 遍历meshA中的每个点
    for i, (point, normal) in enumerate(zip(verticesA, normalsA)):
        # 创建从该点出发的射线，方向为法线方向
        ray_origins = point.unsqueeze(0).numpy()
        ray_directions = normal.unsqueeze(0).numpy()
        
        # 计算射线与meshB的交点
        locations, index_ray, index_tri = ray_intersector.intersects_location(ray_origins, ray_directions)
        
        # 如果有交点，计算距离
        if len(locations) > 0:
            # 取第一个交点（假设只有一个交点）
            intersection_point = torch.tensor(locations[-1], dtype=torch.float32)
            distance = torch.norm(intersection_point - point)
            
            # 更新距离
            distances[i] = distance
    
    return distances

def intersect_vox(obj_mesh, hand_mesh, pitch=0.5):
    '''
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    '''
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume

with torch.no_grad():
        rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                                model_type='mano',
                                use_pca=False,
                                num_pca_comps=51,
                                batch_size=1,
                                flat_hand_mean=True)
rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).contiguous() # [1, 1538, 3], face triangle indexes  

total_penetr_vol_TTA = 0
total_simu_disp_TTA  = 0
total_num = 0
total_contact_p = 0
total_contact_r = 0
total_normal = 0
total_laplacian = 0
total_champer = 0
error= 0 
for i in range(11):
    for j in range(10):
        try:
            hand_mesh_TTA= trimesh.load_mesh("./deform/obj_{}/idx_{}/hand.ply".format(i,j))
            obj_mesh_TTA = trimesh.load_mesh("./deform/obj_{}/idx_{}/obj.ply".format(i,j))
            org_mesh = trimesh.load_mesh("org_path/obj_{}/idx_{}/obj_org.ply".format(i,j))
            #org_mesh_TTA = trimesh.load_mesh("/home/zhaozhe/Pycode/tfVQVAEnonspace/vis_deform_300/obj_{}_{}.ply".format(i,j))
            obj_mesh_TTA = trimesh.Trimesh(vertices = obj_mesh_TTA.vertices,faces = obj_mesh_TTA.faces)
            # object=org_mesh
            # trimesh.repair.fix_normals(object)
            # object = trimesh.convex.convex_hull(object)
            # vol, mesh_dist = intersection_eval(hand, object, res=0.001, visualize_flag=True)
            # vol_list.append(vol)
            # mesh_dist_list.append(mesh_dist)
            object_org_scaled = torch.tensor( obj_mesh_TTA.vertices)
            mesh_obj = Meshes(verts= torch.tensor( obj_mesh_TTA.vertices).unsqueeze(0), faces=torch.tensor(obj_mesh_TTA.faces).unsqueeze(0))  
            obj_normal = mesh_obj.verts_normals_packed().view(1,-1, 3)
            hand_nn_dist_recon, hand_nn_idx_recon = utils_loss.get_NN( torch.tensor(hand_mesh_TTA.vertices).unsqueeze(0).float(),torch.tensor( obj_mesh_TTA.vertices).unsqueeze(0).float())
            contact_map_bool_hand = (hand_nn_dist_recon<3e-4).float()
            interior_hand = utils_loss.get_interior(obj_normal, object_org_scaled.unsqueeze(0),torch.tensor(hand_mesh_TTA.vertices).unsqueeze(0), hand_nn_idx_recon).type(torch.bool)
            contact_map_bool_hand[interior_hand] =  (contact_map_bool_hand[interior_hand]*-1).float()
            interior_f = torch.zeros(torch.tensor( obj_mesh_TTA.vertices).size(0)) 
            interior_f[hand_nn_idx_recon.squeeze(0)] = hand_nn_dist_recon.squeeze(1).squeeze(0) 
            hand_vertices_prior = torch.tensor(torch.tensor(hand_mesh_TTA.vertices)[(abs(contact_map_bool_hand)==1).cpu().detach().bool().squeeze(0)]).float().unsqueeze(0)
            # d =  torch.norm(torch.tensor(movement_gt), dim=1, keepdim=True)
            # movement_gt = movement_gt*(movement_gt>3e-3)
            obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(object_org_scaled.unsqueeze(0).float(), hand_vertices_prior)
            mesh = Meshes(verts=torch.tensor( hand_mesh_TTA.vertices).unsqueeze(0), faces=rh_faces)
            hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)
            hand_normal_prior = hand_normal[0][(abs(contact_map_bool_hand)==1).cpu().detach().bool().squeeze(0)].unsqueeze(0)
            # if not nn_dist:
            #     nn_dist, nn_idx = utils_loss.get_NN(obj_xyz, hand_xyz)
            # print(hand_vertices_prior.size())
            interior = utils_loss.get_interior(hand_normal_prior, hand_vertices_prior, object_org_scaled, obj_nn_idx_recon).type(torch.bool)  # True for interior

            NN_src_xyz = batched_index_select(hand_vertices_prior, obj_nn_idx_recon)  # [B, 3000, 3]
            NN_vector = NN_src_xyz -  torch.tensor(obj_mesh_TTA.vertices)  # [B, 3000, 3]
            # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
            NN_src_normal = batched_index_select(hand_normal_prior, obj_nn_idx_recon)
            #interior = interior。float
            interior_d = (NN_vector * NN_src_normal).sum(dim=-1)
            #interior_dist=nn_dist[interior]
            obj_nn_dist_recon[interior]= obj_nn_dist_recon[interior]*-1
            contact_map_bool = ((obj_nn_dist_recon<1e-4)&(obj_nn_dist_recon > -2e-4)).float()


            new_src_mesh = Meshes(torch.tensor(obj_mesh_TTA.vertices).unsqueeze(0).float(),torch.tensor(obj_mesh_TTA.faces).unsqueeze(0))
        
            loss_chamfer, _= chamfer_distance(torch.tensor(obj_mesh_TTA.vertices).unsqueeze(0).float()[contact_map_bool!=1].unsqueeze(0), torch.tensor(org_mesh.vertices).unsqueeze(0).float()[contact_map_bool!=1].unsqueeze(0), point_reduction='sum', batch_reduction='mean') 
            loss_normal = mesh_normal_consistency(new_src_mesh)
            loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")



            contact_p = contact_map_bool.sum()/torch.tensor(obj_mesh_TTA.vertices).size(0)

            hand = seal(hand_mesh_TTA)
            # penetr_vol=vol
            penetr_vol = intersect_vox_soft(obj_mesh_TTA, hand)
            if  penetr_vol < 1e-8:
                    sample_contact = False
            else:
                    sample_contact = True
                    total_contact_r += 1
            # simulation displacement
            vhacd_exe = "/home/zhaozhe/Pycode/v-hacd-master/TestVHACD"
            try:
                simu_disp = run_simulation(hand_mesh_TTA.vertices, rh_faces.reshape((-1, 3)),
                                        obj_mesh_TTA.vertices, obj_mesh_TTA.faces.reshape((-1, 3)),
                                        vhacd_exe=vhacd_exe)
                    #print('run success')
            except:
                    simu_disp = 0.10
            total_penetr_vol_TTA+=penetr_vol
            total_simu_disp_TTA+=simu_disp
            total_num+=1
            total_contact_p+=contact_p
            total_champer += loss_chamfer
            total_normal += loss_normal
            total_laplacian +=loss_laplacian
        except:
            error+=1
        print("#############################################")
        print(" TTA vol cm3: ", total_penetr_vol_TTA/total_num)
        print(" TTA simu_disp: ", total_simu_disp_TTA/total_num)
        print(" TTA contact_p: ", total_contact_p/total_num)
        print(" TTA chamfer: ", total_champer/total_num)
        print(" TTA simu_disp: ", total_simu_disp_TTA/total_num)
        print(" TTA normal: ", loss_normal/total_num)
        print(" TTA laplacian: ", loss_laplacian/total_num)
        print(" TTA simu_disp: ", total_simu_disp_TTA/total_num)
        print(" TTA contact_r: ", total_contact_r/total_num)
        print(" error: ", error)