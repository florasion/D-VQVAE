import numpy as np
import igl as igl
import time
import pickle
import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
from scipy.spatial import KDTree
import numpy as np
import random
from utils import  utils_loss
from utils import utils as utilss
import mano
import json
from utils.loss import TTT_loss
import trimesh
from metric.simulate import run_simulation

import os
from pytorch3d.structures import Meshes
import open3d as o3d
import numpy as np
from PIL import Image



def pad_point_cloud_with_zeros(x, target_num=12):

    n, f = x.shape

    if n == target_num:
        return x, index  
    if n < target_num:

        num_to_pad = target_num - n
        

        zero_padding = torch.zeros(num_to_pad, f, device=x.device)


        x_padded = torch.cat([x, zero_padding], dim=0)

    return x_padded
def pad_point_cloud_with_zeros_1d(x, target_num=12):
    
    n= x.shape
    f =1
    if n == target_num:
        return x, index  
    
    if n < target_num:

        num_to_pad = target_num - n
        

        zero_padding = torch.zeros(num_to_pad, f, device=x.device)


        x_padded = torch.cat([x, zero_padding], dim=0)

    return x_padded


def pad_point_cloud_with_zeros_index(x, index, target_num=12):
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


import os
from torch import optim, nn, utils, Tensor
import lightning as L
from pytorch3d.structures import Meshes
import mano
from network.softNet import softNet
from torch.utils.data import DataLoader
import torch
import trimesh
import numpy as np
#from metric.simulate import run_simulation

import igl
import os



from utils import utils_loss
import math
softnet = softNet()
checkpoint = torch.load("your_ckpt", map_location=torch.device('cpu'))['state_dict']
softnet.load_state_dict(checkpoint)
softnet.eval()
softnet = softnet.to('cuda')

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
            
            # 更新距离
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



for batch_idx in range(0,12):
    for idx in range (0,10):
        hand_mesh_gif = []
        obj_mesh_gif = []
        obj_mesh = trimesh.load_mesh("org_mesh".format(batch_idx,idx))
        for i in range (0,100):
            hand_mesh= trimesh.load_mesh("hand_mesh_path/obj_{}/idx_{}/hand_{}".format(batch_idx,idx,99-i)+".ply")
            point_num = obj_mesh.vertices.shape[0]

            try:
                with torch.no_grad():
                        rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                                                model_type='mano',
                                                use_pca=False,
                                                num_pca_comps=51,
                                                batch_size=1,
                                                flat_hand_mean=True)
                rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).contiguous() # [1, 1538, 3], face triangle indexes  

                hand_vertices = hand_mesh.vertices

                hand_vertices = torch.tensor(hand_vertices).unsqueeze(0).float()

                object_org_scaled = torch.tensor(obj_mesh.vertices).unsqueeze(0).float()
                object_vertices_full = obj_mesh
                object_vertices = torch.tensor(obj_mesh.vertices).unsqueeze(0).float()

                mesh_obj = Meshes(verts= torch.tensor( obj_mesh.vertices).unsqueeze(0), faces=torch.tensor(obj_mesh.faces).unsqueeze(0))  
                obj_normal = mesh_obj.verts_normals_packed().view(1,-1, 3)

                hand_nn_dist_recon, hand_nn_idx_recon = utils_loss.get_NN( hand_vertices,torch.tensor(obj_mesh.vertices).unsqueeze(0).float())
                contact_map_bool_hand = (hand_nn_dist_recon<3e-4).float()
                interior_hand = utils_loss.get_interior(obj_normal, torch.tensor(obj_mesh.vertices).unsqueeze(0).float(),hand_vertices, hand_nn_idx_recon).type(torch.bool)
                contact_map_bool_hand[interior_hand] =  (contact_map_bool_hand[interior_hand]*-1).float()
                interior_f = torch.zeros(torch.tensor(obj_mesh.vertices).size(0)) 

                interior_f[hand_nn_idx_recon.squeeze(0)] = hand_nn_dist_recon.squeeze(1)

                                    
                hand_vertices_prior = torch.tensor(hand_vertices[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)]).float().unsqueeze(0)
                # d =  torch.norm(torch.tensor(movement_gt), dim=1, keepdim=True)
                # movement_gt = movement_gt*(movement_gt>3e-3)
                if hand_vertices_prior.size(1)>0:

                    obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(object_org_scaled, hand_vertices_prior)
                    mesh = Meshes(verts=hand_vertices, faces=rh_faces)
                    hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)
                    hand_normal_prior = hand_normal[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)].unsqueeze(0)
                    # if not nn_dist:
                    #     nn_dist, nn_idx = utils_loss.get_NN(obj_xyz, hand_xyz)

                    interior = utils_loss.get_interior(hand_normal_prior, hand_vertices_prior, object_org_scaled, obj_nn_idx_recon).type(torch.bool)  # True for interior

                    NN_src_xyz = batched_index_select(hand_vertices_prior, obj_nn_idx_recon)  # [B, 3000, 3]
                    NN_vector = NN_src_xyz - object_org_scaled  # [B, 3000, 3]
                    # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
                    NN_src_normal = batched_index_select(hand_normal_prior, obj_nn_idx_recon)
                    #interior = interior。float
                    interior_d = (NN_vector * NN_src_normal).sum(dim=-1)
                    #interior_dist=nn_dist[interior]
                    obj_nn_dist_recon[interior]= obj_nn_dist_recon[interior]*-1



                    contact_map_bool = ((obj_nn_dist_recon<1e-4)&(obj_nn_dist_recon > -2e-4)).float()

                    interior_d = interior_d * contact_map_bool
                    contact_map_bool[interior] = contact_map_bool[interior]*-1
                    contact_map = obj_nn_dist_recon

                    obj_mesh_dis = trimesh.Trimesh(vertices = object_org_scaled[0].detach(),faces = object_vertices_full.faces)
                    hand_mesh_dis = trimesh.Trimesh(vertices = hand_vertices[0].detach(),faces = rh_faces[0].detach())
                    distance = find_point_distances(obj_mesh_dis,hand_mesh_dis)
                    distance = distance * (contact_map_bool==-1).float().squeeze(0)

                    mask_num = object_vertices.size(1)
                    contact_map_bool_base = pad_point_cloud_with_zeros(contact_map_bool.squeeze(0).unsqueeze(1), target_num=12000).squeeze(1)
                    distance_base = pad_point_cloud_with_zeros(distance.unsqueeze(1), 12000).squeeze(1)
                    object_vertices = pad_point_cloud_with_zeros(object_vertices.squeeze(0), 12000).unsqueeze(0)
                    object_vertices_org = pad_point_cloud_with_zeros(object_org_scaled.squeeze(0), 12000).unsqueeze(0)
                    normal = torch.tensor(obj_mesh_dis.vertex_normals).unsqueeze(0)
                    normal = pad_point_cloud_with_zeros(normal.squeeze(0), 12000).unsqueeze(0)
                    
                    movemant_gt_base = object_vertices - object_vertices_org


                    vertices = object_org_scaled[0].detach()
                    faces = object_vertices_full.faces
                    mesh_in = o3d.geometry.TriangleMesh()
                    mesh_in.vertices = o3d.utility.Vector3dVector(vertices)
                    mesh_in.triangles = o3d.utility.Vector3iVector(faces)
                    #mesh_in.compute_vertex_normals()
                    original_vertices = np.asarray(mesh_in.vertices)

                    voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 16
                    mesh_smp = mesh_in.simplify_vertex_clustering(
                        voxel_size=voxel_size,
                        contraction=o3d.geometry.SimplificationContraction.Average)
                    simplified_vertices = torch.tensor( np.asarray(mesh_smp.vertices)).unsqueeze(0).float()
                    obj_mesh_dis = trimesh.Trimesh(vertices = simplified_vertices[0].detach(),faces = np.asarray(mesh_smp.triangles))
                    simplified_vertices = torch.tensor( np.asarray(obj_mesh_dis.vertices)).unsqueeze(0).float()
                    tree = KDTree( np.asarray(obj_mesh_dis.vertices))
                    smp_distances, indices = tree.query(original_vertices)


                    mesh_obj = Meshes(verts= simplified_vertices, faces=torch.tensor(np.asarray(obj_mesh_dis.faces)).unsqueeze(0))  
                    obj_normal = mesh_obj.verts_normals_packed().view(1,-1, 3)
                    hand_nn_dist_recon, hand_nn_idx_recon = utils_loss.get_NN( hand_vertices,simplified_vertices)
                    contact_map_bool_hand = (hand_nn_dist_recon<3e-4).float()
                    interior_hand = utils_loss.get_interior(obj_normal, simplified_vertices,hand_vertices, hand_nn_idx_recon).type(torch.bool)
                    contact_map_bool_hand[interior_hand] =  (contact_map_bool_hand[interior_hand]*-1).float()
                    interior_f = torch.zeros(torch.tensor(simplified_vertices).size(1)) 
                    interior_f[hand_nn_idx_recon.squeeze(0)] = hand_nn_dist_recon.squeeze(1).squeeze(0) 


                    hand_vertices_prior = torch.tensor(hand_vertices[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)]).float().unsqueeze(0)


                    obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(simplified_vertices, hand_vertices_prior)
                    mesh = Meshes(verts=hand_vertices, faces=rh_faces)
                    hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)
                    hand_normal_prior = hand_normal[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)].unsqueeze(0)

                    if hand_normal_prior.size(1)>0:
                        interior = utils_loss.get_interior(hand_normal_prior, hand_vertices_prior, simplified_vertices, obj_nn_idx_recon).type(torch.bool)  # True for interior

                        NN_src_xyz = batched_index_select(hand_vertices_prior, obj_nn_idx_recon)  # [B, 3000, 3]
                        NN_vector = NN_src_xyz - simplified_vertices  # [B, 3000, 3]
                        # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
                        NN_src_normal = batched_index_select(hand_normal_prior, obj_nn_idx_recon)
                        #interior = interior。float
                        interior_d = (NN_vector * NN_src_normal).sum(dim=-1)
                        #interior_dist=nn_dist[interior]
                        obj_nn_dist_recon[interior]= obj_nn_dist_recon[interior]*-1



                        contact_map_bool = ((obj_nn_dist_recon<1e-4)&(obj_nn_dist_recon > -2e-4)).float()

                        interior_d = interior_d * contact_map_bool
                        contact_map_bool[interior] = contact_map_bool[interior]*-1
                        contact_map = obj_nn_dist_recon

                        
                        hand_mesh_dis = trimesh.Trimesh(vertices = hand_vertices[0].detach(),faces = rh_faces[0].detach())
                        distance = find_point_distances(obj_mesh_dis,hand_mesh_dis)
                        print('obj_mesh_dis:',obj_mesh_dis.vertices.shape)
                        print('contact_map_bool',contact_map_bool.size())
                        print('distance',distance.size())
                        print('simplified_vertices',simplified_vertices.size())

                        if distance.size(0)==contact_map_bool.size(1):
                            
                            distance = distance * (contact_map_bool==-1).float().squeeze(0)

                            mask_num_l1 = contact_map_bool.size(1)
                            contact_map_bool_l1 = pad_point_cloud_with_zeros(contact_map_bool.squeeze(0).unsqueeze(1), target_num=1200).squeeze(1)
                            distance_l1,indices_l1 = pad_point_cloud_with_zeros_index(distance.unsqueeze(1),torch.tensor(indices), target_num=1200)
                            distance_l1 = distance_l1.squeeze(1)

                            object_vertices_org_l1 = pad_point_cloud_with_zeros(simplified_vertices.squeeze(0), target_num=1200).unsqueeze(0)
                            normal_l1 = torch.tensor(obj_mesh_dis.vertex_normals)
                            normal_l1 = pad_point_cloud_with_zeros(normal_l1, target_num=1200).unsqueeze(0)
                            
   
                            smp_distances_l1,indices_l1 = pad_point_cloud_with_zeros_index(torch.tensor(smp_distances).unsqueeze(1),torch.tensor(indices), target_num=12000)



                           
                            mesh_in = o3d.geometry.TriangleMesh()
                            mesh_in.vertices = o3d.utility.Vector3dVector(obj_mesh_dis.vertices)
                            mesh_in.triangles = o3d.utility.Vector3iVector(obj_mesh_dis.faces)
                            #mesh_in.compute_vertex_normals()
                            original_vertices = np.asarray(mesh_in.vertices)

                            voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 8
                            mesh_smp = mesh_in.simplify_vertex_clustering(
                                voxel_size=voxel_size,
                                contraction=o3d.geometry.SimplificationContraction.Average)
                            simplified_vertices = torch.tensor( np.asarray(mesh_smp.vertices)).unsqueeze(0).float()
                            obj_mesh_dis = trimesh.Trimesh(vertices = simplified_vertices[0].detach(),faces = np.asarray(mesh_smp.triangles))
                            simplified_vertices = torch.tensor( np.asarray(obj_mesh_dis.vertices)).unsqueeze(0).float()
                            tree = KDTree( np.asarray(obj_mesh_dis.vertices))
                            smp_distances, indices = tree.query(original_vertices)


                            mesh_obj = Meshes(verts= simplified_vertices, faces=torch.tensor(np.asarray(obj_mesh_dis.faces)).unsqueeze(0))  
                            obj_normal = mesh_obj.verts_normals_packed().view(1,-1, 3)
                            hand_nn_dist_recon, hand_nn_idx_recon = utils_loss.get_NN( hand_vertices,simplified_vertices)
                            contact_map_bool_hand = (hand_nn_dist_recon<3e-4).float()
                            interior_hand = utils_loss.get_interior(obj_normal, simplified_vertices,hand_vertices, hand_nn_idx_recon).type(torch.bool)
                            contact_map_bool_hand[interior_hand] =  (contact_map_bool_hand[interior_hand]*-1).float()
                            interior_f = torch.zeros(torch.tensor(simplified_vertices).size(1)) 
                            interior_f[hand_nn_idx_recon.squeeze(0)] = hand_nn_dist_recon.squeeze(1).squeeze(0) 


                            hand_vertices_prior = torch.tensor(hand_vertices[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)]).float().unsqueeze(0)


                            obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(simplified_vertices, hand_vertices_prior)
                            mesh = Meshes(verts=hand_vertices, faces=rh_faces)
                            hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)
                            hand_normal_prior = hand_normal[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)].unsqueeze(0)

                            if hand_normal_prior.size(1)>0:
                                interior = utils_loss.get_interior(hand_normal_prior, hand_vertices_prior, simplified_vertices, obj_nn_idx_recon).type(torch.bool)  # True for interior

                                NN_src_xyz = batched_index_select(hand_vertices_prior, obj_nn_idx_recon)  # [B, 3000, 3]
                                NN_vector = NN_src_xyz - simplified_vertices  # [B, 3000, 3]
                                # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
                                NN_src_normal = batched_index_select(hand_normal_prior, obj_nn_idx_recon)
                                #interior = interior。float
                                interior_d = (NN_vector * NN_src_normal).sum(dim=-1)
                                #interior_dist=nn_dist[interior]
                                obj_nn_dist_recon[interior]= obj_nn_dist_recon[interior]*-1



                                contact_map_bool = ((obj_nn_dist_recon<1e-4)&(obj_nn_dist_recon > -2e-4)).float()

                                interior_d = interior_d * contact_map_bool
                                contact_map_bool[interior] = contact_map_bool[interior]*-1
                                contact_map = obj_nn_dist_recon

                                #obj_mesh_dis = trimesh.Trimesh(vertices = simplified_vertices[0].detach(),faces = np.asarray(mesh_smp.triangles))
                                hand_mesh_dis = trimesh.Trimesh(vertices = hand_vertices[0].detach(),faces = rh_faces[0].detach())
                                distance = find_point_distances(obj_mesh_dis,hand_mesh_dis)

                                if True:#distance.size(0)==contact_map_bool.size(1):

                                    distance = distance * (contact_map_bool==-1).float().squeeze(0)

                                    mask_num_l2 = contact_map_bool.size(1)
                                    contact_map_bool_l2 = pad_point_cloud_with_zeros(contact_map_bool.squeeze(0).unsqueeze(1), target_num=300).squeeze(1)
                                    distance_l2 = pad_point_cloud_with_zeros(distance.unsqueeze(1), target_num=300)
                                    distance_l2 = distance_l2.squeeze(1)
                                    #object_vertices_l1 = pad_point_cloud_with_zeros(object_vertices.squeeze(0), target_num=1200).unsqueeze(0)
                                    object_vertices_org_l2 = pad_point_cloud_with_zeros(simplified_vertices.squeeze(0), target_num=300).unsqueeze(0)
                                    normal_l2 = torch.tensor(obj_mesh_dis.vertex_normals)
                                    normal_l2 = pad_point_cloud_with_zeros(normal_l2, target_num=300).unsqueeze(0)
                                    
                                    #smp_distances_l2 = pad_point_cloud_with_zeros(torch.tensor(smp_distances).unsqueeze(0), target_num=1000)
                                    smp_distances_l2,indices_l2 = pad_point_cloud_with_zeros_index(torch.tensor(smp_distances).unsqueeze(1),torch.tensor(indices), target_num=1200)

                                    object_org_scaled =  pad_point_cloud_with_zeros(object_org_scaled.squeeze(0),12000).unsqueeze(0)
                                    print('add')


                                    data = {
                                                'contact_map_bool': contact_map_bool_base.unsqueeze(0).cuda(),
                                                'object_vertices': object_vertices.float().cuda(),
                                                'object_vertices_org': object_org_scaled.float().cuda(),
                                                #'offset': offset_batch.squeeze(0).squeeze(0),
                                                'movement_gt':movemant_gt_base.float().cuda(),
                                                'distance':torch.tensor(distance_base).unsqueeze(0).float().detach().cuda(),
                                                'normal':normal.float().cuda(),
                                                'mask_num':[mask_num],
                                                'contact_map_bool_l1': contact_map_bool_l1.unsqueeze(0).cuda(),
                                                #'object_vertices_l1': object_vertices_batch_l1.squeeze(0).float(),
                                                'object_vertices_org_l1': object_vertices_org_l1.float().cuda(),
                                                'distance_l1':distance_l1.float().detach().unsqueeze(0).cuda(),
                                                'normal_l1':normal_l1.float().cuda(),
                                                'mask_num_l1':[mask_num_l1],
                                                'index_l1':indices_l1.unsqueeze(0).cuda(),
                                                'smp_distances_l1':smp_distances_l1.float().cuda(),
                                                'contact_map_bool_l2': contact_map_bool_l2.unsqueeze(0).cuda(),
                                                #'object_vertices_l2': object_vertices_batch_l2.squeeze(0).float(),
                                                'object_vertices_org_l2': object_vertices_org_l2.float().cuda(),
                                                'distance_l2':distance_l2.float().detach().unsqueeze(0).cuda(),
                                                'normal_l2':normal_l2.float().cuda(),
                                                'mask_num_l2':[mask_num_l2],
                                                'index_l2':indices_l2.unsqueeze(0).cuda(),
                                                'smp_distances_l2':smp_distances_l2.float().cuda(),

                                        }
                                    object_pred ,movement= softnet(data)
                                    from pytorch3d.ops import taubin_smoothing
                                    from pytorch3d.structures import Meshes
                                    contact_map_bool = contact_map_bool_base[:point_num]
                                    object_pred = object_pred[:,:point_num,:]
                                    verts=[object_pred.squeeze(0).to('cuda')]
                                    #faces=[face[0]]
                                    face = [torch.tensor(obj_mesh.faces).to('cuda')]
                                    pytorch3d_mesh = Meshes(verts,face)
                                    pytorch3d_mesh = taubin_smoothing(meshes=pytorch3d_mesh, lambd=0.53, mu= -0.53, num_iter=2)
                                    obj_mesh.vertices = pytorch3d_mesh.verts_list()[0].cpu().detach().numpy()
                                    mask = abs(contact_map_bool).cpu().detach().bool().squeeze(0)
                                    #print(mask.size())point_num
                                    colors = np.ones((len(obj_mesh.vertices), 4)) * [1, 1, 1, 1]  # RGBA 白色
                                    colors[abs(contact_map_bool).cpu().detach().bool().squeeze(0)] = [0, 0, 1, 1]  
                                    colors[(contact_map_bool==-1).cpu().detach().bool().squeeze(0)] = [1, 0, 0, 1]  # RGBA 红色
                                    obj_mesh.visual.vertex_colors = colors
                                    
                                    print("deform")
            except:
                print("error")
            obj_mesh_tmp = trimesh.Trimesh(vertices=obj_mesh.vertices, faces=obj_mesh.faces)
            hand_mesh_gif.append(hand_mesh)
            obj_mesh_gif.append(obj_mesh_tmp)
            print('step:',i)




        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)



        colors = {'light_red': [0.85882353, 0.74117647, 0.65098039],
            'light_blue': [145/255, 191/255, 219/255]}

        images = []
        for i in range(0,100):
            vis.clear_geometries()
            trimesh_mesh = hand_mesh_gif[i]

            vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
            faces = o3d.utility.Vector3iVector(trimesh_mesh.faces)
            o3d_mesh_hand = o3d.geometry.TriangleMesh(vertices, faces)
            o3d_mesh_hand.paint_uniform_color(colors['light_red']) 
            o3d_mesh_hand.compute_vertex_normals()

            obj_mesh_ = obj_mesh_gif[i]
            vertices = o3d.utility.Vector3dVector(obj_mesh_.vertices)
            faces = o3d.utility.Vector3iVector(obj_mesh_.faces)
            o3d_mesh = o3d.geometry.TriangleMesh(vertices, faces)
            o3d_mesh.paint_uniform_color(colors['light_blue']) 
            o3d_mesh.compute_vertex_normals()

            vis.add_geometry(o3d_mesh)
            vis.add_geometry(o3d_mesh_hand)
            image = vis.capture_screen_float_buffer(do_render=True)
            image = np.asarray(image)
            image = (255 * image).astype(np.uint8)
            

            images.append(Image.fromarray(image))

        vis.destroy_window()


        if not os.path.exists("./deform/obj_{}/idx_{}".format(batch_idx,idx)):

            os.makedirs("./deform/obj_{}/idx_{}".format(batch_idx,idx))
        hand_mesh.export("./deform/obj_{}/idx_{}/hand".format(batch_idx,idx)+".ply")
        obj_mesh.export("./deform/obj_{}/idx_{}/obj".format(batch_idx,idx)+".ply")
        images[0].save('./deform/obj_{}/idx_{}/grasp_mesh.gif'.format(batch_idx,idx), save_all=True, append_images=images[1:], duration=50, loop=0)
