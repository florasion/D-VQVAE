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
        #self.obj_cmap_path = '/data/zz/ObMan/obman/processed/obj_cmap_contactdb_{}.npy'.format(mode)
        self.hand_v_path = 'hand_path/hand.npy'
        #self.z_q_path = '/data/zz/ObMan/obman/processed/z_q_{}.npy'.format(mode)
        self.transformer = False
        self.gentransformer=True
        self.base_path = 'dataset_path/HMDO'
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
        self.__load_dataset__()

        self.dataset_size = int(len(self.all_contact_map)/self.batch_size)

        self.transform = transforms.ToTensor()
        self.sample_nPoint = 3000
        #self.batch_size = batch_size

    def __load_dataset__(self):
        print('loading dataset start')
        self.process_sequences()
        #if self.transformer:
        #    self.all_z_q = np.load(self.z_q_path)
        print('loading dataset finish')

    def process_sequences(self):

        sequences = [f'sequence{i:02d}' for i in range(1, 14)]
        with torch.no_grad():
            rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                                model_type='mano',
                                use_pca=False,
                                num_pca_comps=51,
                                batch_size=1,
                                flat_hand_mean=True)
        rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).contiguous() # [1, 1538, 3], face triangle indexes    

        max=0
        for seq in sequences:
            print(seq)
            

            hand_path = os.path.join(self.base_path, seq, self.hand_folder)
            object_path = os.path.join(self.base_path, seq, self.object_folder)
            object_mesh = trimesh.load(os.path.join(object_path, 'org_mesh.ply')) 

            object_full = object_mesh.vertices
            hand_annotation_path = os.path.join(self.base_path, seq, self.hand_annotation)
            i = 0
            offset = 0
            contact_map_batch = torch.tensor([])
            contact_map_bool_batch = torch.tensor([])
            object_vertices_batch = torch.tensor([])
            object_vertices_org_batch = torch.tensor([])
            offset_batch = torch.tensor([])
            mano_p_before = torch.tensor([])
            object_vertices_before = torch.tensor([])

            timestep = 0 
            test_i = 0
            for hand_file, object_file,annotation in zip(sorted(os.listdir(hand_path)), sorted(os.listdir(object_path)),sorted(os.listdir(hand_annotation_path))):


                if test_i <10:
                    test_i+=1
                    if self.mode == 'train':
                        continue
                if test_i >= 10:
                    if self.mode == 'val':
                        continue

                try:
                #if True:
                    # hand_vertices , face_id =  trimesh.sample.sample_surface(trimesh.load(os.path.join(hand_path, hand_file)), 1000)
                    # object_vertices_full = trimesh.load(os.path.join(object_path, object_file))
                    # object_vertices , face_id =  trimesh.sample.sample_surface(object_vertices_full, 3000)
                    # object_vertices_full = object_vertices_full.vertices
                    # with open(os.path.join(hand_annotation_path,annotation), 'r') as file:
                    #     content = file.read()
                    # numbers = content.split(',')
                    # scale=float(numbers[0])

                    # hand_vertices=hand_vertices/scale
                    # object_vertices=object_vertices/scale
                    # object_full_scaled = object_full/scale
                    # object_vertices_full_scaled = object_vertices_full/scale
                    # object_vertices_org_scaled = object_vertices_org/scale
                    # R, s, s1, t1 , t2 , s2 = align_w_scale(object_vertices_full_scaled,object_full_scaled,return_trafo = True)
                    # object_vertices_org_rot = np.dot((object_vertices_org_scaled - t2)/s2, R.T) * s* s1 + t1
                    #hand_vertices , face_id =  trimesh.sample.sample_surface(trimesh.load(os.path.join(hand_path, hand_file)), 1000)
                    object_vertices_full = trimesh.load(os.path.join(object_path, object_file))
                    object_vertices = object_vertices_full.vertices
                    with open(os.path.join(hand_annotation_path,annotation), 'r') as file:
                        content = file.read()
                    numbers = content.split(',')
                    scale=float(numbers[0])
                    number = []
                    for i in range(1,len(numbers)-1):
                        #print(numbers[i])
                        number.append(float(numbers[i]))
                    content = torch.tensor(number).unsqueeze(0)
                    mano_p =content
                    
                    recon_param  = torch.tensor([[0,0,0,0,0,0,0,0,0,0]])
                    recon_mano = rh_mano(betas=recon_param, global_orient=mano_p[:,3:6],
                                                        hand_pose=mano_p[:,6:51], transl=mano_p[:, 0:3])
                    hand_vertices = recon_mano.vertices#.to(device)
                    #hand_vertices=hand_vertices/scale
                    object_vertices=object_vertices/scale
                    object_org_scaled = object_full/scale




                    R,  t1  = align_w_scale(object_vertices,object_org_scaled,return_trafo = True)
                    object_org_scaled = np.dot((object_org_scaled ), R.T)  + t1
                    object_org_scaled = torch.tensor(object_org_scaled).float().unsqueeze(0)
                    object_vertices = torch.tensor(object_vertices).float().unsqueeze(0)

                    hand_vertices = hand_vertices
                    

                    movement_gt = object_vertices - object_org_scaled 




                    d =  torch.norm(torch.tensor(movement_gt), dim=2, keepdim=True)
                    
                    if np.array(((d < 3e-3) & (d > 3e-4))).sum()==0:
                        mean_move = 0
                        object_org_scaled = object_org_scaled
                    else:
                        mean_move = (movement_gt*np.array(((d < 3e-3) & (d > 3e-4)))).sum(0)/np.array(((d < 3e-3) & (d > 3e-4))).sum()
                        object_org_scaled = object_org_scaled + mean_move
                    
                    movement_gt = object_vertices - object_org_scaled 



                    obj_mean = object_org_scaled[0].mean(axis=0)
                    object_org_scaled = object_org_scaled - obj_mean
                    object_vertices = object_vertices - obj_mean
                    hand_vertices = hand_vertices - obj_mean



                    mesh_obj = Meshes(verts= torch.tensor( object_org_scaled), faces=torch.tensor(object_vertices_full.faces).unsqueeze(0))  
                    obj_normal = mesh_obj.verts_normals_packed().view(1,-1, 3)
                    hand_nn_dist_recon, hand_nn_idx_recon = utils_loss.get_NN( hand_vertices,object_org_scaled)
                    contact_map_bool_hand = (hand_nn_dist_recon<3e-4).float()
                    interior_hand = utils_loss.get_interior(obj_normal, object_org_scaled,hand_vertices, hand_nn_idx_recon).type(torch.bool)
                    contact_map_bool_hand[interior_hand] =  (contact_map_bool_hand[interior_hand]*-1).float()
                    interior_f = torch.zeros(torch.tensor(object_org_scaled).size(1)) 
                    interior_f[hand_nn_idx_recon.squeeze(0)] = hand_nn_dist_recon.squeeze(1).squeeze(0) 

                    
                    hand_vertices_prior = torch.tensor(hand_vertices[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)]).float().unsqueeze(0)

                    if hand_vertices_prior.size(1)<1:
                        continue
                    obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(object_org_scaled, hand_vertices_prior)
                    mesh = Meshes(verts=hand_vertices, faces=rh_faces)
                    hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)
                    hand_normal_prior = hand_normal[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)].unsqueeze(0)
                    # if not nn_dist:
                    #     nn_dist, nn_idx = utils_loss.get_NN(obj_xyz, hand_xyz)
                    print(hand_vertices_prior.size())
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
                    if contact_map_bool.sum() <300:
                        continue
                    interior_d = interior_d * contact_map_bool
                    contact_map_bool[interior] = contact_map_bool[interior]*-1
                    contact_map = obj_nn_dist_recon

                    obj_mesh_dis = trimesh.Trimesh(vertices = object_org_scaled[0].detach(),faces = object_vertices_full.faces)
                    hand_mesh_dis = trimesh.Trimesh(vertices = hand_vertices[0].detach(),faces = rh_faces[0].detach())
                    distance = find_point_distances(obj_mesh_dis,hand_mesh_dis)
                    distance = distance * (contact_map_bool==-1).float().squeeze(0)

                    print(hand_file)
                    self.all_interior_d.append(interior_d)
                    self.all_contact_map.append(contact_map)
                    self.all_contact_map_bool.append(contact_map_bool)
                    self.all_object_vertices.append(object_vertices)
                    self.all_object_vertices_org.append(object_org_scaled)
                    self.all_movement_gt.append(movement_gt)
                    self.all_face.append(torch.tensor(object_vertices_full.faces, dtype=torch.int64))
                    self.all_mano_p.append(mano_p)
                    self.all_mano_p_before.append(mano_p_before)
                    self.all_object_vertices_before.append(object_vertices_before)
                    self.all_interior_f.append(interior_f.unsqueeze(0))
                    self.all_distance.append(torch.tensor(distance).unsqueeze(0))
                    self.all_normal.append(torch.tensor(obj_mesh_dis.vertex_normals).unsqueeze(0))


                except:
                    print('error')
                    continue
            #break
                #self.all_object_vertices_full.append(object_full)
    def __len__(self):
        return self.dataset_size# - (self.dataset_size % self.batch_size)  # in case of unmatched mano batch size

    def __getitem__(self, idx):
        # obj_pc
        # obj_pc = torch.tensor(self.all_object_vertices[idx], dtype=torch.float32)  # [4, 3000]
        # obj_org = torch.tensor(self.all_object_vertices_org[idx], dtype=torch.float32)
        offset = 0
        offset+= self.all_object_vertices[idx*self.batch_size].size(1)
        contact_map_batch = self.all_contact_map[idx*self.batch_size]
        interior_d_batch = self.all_interior_d[idx*self.batch_size]
        contact_map_bool_batch = self.all_contact_map_bool[idx*self.batch_size]
        interior_f_batch = self.all_interior_f[idx*self.batch_size]
        distance_batch = self.all_distance[idx*self.batch_size]

        object_vertices_batch = self.all_object_vertices[idx*self.batch_size]
        object_vertices_org_batch = self.all_object_vertices_org[idx*self.batch_size]#.unsqueeze(0)
        normal_batch = self.all_normal[idx*self.batch_size]

        face_batch = [self.all_face[idx*self.batch_size]]
        offset_batch = torch.tensor(offset).unsqueeze(0)
        for i in range(1,self.batch_size):


            offset+= self.all_object_vertices[idx*self.batch_size+i].size(1)
            contact_map_batch = torch.cat((contact_map_batch,self.all_contact_map[idx*self.batch_size+i]),1)
            contact_map_bool_batch = torch.cat((contact_map_bool_batch,self.all_contact_map_bool[idx*self.batch_size+i]),1)
            interior_d_batch = torch.cat((interior_d_batch,self.all_interior_d[idx*self.batch_size+i]),1)
            interior_f_batch = torch.cat((interior_f_batch,self.all_interior_f[idx*self.batch_size+i]),1)
            distance_batch = torch.cat((distance_batch,self.all_distance[idx*self.batch_size+i]),1)

            object_vertices_batch = torch.cat((object_vertices_batch, self.all_object_vertices[idx*self.batch_size+i]),1)
            object_vertices_org_batch = torch.cat((object_vertices_org_batch,self.all_object_vertices_org[idx*self.batch_size+i]),1)
            normal_batch = torch.cat((normal_batch,self.all_normal[idx*self.batch_size+i]),1)
            
            face_batch.append(self.all_face[idx*self.batch_size+i])
            offset_batch = torch.cat((offset_batch,torch.tensor(offset).unsqueeze(0)),0)
        movement_gt_batch = object_vertices_batch-object_vertices_org_batch
        data = {
            #'contact_map': contact_map_batch.squeeze(0),
            'contact_map_bool': contact_map_bool_batch.squeeze(0),
            'object_vertices': object_vertices_batch.squeeze(0).float(),
            'object_vertices_org': object_vertices_org_batch.squeeze(0).float(),
            'offset': offset_batch.squeeze(0).squeeze(0),
            'movement_gt':movement_gt_batch.squeeze(0).float(),
            'face':face_batch,
            # 'interior_d':interior_d_batch.squeeze(0).float().detach(),
            # 'interior_f':interior_f_batch.squeeze(0).float().detach(),
            'distance':distance_batch.squeeze(0).float().detach(),
            'normal':normal_batch.squeeze(0).float()
        }


        # # obj cmap contactdb
        # #obj_cmap = torch.tensor(self.all_obj_cmap[idx])  # [3000, 10]
        # #obj_cmap = obj_cmap > 0

        # # hand mano param
        # hand_v = torch.tensor(self.all_hand_vertices[idx], dtype=torch.float32)  # [778]
        # #object_full = torch.tensor(self.all_object_vertices_full[idx], dtype=torch.float32)
        # #print(hand_v.size())
        # #if self.gentransformer:
        # #    return (obj_pc,hand_param,idx)
        # #if self.transformer:
        # #    z_q = torch.tensor(self.all_z_q[idx], dtype=torch.float32)
        # #    return (obj_pc,hand_param,z_q,idx)

        #data = self.all_data[idx]
        return data

