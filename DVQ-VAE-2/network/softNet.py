import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch import optim, nn, utils, Tensor
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import trimesh
from network.point_Unet import Point_Unet_block,PointCloudMappingNN,PointTransformer
from utils import utils_loss
import pytorch_lightning as pl
import lightning as L
from pytorch3d.loss import chamfer_distance
from torch.optim import lr_scheduler
from network.pointconv import PointConvDensityClsSsg
from pytorch3d.loss import mesh_normal_consistency
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures import Meshes
from utils.pointconv_util import PointConvDensitySetAbstraction
from pytorch3d.ops import taubin_smoothing
import os
from pytorch3d.ops.knn import knn_gather, knn_points
import numpy as np
import time
import mano

def align_w_scale(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        #return R, s, s1, t1 - t2
        return R, s, s1, t1 , t2 , s2
    else:
        return mtx2_t



def get_neighbors_with_trimesh(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh.vertex_neighbors

def neighbors_to_adjacency_tensor(neighbors, n_vertices):
    adjacency_matrix = torch.zeros((n_vertices, n_vertices), dtype=torch.float32)
    for i, nbrs in enumerate(neighbors):
        adjacency_matrix[i, list(nbrs)] = 1.0
    return adjacency_matrix




def deformation_loss(original_pc, deformed_pc, t, graph_nodes, adjacency_matrix):
    """
    计算形变损失函数。

    参数：
        original_pc (Tensor): 原始点云，形状为 (n_points, 3)。
        deformed_pc (Tensor): 形变后的点云，形状为 (n_points, 3)。
        t (Tensor): 平移向量，形状为 (m, 3)，m为图节点数量。
        graph_nodes (Tensor): 图节点位置，形状为 (m, 3)。
        adjacency_matrix (Tensor): 邻接矩阵，形状为 (m, m)。
        handle_constraints (List[Tuple[int, Tensor]]): 手柄约束。

    返回：
        loss (Tensor): 标量损失值。
    """
    device = original_pc.device
    m = graph_nodes.shape[0]
    
    # 旋转误差 Erot = 0，因为假设 Rj = I
    Erot = torch.tensor(0.0, device=device)
    
    # 计算正则化误差 Ereg
    # term = t_j - t_k - g_j
    t_j = t.unsqueeze(1).expand(-1, m, -1)  # (m, m, 3)
    t_k = t.unsqueeze(0).expand(m, -1, -1)  # (m, m, 3)
    g_j = graph_nodes.unsqueeze(1).expand(-1, m, -1)  # (m, m, 3)
    
    term = t_j - t_k - g_j  # (m, m, 3)
    
    # 应用邻接矩阵
    adjacency_mask = adjacency_matrix.bool()  # (m, m)
    term = term[adjacency_mask]  # (num_edges, 3)
    
    Ereg = torch.sum(torch.sum(term ** 2, dim=1) / 2.0)
    
    loss = Ereg 
    return loss



def deformation_loss_with_rotation(
    original_pc, deformed_pc, t, R, graph_nodes, adjacency_matrix, handle_constraints, lambda_reg=1.0, lambda_con=1.0
):
    """
    计算形变损失函数，考虑旋转矩阵 R_j。

    参数：
        original_pc (Tensor): 原始点云，形状为 (n_points, 3)。
        deformed_pc (Tensor): 形变后的点云，形状为 (n_points, 3)。
        t (Tensor): 平移向量，形状为 (m, 3)，m为图节点数量。
        R (Tensor): 旋转矩阵，形状为 (m, 3, 3)，m为图节点数量。
        graph_nodes (Tensor): 图节点位置，形状为 (m, 3)。
        adjacency_matrix (Tensor): 邻接矩阵，形状为 (m, m)。
        handle_constraints (List[Tuple[int, Tensor]]): 手柄约束。
        lambda_reg (float): 正则化损失的权重。
        lambda_con (float): 约束损失的权重。

    返回：
        loss (Tensor): 标量损失值。
    """
    device = original_pc.device
    m = graph_nodes.shape[0]

    # 计算旋转误差 E_{rot}
    Erot = 0.0
    adjacency_indices = torch.nonzero(adjacency_matrix, as_tuple=True)  # 获取所有邻接节点对 (j, k)
    for j, k in zip(adjacency_indices[0], adjacency_indices[1]):
        g_jk = graph_nodes[k] - graph_nodes[j]  # 初始距离向量 g_j
        rotation_error = R[j] @ g_jk - R[k] @ g_jk - (t[j] - t[k])  # 计算旋转误差
        Erot += torch.sum(rotation_error ** 2)

    # 计算正则化误差 E_{reg}
    t_j = t.unsqueeze(1).expand(-1, m, -1)  # (m, m, 3)
    t_k = t.unsqueeze(0).expand(m, -1, -1)  # (m, m, 3)
    g_j = graph_nodes.unsqueeze(1).expand(-1, m, -1)  # (m, m, 3)
    term = t_j - t_k - g_j  # (m, m, 3)
    adjacency_mask = adjacency_matrix.bool()  # (m, m)
    term = term[adjacency_mask]  # (num_edges, 3)
    Ereg = lambda_reg * torch.sum(torch.sum(term ** 2, dim=1) / 2.0)

    # 计算约束误差 E_{con}
    Econ = 0.0
    if handle_constraints:
        v_indices, qls = zip(*handle_constraints)
        v_indices = torch.tensor(v_indices, dtype=torch.long, device=device)
        qls = torch.stack(qls).to(device)  # (num_constraints, 3)

        # 受约束顶点的初始位置
        v_original = original_pc[v_indices]

        # 受约束顶点的目标位置
        v_deformed = v_original + t[0].unsqueeze(0).repeat(len(v_indices), 1)  # 假设仅受第一个节点的影响

        # 计算约束误差
        Econ = lambda_con * torch.sum((v_deformed - qls) ** 2) / 2.0

    # 总损失
    loss = Erot + Ereg + Econ
    return loss


class softNet(pl.LightningModule):
    def __init__(self):
        super(softNet, self).__init__()

        self.point_Unet = Point_Unet_block(input_dim=4, hidden_dim=64, output_dim=4)
        self.point_Unet_1 = Point_Unet_block(input_dim=8, hidden_dim=64, output_dim=8)

        self.point_Map = PointCloudMappingNN(input_dim=12, hidden_dim=128, output_dim=12)
        self.point_Map_1 = PointCloudMappingNN(input_dim=16, hidden_dim=128, output_dim=3)

        self.face = []
        self.rh_faces = []

    def load_face(self,batchsize = 4):
        sequences = [f'seq{i:02d}' for i in range(1, 14)]
        for seq in sequences:
            object_path = os.path.join('obj_path', seq)
            object_mesh = trimesh.load(os.path.join(object_path, 'org_mesh.ply')) 
            self.face.append(torch.tensor(object_mesh.faces).unsqueeze(0).to('cuda'))
        with torch.no_grad():
            rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                                model_type='mano',
                                use_pca=True,
                                num_pca_comps=45,
                                batch_size=batchsize,
                                flat_hand_mean=True).to('cuda')
        rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).contiguous() # [1, 1538, 3], face triangle indexes
        self.rh_faces = rh_faces.repeat(batchsize, 1, 1).to('cuda') # [N, 1538, 3]



    # def validation_step(self, batch, batch_idx):
    #     data_dict  = batch
    #     obj_pc = data_dict['object_vertices']
    #     normal = data_dict['normal']
    #     distance = data_dict['distance']
    #     contact_map_bool = data_dict['contact_map_bool']
    #     object_org = data_dict['object_vertices_org']
    #     offset = data_dict['offset']
    #     face = data_dict['face']
    #     movement_gt =  data_dict['movement_gt']#.squeeze(0)
    #     abs_map = abs(contact_map_bool)
    #     point_num =abs_map.sum()

    #     gt_normal = normal#gt_mesh.verts_normals_packed()#.view(-1, 778, 3)
    #     #print(contact_map_bool.size())
    #     #contact_map = contact_map*abs_map
    #     grid_size = torch.tensor([0.001]).to('cuda')
    #     #movement_gt =  obj_pc - object_org 

    #     data_dict ={
    #         'offset':offset.squeeze(0),
    #         'feat':torch.cat((contact_map_bool.unsqueeze(2),distance.unsqueeze(2),gt_normal ),dim=2).squeeze(0), 
    #         'coord':object_org.squeeze(0), 
    #         'grid_size':grid_size
    #     }
    #     #print(contact_map)
    #     obj_pred,movement,d = self.forward(data_dict)
    #     obj_pred = obj_pred.unsqueeze(0)

    #     for i in range(1,offset.size()[1]):
    #         obj_pred_smooth = torch.cat((obj_pred_smooth, pytorch3d_mesh.verts_list()[i]),0)
    #     obj_pred_smooth = obj_pred_smooth.unsqueeze(0)
    #     movement = obj_pred_smooth - object_org
    #     #trimesh.Trimesh(vertices=obj_pred[0].detach().cpu().numpy(), faces=self.rh_mano.faces)
    #     #loss_smoothing = mesh_laplacian_smoothing(obj_pred[:,:offset[0][0],:], method='uniform')
    #     loss_chamfer , _= chamfer_distance(obj_pred_smooth[:,:offset[0][0],:], obj_pc[:,:offset[0][0],:], point_reduction='sum', batch_reduction='mean') 
    #     for i in range(0,offset.size()[1]-1):
    #         loss_chamfer_tmp,_ =  chamfer_distance(obj_pred_smooth[:,offset[0][i]:offset[0][i+1],:], obj_pc[:,offset[0][i]:offset[0][i+1],], point_reduction='sum', batch_reduction='mean')
    #         loss_chamfer += loss_chamfer_tmp
    #     #loss_smoothing = mesh_laplacian_smoothing(pytorch3d_mesh, method='uniform')
    #     loss_mse = F.mse_loss(movement_gt, movement, reduction='none').sum()
    #     loss_mse_2 = F.mse_loss(obj_pred, obj_pc, reduction='none').sum()/obj_pc.size(1)
    #     cos = nn.CosineSimilarity(dim=2)
    #     cos_sim = cos(movement_gt, movement)
    #     cos_sim_log = torch.log((cos_sim+1.00001)/2)*-1*abs(contact_map_bool)

    #     loss_cos = cos_sim_log.sum(dim=1, keepdim=True)/point_num#.to('cuda'))



    #     loss =  loss_mse + 0.3 * loss_cos + 0.2 * loss_chamfer +loss_mse_2 + 0.001*deform_loss#+ 10 * loss_smoothing  +loss_consistency#+loss_d
    #     self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("val_loss_chamfer", loss_chamfer, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("val_loss_mse", loss_mse, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("val_loss_cos", loss_cos, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("val_loss_mse_2", loss_mse_2, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("val_loss_deform", deform_loss, on_epoch=True, prog_bar=True, logger=True)



    def CMap_loss3(self,obj_xyz, hand_xyz, cmap):
        '''
        # prior cmap loss on gt cmap
        :param obj_xyz: [B, N1, 3]
        :param hand_xyz: [B, N2, 3]
        :param cmap: [B, N1] for contact map from NN dist thresholding
        :param hand_faces_index: [B, 1538, 3] hand index in [0,N2] for 3 vertices in a face
        :return:
        '''

        # finger_vertices = [309, 317, 318, 319, 320, 322, 323, 324, 325,
        #                    326, 327, 328, 329, 332, 333, 337, 338, 339, 343, 347, 348, 349,
        #                    350, 351, 352, 353, 354, 355,  # 2nd finger
        #                    429, 433, 434, 435, 436, 437, 438, 439, 442, 443, 444, 455, 461, 462, 463, 465, 466,
        #                    467,  # 3rd
        #                    547, 548, 549, 550, 553, 566, 573, 578,  # 4th
        #                    657, 661, 662, 664, 665, 666, 667, 670, 671, 672, 677, 678, 683, 686, 687, 688, 689, 690, 691,
        #                    692, 693, 694, 695,  # 5th
        #                    736, 737, 738, 739, 740, 741, 743, 753, 754, 755, 756, 757, 759, 760, 761, 762, 763, 764, 766,
        #                    767, 768,  # 1st
        #                    73, 96, 98, 99, 772, 774, 775, 777]  # hand

        B = obj_xyz.size(0)

        obj_CD, _ = utils_loss.get_NN(obj_xyz, hand_xyz)  # [B, N1] NN distance from obj pc to hand pc
        cmap = (cmap == -1)
        # compute contact map loss
        n_points = torch.sum(cmap)
        cmap_loss = obj_CD[cmap].sum() / (B * n_points)

        return 3000.0 * cmap_loss


    def inter_penetr_loss(self,hand_xyz, hand_face, obj_xyz, nn_dist, nn_idx):
        '''
        get penetrate object xyz and the distance to its NN
        :param hand_xyz: [B, 778, 3]
        :param hand_face: [B, 1538, 3], hand faces vertex index in [0:778]
        :param obj_xyz: [B, 3000, 3]
        :return: inter penetration loss
        '''
        B = hand_xyz.size(0)
        mesh = Meshes(verts=hand_xyz, faces=hand_face)
        hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)

        # if not nn_dist:
        #     nn_dist, nn_idx = utils_loss.get_NN(obj_xyz, hand_xyz)
        interior = utils_loss.get_interior(hand_normal, hand_xyz, obj_xyz, nn_idx).type(torch.bool)  # True for interior
        interior_dist=nn_dist[interior]
        penetr_dist = (interior_dist).sum() / B  # batch reduction
        return 100.0 * penetr_dist

    def get_interior(self,src_face_normal, src_xyz, trg_xyz, trg_NN_idx):
        '''
        :param src_face_normal: [B, 778, 3], surface normal of every vert in the source mesh
        :param src_xyz: [B, 778, 3], source mesh vertices xyz
        :param trg_xyz: [B, 3000, 3], target mesh vertices xyz
        :param trg_NN_idx: [B, 3000], index of NN in source vertices from target vertices
        :return: interior [B, 3000], inter-penetrated trg vertices as 1, instead 0 (bool)
        '''
        N1, N2 = src_xyz.size(1), trg_xyz.size(1)

        # get vector from trg xyz to NN in src, should be a [B, 3000, 3] vector
        NN_src_xyz = batched_index_select(src_xyz, trg_NN_idx)  # [B, 3000, 3]
        NN_vector = NN_src_xyz - trg_xyz  # [B, 3000, 3]

        # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
        NN_src_normal = batched_index_select(src_face_normal, trg_NN_idx)

        interior = (NN_vector * NN_src_normal).sum(dim=-1) > 0  # interior as true, exterior as false
        return interior

    def batched_index_select(self,input, index, dim=1):
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



    def get_NN(self,src_xyz, trg_xyz,src_mask, k=1):
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
        src_nn = knn_points(src_xyz, trg_xyz, lengths1=src_mask, lengths2=trg_lengths, K=k)  # [dists, idx]
        nn_dists = src_nn.dists[..., 0]
        nn_idx = src_nn.idx[..., 0]
    
        return nn_dists, nn_idx

    def training_step(self, batch, batch_idx):

        # training_step defines the train loop.
        # it is independent of forward
        data_dict  = batch
        obj_pc = data_dict['object_vertices']
        normal = data_dict['normal']
        distance = data_dict['distance']
        contact_map_bool = data_dict['contact_map_bool']
        object_org = data_dict['object_vertices_org']
        movement_gt =  data_dict['movement_gt']
        mask_num = data_dict['mask_num']
        hand_xyz = data_dict['hand_xyz']
        face_id = data_dict['face_id']

        object_vertices_l1 = data_dict['object_vertices_org_l1']
        index_l1 = data_dict['index_l1']
        contact_map_bool_l1 = data_dict['contact_map_bool_l1']
        #object_org_l1 = data_dict['object_vertices_org_l1']
        distance_l1 = data_dict['distance_l1']
        normal_l1 = data_dict['normal_l1']
        mask_num_l1 = data_dict['mask_num_l1']
        smp_distances_l1 = data_dict['smp_distances_l1']

        object_vertices_l2 = data_dict['object_vertices_org_l2']
        index_l2 = data_dict['index_l2']
        contact_map_bool_l2 = data_dict['contact_map_bool_l2']
        #object_org_l2 = data_dict['object_vertices_org_l2']
        distance_l2 = data_dict['distance_l2']
        normal_l2 = data_dict['normal_l2']
        mask_num_l2 = data_dict['mask_num_l2']
        smp_distances_l2 = data_dict['smp_distances_l2']

        mask = torch.zeros_like(distance)
        for i in range(0,mask_num.size()[0]):
            mask[i,:mask_num[i]] = 1
        mask = mask.unsqueeze(2)

        mask_l1 = torch.zeros_like(distance_l1)
        for i in range(0,mask_num_l1.size()[0]):
            mask_l1[i,:mask_num_l1[i]] = 1
        mask_l1 = mask_l1.unsqueeze(2)

        mask_l2 = torch.zeros_like(distance_l2)
        for i in range(0,mask_num_l2.size()[0]):
            mask_l2[i,:mask_num_l2[i]] = 1
        mask_l2 = mask_l2.unsqueeze(2)        
        contact_map_bool_org = contact_map_bool


        contact_map_bool = abs(contact_map_bool)
        contact_map_bool_l1 = abs(contact_map_bool_l1)
        contact_map_bool_l2 = abs(contact_map_bool_l2)

        abs_map = abs(contact_map_bool)
        point_num =abs_map.sum()
        gt_normal =normal# gt_mesh.verts_normals_packed().detach()#.view(-1, 778, 3)

        feat =  torch.cat((contact_map_bool.unsqueeze(2),distance.unsqueeze(2)*normal*-1 ),dim=2).squeeze(0)#5
        #print(feat.size())
        l1_feat = torch.cat((contact_map_bool_l1.unsqueeze(2),distance_l1.unsqueeze(2)*normal_l1*-1 ),dim=2).squeeze(0)#5
        l2_feat = torch.cat((contact_map_bool_l2.unsqueeze(2),distance_l2.unsqueeze(2)*normal_l2*-1 ),dim=2).squeeze(0)#5
        feat_trans = self.point_Unet(feat,object_org,normal, index_l1, 1200,mask) #5
        
        l1_feat_trans = self.point_Unet_1(torch.cat((feat_trans,l1_feat),dim = 2),object_vertices_l1,normal_l1, index_l2, 300,mask_l1)#5
        
        l2_upsample = self.point_Map(torch.cat((l2_feat,l1_feat_trans),dim = 2),object_vertices_l1,normal_l1,index_l2,None,mask_l1)#15

        l1_upsample = self.point_Map_1(l2_upsample,object_org,normal,index_l1,feat,mask)#3


        movement =  l1_upsample  #*normal*-1
  
        movement = movement*mask
        obj_pred = object_org+movement
        #obj_pred = obj_pred#.unsqueeze(0)

        obj_pred_smooth = obj_pred#pytorch3d_mesh.verts_list()[0]

        #obj_pred_smooth = obj_pred_smooth.unsqueeze(0)
        #movement = obj_pred_smooth - object_org

        faces = []
        verts = []
        # print(mask_num)
        # print(mask_num.size())
        # print(obj_pred_smooth.size())
        # print(len(self.face))
        # print(face_id)
        # print(self.face[face_id[0]].size())
        # print(obj_pred_smooth[0,:mask_num[0],:].size())
        loss_chamfer = 0
        for i in range(0,mask_num.size()[0]):
            verts.append(obj_pred_smooth[i,:mask_num[i],:].squeeze(0))
            faces.append(self.face[face_id[i]-1].squeeze(0))
            #print(obj_pred_smooth[i,:mask_num[i],:].size())
            loss_chamfer_tmp , _= chamfer_distance(obj_pred_smooth[i,:mask_num[i],:].unsqueeze(0), obj_pc[i,:mask_num[i],:].unsqueeze(0), point_reduction='sum', batch_reduction='mean') 
            loss_chamfer += loss_chamfer_tmp
        new_src_mesh = Meshes(verts,faces)
        loss_normal = mesh_normal_consistency(new_src_mesh)
        loss_chamfer = loss_chamfer/mask_num.size()[0]
        # mesh laplacian smoothing
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
                
        #loss_smoothing = mesh_laplacian_smoothing(pytorch3d_mesh, method='uniform')

        loss_mse = F.mse_loss(movement_gt*mask, movement*mask, reduction='none').sum()/mask.sum()
        loss_mse_2 = (F.mse_loss(obj_pred*mask, obj_pc*mask, reduction='none').sum()/mask.sum())
        cos = nn.CosineSimilarity(dim=2)
        cos_sim = cos(movement_gt*mask, movement*mask)
        cos_sim_log = torch.log((cos_sim+1.00001)/2)*-1*abs(contact_map_bool)
        #torch.save(cos_sim_log, "./myTensor.pt")
        loss_cos = (cos_sim_log.sum(dim=1, keepdim=True)/abs(contact_map_bool).sum(dim = 1)).mean()#.to('cuda'))
        loss_contact = self.CMap_loss3(obj_pred_smooth, hand_xyz, contact_map_bool_org)


        obj_nn_dist_recon, obj_nn_idx_recon = self.get_NN(obj_pred_smooth, hand_xyz,mask_num)
        penetr_loss = self.inter_penetr_loss(hand_xyz, self.rh_faces, obj_pred_smooth,
                                        obj_nn_dist_recon, obj_nn_idx_recon)

        
        loss =  100*loss_mse+ 0.1*loss_chamfer+loss_contact+10*loss_laplacian+loss_normal#+penetr_loss#+#0.001*deform_loss#+ 10 * loss_smoothing  +loss_consistency#+loss_d


                
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_chamfer", loss_chamfer, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_penetr", penetr_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_contact", loss_contact, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_mse", loss_mse, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_cos", loss_cos, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_mse_2", loss_mse_2, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_normal", loss_normal, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_laplacian", loss_laplacian, on_epoch=True, prog_bar=True, logger=True)
        
        #self.log("train_loss_deform", deform_loss, on_epoch=True, prog_bar=True, logger=True)

        return loss


    # def on_train_epoch_end(self):
    #     tensorboard = self.logger.experiment
    #     for name, layer in self.named_parameters():
    #             if layer.grad is not None:
    #                 tensorboard.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(),self.epoch)
    #                 tensorboard.add_histogram(name + '_data', layer.cpu().data.numpy(),self.epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-5)
 

        max_lr = 1e-5
        base_lr = max_lr/4.0
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
            base_lr=base_lr,max_lr=max_lr,
            step_size_up=5*300,cycle_momentum=False)
        self.print("set lr = "+str(max_lr))
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(200 * x) for x in [0.3, 0.6, 0.8, 0.9]], gamma=0.5)

        return ([optimizer],[scheduler])



    def forward(self, data_dict):
        
        #point_with_map = torch.cat((object_org, contactmap.unsqueeze(2)), dim=2)
        # #print('point_with_map',point_with_map.size())   
        # contact_feat,_,_ = self.obj_encoder_soft(point_with_map.permute(0, 2, 1))
        # deform = self.deform_decoder(contact_feat).resize(B,3000,3)

        obj_pc = data_dict['object_vertices']
        normal = data_dict['normal']
        distance = data_dict['distance']
        contact_map_bool = data_dict['contact_map_bool']
        object_org = data_dict['object_vertices_org']
        movement_gt =  data_dict['movement_gt']
        mask_num = data_dict['mask_num']

        object_vertices_l1 = data_dict['object_vertices_org_l1']
        index_l1 = data_dict['index_l1']
        contact_map_bool_l1 = data_dict['contact_map_bool_l1']
        #object_org_l1 = data_dict['object_vertices_org_l1']
        distance_l1 = data_dict['distance_l1']
        normal_l1 = data_dict['normal_l1']
        mask_num_l1 = data_dict['mask_num_l1']
        smp_distances_l1 = data_dict['smp_distances_l1']

        object_vertices_l2 = data_dict['object_vertices_org_l2']
        index_l2 = data_dict['index_l2']
        contact_map_bool_l2 = data_dict['contact_map_bool_l2']
        #object_org_l2 = data_dict['object_vertices_org_l2']
        distance_l2 = data_dict['distance_l2']
        normal_l2 = data_dict['normal_l2']
        mask_num_l2 = data_dict['mask_num_l2']
        smp_distances_l2 = data_dict['smp_distances_l2']
        contact_map_bool = abs(contact_map_bool)
        contact_map_bool_l1 = abs(contact_map_bool_l1)
        contact_map_bool_l2 = abs(contact_map_bool_l2)
        mask = torch.zeros_like(distance)
        print(mask_num)
        for i in range(0,obj_pc.size()[0]):
            mask[i,:mask_num[i]] = 1
        mask = mask.unsqueeze(2)
        print(mask.size())
        print(mask)
        mask_l1 = torch.zeros_like(distance_l1)
        for i in range(0,obj_pc.size()[0]):
            mask_l1[i,:mask_num_l1[i]] = 1
        mask_l1 = mask_l1.unsqueeze(2)

        mask_l2 = torch.zeros_like(distance_l2)
        for i in range(0,obj_pc.size()[0]):
            mask_l2[i,:mask_num_l2[i]] = 1
        mask_l2 = mask_l2.unsqueeze(2)  
        abs_map = abs(contact_map_bool)
        point_num =abs_map.sum()
        gt_normal =normal# gt_mesh.verts_normals_packed().detach()#.view(-1, 778, 3)


        feat =  torch.cat((contact_map_bool.unsqueeze(2),distance.unsqueeze(2)*normal*-1 ),dim=2)
        #print
        l1_feat = torch.cat((contact_map_bool_l1.unsqueeze(2),distance_l1.unsqueeze(2)*normal_l1*-1 ),dim=2)
        l2_feat = torch.cat((contact_map_bool_l2.unsqueeze(2),distance_l2.unsqueeze(2)*normal_l2*-1 ),dim=2)
        feat_trans = self.point_Unet(feat,object_org,normal, index_l1, 1200,mask) #5
        
        l1_feat_trans = self.point_Unet_1(torch.cat((feat_trans,l1_feat),dim = 2),object_vertices_l1,normal_l1, index_l2, 300,mask_l1)#5
        
        l2_upsample = self.point_Map(torch.cat((l2_feat,l1_feat_trans),dim = 2),object_vertices_l1,normal_l1,index_l2,None,mask_l1)#15

        l1_upsample = self.point_Map_1(l2_upsample,object_org,normal,index_l1,feat,mask)#3

        movement = l1_upsample*mask
        obj_pred =object_org+movement
        ##print(movemant)
        return obj_pred,movement#, d['feat']
    
class MyFullyConnectedLayer(nn.Module):
    def __init__(self):
        super(MyFullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(256, 512)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 30)

    def forward(self, x):
        x = self.fc(x)  
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = x.view(-1, 10, 3)
        return x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()
        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)
        #print('encoder', self.MLP)

    def forward(self, x):

        #print('x size before MLP {}'.format(x.size()))
        x = self.MLP(x)
        #print('x size after MLP {}'.format(x.size()))
        means = self.linear_means(x)
        #print('mean size {}, log_var size {}'.format(means.size(), log_vars.size()))
        return means


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):
        super().__init__()

        self.MLP = nn.Sequential()
        input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

    def forward(self, z):
        x = self.MLP(z)
        return x



if __name__ == '__main__':
    model = affordanceNet(
        obj_inchannel=4)
    obj_xyz = torch.randn(3, 4, 3000)
    hand_param = torch.randn(3, 3, 778)
    print('params {}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    #model.eval()
    model.train()
    recon, _, _, _ = model(obj_xyz, hand_param)
    #recon = model(obj_xyz, hand_param)
    print(recon.size())
