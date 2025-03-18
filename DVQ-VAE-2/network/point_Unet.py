import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch import optim, nn, utils, Tensor
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from network.pointnet_encoder import PointNetEncoder
import trimesh
from utils import utils_loss
from pytorch3d.loss import chamfer_distance
from network.pointconv import PointConvDensityClsSsg
from pytorch3d.structures import Meshes
from utils.pointconv_util import PointConvDensitySetAbstraction
from pytorch3d.ops import taubin_smoothing
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, f = x.size()
        x = x.view(-1, f)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(b, n, -1)
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F

class PointTransformerLayer(nn.Module):
    def __init__(self, in_channels,out_channels, k=16, n_head = 1):
        super(PointTransformerLayer, self).__init__()
        self.k = k  # Number of neighbors
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Linear layers for key, query, and value transformations
        self.fc_q = MLP(input_dim = in_channels,hidden_dim = 256,output_dim = out_channels * n_head)#nn.Linear(in_channels, out_channels)
        self.fc_k = MLP(input_dim = in_channels,hidden_dim = 256,output_dim = out_channels * n_head)#nn.Linear(in_channels, out_channels)
        self.fc_v = MLP(input_dim = in_channels,hidden_dim = 256,output_dim = out_channels * n_head)#nn.Linear(in_channels, out_channels)

        # Positional encoding
        self.pos_encoding = nn.Sequential(
            nn.Linear(2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.dropout = nn.Dropout(0.7)
        self.dropout_out = nn.Dropout(0.7)
        self.layer_norm = nn.LayerNorm(out_channels, eps=1e-6)
        # Linear layer for attention output transformation
        self.fc_out = MLP(input_dim = out_channels,hidden_dim = 256,output_dim = out_channels)

    def forward(self, x, pos,normal,mask):
        """
        x: (B, N, C) where B is batch size, N is number of points, C is feature dimension
        pos: (B, N, 3) where 3 is the spatial coordinates of the points
        """
        B, N, C = x.shape

        # Compute pairwise distance between points
        dist = torch.cdist(pos, pos)  # (B, N, N)
        masked_dist = dist * mask  # 如果只希望保留 mask 中为 1 的距离，其他距离为 0
        # 如果希望将屏蔽的位置设为无穷大：
        dist = dist.masked_fill(mask == 0, float('inf'))
        # Select the k nearest neighbors
        _, idx = dist.topk(self.k, dim=-1, largest=False)  # (B, N, k)

        # Query, key, and value projections
        q = self.fc_q(x)  # (B, N, out_channels)
        k = self.fc_k(x)  # (B, N, out_channels)
        v = self.fc_v(x)  # (B, N, out_channels)
        residual = q
        # Gather k-nearest neighbors
        k = torch.gather(k.unsqueeze(2).expand(-1, -1, self.k, -1), 1, idx.unsqueeze(3).expand(-1, -1, -1, k.shape[-1]))  # (B, N, k, out_channels)
        v = torch.gather(v.unsqueeze(2).expand(-1, -1, self.k, -1), 1, idx.unsqueeze(3).expand(-1, -1, -1, v.shape[-1]))  # (B, N, k, out_channels)
        q = q.unsqueeze(2).expand(-1, -1, self.k, -1)  # (B, N, k, out_channels)
        
        # Compute positional encoding for pairwise relative positions
        pos_relative = (pos.unsqueeze(2).expand(-1, -1, self.k, -1) - torch.gather(pos.unsqueeze(2).expand(-1, -1, self.k, -1), 1, idx.unsqueeze(3).expand(-1, -1, -1, 3)))  # (B, N, k, 3)
        pos_enc = self.relative_pos_to_polar(pos_relative, normal, idx)  # Polar coordinate encoding
        pos_enc = self.pos_encoding(pos_enc)

        k = k + pos_enc
        v = v + pos_enc
        

        # Compute attention weights
        #attn = torch.sum(q * k, dim=-1) / np.sqrt(k.shape[-1])  # (B, N, k)

        attn =  self.dropout(F.softmax(torch.matmul(q,k.permute(0,1,3,2)) / np.sqrt(k.shape[-1]), dim=-1))  # (B, N, k, out_channels)
        # Apply attention to the value features

        out = torch.sum(torch.matmul(attn , v), dim=2)/self.k  # (B, N, out_channels)

        # Final linear transformation
        out = self.dropout_out(self.fc_out(out))+residual  # (B, N, out_channels)

        out = self.layer_norm(out)
        
        return out
    def relative_pos_to_polar(self, pos_relative, normal, idx):
        """
        Convert relative positions to polar coordinates using normals as reference axes.
        pos_relative: (B, N, k, 3) relative positions in Cartesian coordinates
        normal: (B, N, 3) normals at each point
        idx: (B, N, k) indices of the k-nearest neighbors
        """
        B, N, k, _ = pos_relative.shape
        
        # Normalize the normals
        normal = F.normalize(normal, p=2, dim=-1)

        # Compute radial distance
        radial_dist = torch.norm(pos_relative, dim=-1, keepdim=True)  # (B, N, k, 1)

        # Project relative position onto the normal (dot product to find the component along the normal)
        pos_dot_normal = torch.sum(pos_relative * normal.unsqueeze(2).expand(-1, -1, k, -1), dim=-1, keepdim=True)

        # Compute angular distances (theta: angle with normal, phi: azimuth in xy-plane)
        theta = torch.acos(pos_dot_normal / (radial_dist + 1e-8))  # (B, N, k, 1), angle with normal
        #phi = torch.atan2(pos_relative[..., 1], pos_relative[..., 0])  # (B, N, k, 1), azimuth angle

        # Concatenate radial distance, theta, and phi to form the polar coordinate representation
        polar_coords = torch.cat([radial_dist, theta], dim=-1)  # (B, N, k, 3)

        return polar_coords

class PointTransformer(nn.Module):
    def __init__(self, out_dim, dim_in=3, embed_dim=64, k=16):
        super(PointTransformer, self).__init__()
        self.k = k  # Number of nearest neighbors to use
        
        # Input embedding
        self.fc_in = nn.Linear(dim_in, embed_dim)
        
        # Transformer layers
        self.transformer1 = PointTransformerLayer(embed_dim, embed_dim, k=self.k)
        self.transformer2 = PointTransformerLayer(embed_dim, embed_dim, k=self.k)
        self.transformer3 = PointTransformerLayer(embed_dim, embed_dim, k=self.k)

        # Output layer for classification
        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x,pos,normal,mask):
        """
        x: (B, N, C) where B is batch size, N is number of points, C is the input feature dimension (usually 3 for coordinates).
        """
        x = self.fc_in(x)  # Input embedding (B, N, embed_dim)

        # Pass through multiple Point Transformer layers
        x = self.transformer1(x, pos,normal,mask)
        x = self.transformer2(x, pos,normal,mask)
        x = self.transformer3(x, pos,normal,mask)

        # Global max pooling
        #x = torch.max(x, dim=1)[0]  # (B, embed_dim)

        # Final classification layer
        out = self.fc_out(x)  # (B, num_classes)

        return out




class Point_Unet_block(nn.Module):
    def __init__(self,input_dim=128,hidden_dim=256, output_dim=128, k_n = 16):
        super(Point_Unet_block, self).__init__()
        self. point_transformer = PointTransformer(out_dim=output_dim, dim_in = input_dim,embed_dim = hidden_dim, k=k_n)



    def forward(self, x,pos,normal, index,m,mask):

        # 假设输入特征为 b * n * f
        b, n, f = x.size()
        x = self.point_transformer(x,pos,normal,mask)
        index_expanded = index
       

        # 创建用于存储输出的聚类后的特征 tensor，形状为 b * m * f
        x_agg = torch.zeros(b, m, f).to('cuda')

        # 创建一个掩码，跳过 index 为 -1 的位置
        mask = (index >= 0).to('cuda')

        # 将 index 为 -1 的部分暂时设为 0，以便能够进行 scatter_add 操作（会被忽略）
        index_valid = index.masked_fill(~mask, 0).to('cuda')

        # 使用 scatter_add 聚合 n 个点到 m 个聚类中心，基于有效的 index 映射，掩码为 True 的地方被保留
        x_agg = x_agg.scatter_add(1, index_valid.unsqueeze(-1).expand(-1, -1, f), x * mask.unsqueeze(-1))

        # 使用 one-hot 编码计算每个聚类中心的点的数量
        # one-hot 编码生成，维度为 b * n * m
        # 使用 one-hot 编码计算每个聚类中心的点的数量，忽略 index 为 -1 的点
        one_hot = torch.zeros(b, n, m).to('cuda').scatter_(2, index_valid.unsqueeze(-1), mask.unsqueeze(-1).float())


        # 计算每个聚类中心的点的数量，维度为 b * m
        cluster_count = one_hot.sum(dim=1).unsqueeze(-1)  # b * m * 1
        
        # 避免除以 0，防止某些聚类没有任何点
        x_agg = x_agg / torch.clamp(cluster_count, min=1)
        
        return x_agg


class PointCloudMappingNN(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128,k_n =16):
        super(PointCloudMappingNN, self).__init__()
        self. point_transformer = PointTransformer(out_dim=output_dim, dim_in = input_dim,embed_dim = hidden_dim, k=k_n)

    def forward(self, x,pos,normal, index,feat,mask):
        # x: b * m * f, index: b * n
        # 使用 gather 方法根据 index 选择特征
        # 需要将 index 扩展以适应 x 的形状
        n = index.size(1)  # n 点的数量
        b, m, f = x.size()
        # 通过 index 选择特征，扩展 index 维度以适应 gather 操作

        index_valid = index.clone()  # 复制 index
        index_valid[index_valid == -1] = 0  # 将 -1 的位置设为 0
        index_expanded = index_valid.unsqueeze(-1)  # 形状变为 b * n * 1
        selected_features = x.gather(1, index_expanded.expand(-1, -1, f))  # b * n * f
        selected_features[index == -1] = 0
        if feat!=None:
            selected_features = torch.cat((selected_features,feat),dim = 2)
        selected_features = self.point_transformer(selected_features,pos,normal,mask)
        return selected_features
