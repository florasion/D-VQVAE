import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F



class Contra_Net(nn.Module):
    def __init__(self):
        super(Contra_Net, self).__init__()
        self.relu = nn.ReLU()
        self.encoder = Encoder([10,256, 512], 45)
        self.decoder = Encoder([45, 256, 512], 45)



    def forward(self, joint,param):
        B=joint.size(0)
        joint_0 = joint[:, 0, :]
        joint_1 = joint[:, 1, :]
        joint_2 = joint[:, 2, :]    
        joint_3 = joint[:, 3, :]
        joint_4 = joint[:, 4, :]
        joint_5 = joint[:, 5, :]
        joint_6 = joint[:, 6, :]
        joint_7 = joint[:, 7, :]
        joint_8 = joint[:, 8, :]
        joint_9 = joint[:, 9, :]
        joint_10 = joint[:, 10, :]
        joint_11 = joint[:, 11, :]
        joint_12 = joint[:, 12, :]
        joint_13 = joint[:, 13, :]
        joint_14 = joint[:, 14, :]
        joint_15 = joint[:, 15, :]
        dot_product =  torch.zeros((B, 10)).to('cuda') 

        n_0 = joint_1-joint_0
        n_1 = joint_2-joint_1
        n_2 = joint_3-joint_2
        n_3 = joint_4-joint_0
        n_4 = joint_5-joint_4
        n_5 = joint_6-joint_5
        n_6 = joint_7-joint_0
        n_7 = joint_8-joint_7
        n_8 = joint_9-joint_8
        n_9 = joint_10-joint_0
        n_10 = joint_11-joint_10
        n_11 = joint_12-joint_11
        n_12 = joint_13-joint_0
        n_13 = joint_14-joint_13
        n_14 = joint_15-joint_14

        norm = torch.zeros((B, 10)).to('cuda') 
        dot_product[:,0]=(n_0 * n_1).sum(dim=1)
        norm[:,0]=n_0.norm(dim=1)*n_1.norm(dim=1)
        dot_product[:,1]=(n_1 * n_2).sum(dim=1)
        norm[:,1]=n_1.norm(dim=1)*n_2.norm(dim=1)
        dot_product[:,2]=(n_3 * n_4).sum(dim=1)
        norm[:,2]=n_3.norm(dim=1)*n_4.norm(dim=1)
        dot_product[:,3]=(n_4 * n_5).sum(dim=1)
        norm[:,3]=n_4.norm(dim=1)*n_5.norm(dim=1)
        dot_product[:,4]=(n_6 * n_7).sum(dim=1)
        norm[:,4]=n_6.norm(dim=1)*n_7.norm(dim=1)
        dot_product[:,5]=(n_7 * n_8).sum(dim=1)
        norm[:,5]=n_7.norm(dim=1)*n_8.norm(dim=1)
        dot_product[:,6]=(n_9 * n_10).sum(dim=1)
        norm[:,6]=n_9.norm(dim=1)*n_10.norm(dim=1)
        dot_product[:,7]=(n_10 * n_11).sum(dim=1)
        norm[:,7]=n_10.norm(dim=1)*n_11.norm(dim=1)
        dot_product[:,8]=(n_12 * n_13).sum(dim=1)
        norm[:,8]=n_12.norm(dim=1)*n_13.norm(dim=1)
        dot_product[:,9]=(n_13 * n_14).sum(dim=1)
        norm[:,9]=n_13.norm(dim=1)*n_14.norm(dim=1)

        cosine_similarity = dot_product / norm
        angles_rad = torch.acos(cosine_similarity)

        angles_deg = torch.rad2deg(angles_rad)
        label = self.relu(self.encoder(angles_deg))
        param_de = self.decoder(param)
        torch.cuda.empty_cache()
        return  param+param_de*label,label


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()
        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        #print('encoder', self.MLP)

    def forward(self, x):

        #print('x size before MLP {}'.format(x.size()))
        x = self.MLP(x)
        #print('x size after MLP {}'.format(x.size()))
        means = self.linear_means(x)
        #print('mean size {}, log_var size {}'.format(means.size(), log_vars.size()))
        return means