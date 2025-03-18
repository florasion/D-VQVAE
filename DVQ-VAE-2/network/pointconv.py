"""
Classification Model
Author: Wenxuan Wu
Date: September 2019
"""
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from utils.pointconv_util import PointConvDensitySetAbstraction

class PointConvDensityClsSsg(nn.Module):
    def __init__(self):
        super(PointConvDensityClsSsg, self).__init__()
        self.sa5 = PointConvDensitySetAbstraction(npoint=3000, nsample=32, in_channel=1 + 3, mlp=[256, 128, 128], bandwidth =0.1, group_all=False,relu = False)
        self.sa6 = PointConvDensitySetAbstraction(npoint=3000, nsample=32, in_channel=128 + 3, mlp=[128, 64, 3], bandwidth = 0.1, group_all=False,relu = False)
        # self.sa1 = PointConvDensitySetAbstraction(npoint=3000, nsample=32, in_channel=feature_dim + 3, mlp=[64, 64, 128], bandwidth = 0.1, group_all=False,relu = True)
        # self.sa2 = PointConvDensitySetAbstraction(npoint=3000, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], bandwidth = 0.2, group_all=False,relu = True)
        # # self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], bandwidth = 0.4, group_all=True)
        # # self.sa4 = PointConvDensitySetAbstraction(npoint=128, nsample=1, in_channel=1024 + 3, mlp=[1024, 512, 256], bandwidth = 0.4, group_all=False)
        # self.sa5 = PointConvDensitySetAbstraction(npoint=3000, nsample=64, in_channel=256 + 3, mlp=[256, 128, 128], bandwidth = 0.2, group_all=False,relu = True)
        # self.sa6 = PointConvDensitySetAbstraction(npoint=3000, nsample=32, in_channel=128 + 3, mlp=[128, 64, 3], bandwidth = 0.1, group_all=False,relu = False)
        # self.fc1 = nn.Linear(1024, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.drop1 = nn.Dropout(0.7)
        # self.fc2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.drop2 = nn.Dropout(0.7)
        # self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz, feat):
        B, _, _ = xyz.shape
        # l1_xyz, l1_points = self.sa1(xyz, feat)
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l5_xyz, l5_points = self.sa5(xyz, feat)
        l6_xyz, l6_points = self.sa6(xyz, l5_points)
        # x = l3_points.view(B, 1024)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.fc3(x)
        # x = F.log_softmax(x, -1)
        return l6_points

if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,3,2048))
    label = torch.randn(8,16)
    feat = torch.randn(8,1,2048)
    model = PointConvDensityClsSsg()
    output= model(input,feat)
    print(output.size())

