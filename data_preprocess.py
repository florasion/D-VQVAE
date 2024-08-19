from torch.utils.data import Dataset
import torch
import os
import pickle
from torchvision import transforms
import numpy as np
from utils import utils
import time
from PIL import Image
import json
import mano

'''process the labels, save into single npy file'''
'''obj_pc, mano_param, obj_cmap'''

###change the path to your own path
torch.cuda.set_device(1)
mode = 'test'
file_path = '/data/ObMan/obman/{}.txt'.format(mode)
mano_trans_path = '/data/ObMan/mano-fit/{}/mano_trans.json'.format(mode)
mano_rot_path = '/data/ObMan/mano-fit/{}/global_rot.json'.format(mode)
obj_mesh_path = '/data/ObMan/obman/{}/models_resampled'.format(mode)
img_root='/data/ObMan/obman'
img_list = utils.readTxt_obman(file_path)
mano_trans = json.load(open(mano_trans_path, 'r'))
mano_rot = json.load(open(mano_rot_path, 'r'))
dataset_size = len(img_list)

obj_pc_list, hand_param_list ,hand_v_list=[], [], []


with torch.no_grad():
    rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                              model_type='mano',
                              use_pca=True,
                              num_pca_comps=45,
                              batch_size=1,
                              flat_hand_mean=True).to('cpu')


for idx in range(dataset_size):
    line = img_list[idx].strip()
    img_path = os.path.join(img_root, mode, 'rgb_obj', line)
    #print(img_path)
    meta_path = img_path.replace('rgb_obj', 'meta').replace('jpg', 'pkl')
    #print(meta_path)
    meta = pickle.load(open(meta_path, 'rb'))
    print(idx)
    # hand mano param
    hand_pose = torch.tensor(meta['hand_pose'])  # [45]
    hand_shape = torch.tensor(meta['shape'])  # [10]
    hand_trans = torch.tensor(mano_trans[idx])
    hand_rot = torch.tensor(mano_rot[idx])  # [3]
    hand_param = torch.cat((hand_shape, hand_rot, hand_pose, hand_trans), dim=0)  # [61]
    hand_param_list.append(hand_param)
    hand_param=hand_param.unsqueeze(0)
    #print(hand_param.size())
    hand_v = rh_mano(betas=hand_param[:, :10], global_orient=hand_param[:, 10:13],
                             hand_pose=hand_param[:, 13:58], transl=hand_param[:, 58:])
    hand_v=hand_v.vertices.detach().to('cpu')
    hand_v_list.append(hand_v)
    # obj xyz
    
    obj_model_path = os.path.join(obj_mesh_path, str(idx) + '.npy')
    obj_xyz_normalized = np.load(obj_model_path)  # [10000, 3]
    obj_xyz_normalized = obj_xyz_normalized[:3000, :]  # [3000, 3]
    obj_pose = np.array((meta['affine_transform']))
    obj_xyz_transformed = utils.vertices_transformation(obj_xyz_normalized, obj_pose)
    obj_xyz_transformed = torch.tensor(obj_xyz_transformed, dtype=torch.float32)
    obj_scale = meta['obj_scale']
    obj_scale_tensor = torch.tensor(obj_scale).type_as(obj_xyz_transformed).repeat(3000, 1)  # [3000, 1]
    obj_pc = torch.cat((obj_xyz_transformed, obj_scale_tensor), dim=-1)  # [3000, 4]
    obj_pc = obj_pc.permute(1, 0)  # [4, 3000]
    obj_pc_list.append(obj_pc)
    
    # object contact map from contactdb model
    #obj_cmap_path = obj_model_path.replace('models_resampled', 'cmap_contactdb')
    #obj_cmap = np.load(obj_cmap_path).T   # [10, N] -> [N, 10]
    #obj_cmap = torch.tensor(obj_cmap[:3000, :])  # [3000, 10]
    #obj_cmap = obj_cmap > 0  # bool
    #obj_cmap_list.append(obj_cmap)


# save obj pc
#obj_pc_tensor = torch.stack(obj_pc_list, dim=0)  # [S, 4, 3000]
#obj_pc_tensor = obj_pc_tensor.cpu().numpy()
#pc_save_path = '/data/zz/ObMan/obman/processed/objz_pc_{}.npz'.format(mode)
#np.savez(pc_save_path, obj_pc_tensor)

# save mano param
#hand_param_tensor = torch.stack(hand_param_list, dim=0)  # [S, 61]
#hand_param_tensor = hand_param_tensor.cpu().numpy()
#param_save_path = '/data/zz/ObMan/obman/processed/handz_param_{}.npz'.format(mode)
#np.savez(param_save_path, hand_param_tensor)



hand_param_tensor = torch.stack(hand_v_list, dim=0)  # [S, 61]
hand_param_tensor = hand_param_tensor.cpu().numpy()
param_save_path = '/data/zz/ObMan/obman/processed/hand_v_{}.npy'.format(mode)
np.save(param_save_path, hand_param_tensor)

# save contactdb cmap
#obj_cmap_tensor = torch.stack(obj_cmap_list, dim=0)  # [S, 3000, 10]
#obj_cmap_tensor = obj_cmap_tensor.cpu().numpy()
#cmap_save_path = '/hand-object-3/download/dataset/ObMan/obman/processed/obj_cmap_contactdb_{}.npy'.format(mode)
#np.save(cmap_save_path, obj_cmap_tensor)


