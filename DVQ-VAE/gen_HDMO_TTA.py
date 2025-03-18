import numpy as np
import igl as igl
import time
import pickle
import time
import torch
import argparse
from torch.utils.data import DataLoader
from dataset.dataset_HMDO import FHAB_diversity
from network.affordanceNet_obman_mano_vertex import affordanceNet
from network.cmapnet_objhand import pointnet_reg
import numpy as np
import random
from utils import utils, utils_loss
import mano
import json
from utils.loss import TTT_loss
import trimesh
from metric.simulate import run_simulation
from network.gen_net import GenNet
from network.cmapnet_objhand import pointnet_reg
GenNet=GenNet()

device = 'cuda'
checkpoint = torch.load("ckpt_path", map_location=torch.device('cpu'))['network']
model_dict =  GenNet.state_dict()
state_dict = {k:v for k,v in checkpoint.items() if k in model_dict.keys()}
#print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
model_dict.update(state_dict)
GenNet.load_state_dict(model_dict)
#ARmodel = GatedPixelCNN(512, 512, args.n_layers).to(device)
print('loaded vqvae')
pix_checkpoint = torch.load("/LATENT_BLOCK_pixelcnn.pt", map_location=torch.device('cpu'))
#model_dict =  GenNet.state_dict()
#state_dict = {k:v for k,v in pix_checkpoint.items() if k in model_dict.keys()}
#print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
#model_dict.update(state_dict)
GenNet.GatedPixelCNN.load_state_dict(pix_checkpoint)
GenNet=GenNet.to(device)
print('loaded pixelcnn')
# dataset
cmap_model = pointnet_reg(with_rgb=False)  # ContactNet
checkpoint_cmap = torch.load('./checkpoints/model_cmap_best.pth', map_location=torch.device('cpu'))['network']
cmap_model.load_state_dict(checkpoint_cmap)
cmap_model = cmap_model.to(device)
print('load cmap')
dataset = FHAB_diversity()
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
print('load dataset')
# mano hand model
with torch.no_grad():
    rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                        model_type='mano',
                        use_pca=True,
                        num_pca_comps=45,
                        batch_size=1,
                        flat_hand_mean=True).to(device)
GenNet.set_rh_mano(rh_mano)
#GenNet.mask()
#rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).to(device)  # [1, 1538, 3], face indexes
with open("models/mano/closed_mano_faces.pkl", 'rb') as f:
    rh_faces = torch.tensor(pickle.load(f))
model = GenNet


model.eval()
rh_mano.eval()
total_penetr_vol=0.0
total_simu_disp=0.0
total_num = 0
total_contact =0
total_time = 0.0
total_high_q = 0
total_q = 0
vol_list = []
mesh_dist_list = []
for batch_idx, (obj_pc, origin_verts, origin_faces) in enumerate(dataloader):

    obj_xyz = obj_pc.permute(0,2,1)[:,:,:3].squeeze(0).cpu().numpy()  # [3000, 3]
    origin_verts = origin_verts.squeeze(0).numpy()  # [N, 3]
    recon_params, R_list, trans_list, r_list = [], [], [], []
    
    for i in range(10):
        # generate random rotation
        rot_angles = np.random.random(3) * np.pi * 2
        theta_x, theta_y, theta_z = rot_angles[0], rot_angles[1], rot_angles[2]
        Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
        Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
        rot = Rx @ Ry @ Rz  # [3, 3]
        # generate random translation
        trans = np.array([-0.0793, 0.0208, -0.6924]) + np.random.random(3) * 0.2 *0
        trans = trans.reshape((3, 1))
        R = np.hstack((rot, trans))  # [3, 4]
        obj_xyz_transformed = np.matmul(R[:3,0:3], obj_xyz.copy().T) + R[:3,3].reshape(-1,1)  # [3, 3000]
        obj_mesh_verts = (np.matmul(R[:3,0:3], origin_verts.copy().T) + R[:3,3].reshape(-1,1)).T  # [N, 3]
        obj_xyz_transformed = torch.tensor(obj_xyz_transformed, dtype=torch.float32)
        obj_pc_transformed = obj_pc.clone()
        obj_pc_transformed[0, :3, :] = obj_xyz_transformed  # [1, 4, N]
        obj_pc_TTT = obj_pc_transformed.detach().clone().to(device)
        hand_=torch.zeros([obj_pc_TTT.size()[0],3,778]).to(device)
        obj_pc=obj_pc.to(device)
        time_start=time.time()
        recon_hand ,recon_pos = model.gen(obj_pc_TTT)
        time_end = time.time()
        gen_time = time_end-time_start
        print("gen_time:",gen_time)
        if total_time == 0:
            total_time+=0.11
        else:
            total_time +=gen_time
        B=obj_pc.size()[0]
        recon_param = torch.zeros((B, 61)).to(device)
        recon_param[:, 0:10] = recon_hand[:, 0:10]
        recon_param[:, 10:13] = recon_pos[:, 0:3]
        recon_param[:, 13:58] = recon_hand[:, 10:55]
        recon_param[:, 58:61] = recon_pos[:, 3:6]
        recon_param = recon_param.detach()  # recon [1,61] mano params
        recon_param = torch.autograd.Variable(recon_param, requires_grad=True)
        optimizer = torch.optim.SGD([recon_param], lr=0.00000625, momentum=0.8)
        cam_extr = np.array([[1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., -1., 0.]]).astype(np.float32)
        obj_mesh_org = trimesh.Trimesh(vertices=obj_mesh_verts,
                                   faces=origin_faces.squeeze(0).cpu().numpy().astype(np.int32))  # obj
        final_mano = rh_mano(betas=recon_param[:, :10], global_orient=recon_param[:, 10:13],
                             hand_pose=recon_param[:, 13:58], transl=recon_param[:, 58:])
        final_mano_verts = final_mano.vertices.squeeze(0).detach().cpu().numpy()  # [778, 3]
        final_mano_verts = final_mano_verts.dot(cam_extr[:3,:3].T)
        try:
            hand_mesh_org = trimesh.Trimesh(vertices=final_mano_verts, faces=rh_faces.cpu().numpy().reshape((-1, 3)))
        except:
            continue

        for j in range(0,300):  # non-learning based optimization steps
            optimizer.zero_grad()

            recon_mano = rh_mano(betas=recon_param[:, :10], global_orient=recon_param[:, 10:13],
                                 hand_pose=recon_param[:, 13:58], transl=recon_param[:, 58:])
            recon_xyz = recon_mano.vertices.to(device)  # [B,778,3], hand vertices

            # calculate cmap from current hand
            obj_nn_dist_affordance, _ = utils_loss.get_NN(obj_pc_TTT.permute(0, 2, 1)[:, :, :3], recon_xyz)
            cmap_affordance = utils.get_pseudo_cmap(obj_nn_dist_affordance)  # [B,3000]
             # predict target cmap by ContactNet
            recon_cmap = cmap_model(obj_pc_TTT[:, :3, :], recon_xyz.permute(0, 2, 1).contiguous())  # [B,3000]
            recon_cmap = (recon_cmap / torch.max(recon_cmap, dim=1)[0]).detach()
            #print(rh_faces.size())
            penetr_loss, consistency_loss, contact_loss = TTT_loss(recon_xyz, rh_faces.unsqueeze(0),
                                                                   obj_pc_TTT[:, :3, :].permute(0,2,1).contiguous(),
                                                                   cmap_affordance, recon_cmap)
            loss = 1 * contact_loss  + 5 * penetr_loss
            loss.backward()
            optimizer.step()
            if j == 0 or j == 299:
                print("Object sample {}, pose {}, iter {}, "
                      "penetration loss {:9.5f}, "
                     "consistency loss {:9.5f}, "
                     "contact loss {:9.5f}".format(batch_idx, i, j,
                                                   penetr_loss.item(), consistency_loss.item(), contact_loss.item()))
        cam_extr = np.array([[1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., -1., 0.]]).astype(np.float32)
        obj_mesh_verts = obj_mesh_verts.dot(cam_extr[:3,:3].T)  # [N,3]
        obj_mesh = trimesh.Trimesh(vertices=obj_mesh_verts,
                                   faces=origin_faces.squeeze(0).cpu().numpy().astype(np.int32))  # obj
        final_mano = rh_mano(betas=recon_param[:, :10], global_orient=recon_param[:, 10:13],
                             hand_pose=recon_param[:, 13:58], transl=recon_param[:, 58:])
        final_mano_verts = final_mano.vertices.squeeze(0).detach().cpu().numpy()  # [778, 3]
        final_mano_verts = final_mano_verts.dot(cam_extr[:3,:3].T)
        try:
            hand_mesh = trimesh.Trimesh(vertices=final_mano_verts, faces=rh_faces.cpu().numpy().reshape((-1, 3)))
        except:
            continue
                
        obj_mesh.export("./vis_deform_300/"+'obj_org_{}_{}'.format(batch_idx,i)+".ply")
        hand_mesh_org.export("./vis_deform_300/"+'hand_org_{}_{}'.format(batch_idx,i)+".ply")
        obj_mesh.export("./vis_deform_300/"+'obj_{}_{}'.format(batch_idx,i)+".ply")
        hand_mesh.export("./vis_deform_300/"+'hand_{}_{}'.format(batch_idx,i)+".ply")

