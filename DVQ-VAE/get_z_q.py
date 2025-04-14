from torch.utils.data import DataLoader
from dataset.dataset_obman_mano import obman
import torch
import argparse
import numpy as np
import mano
import json
from network.DVQVAE import DVQVAE

import mano
from utils import utils_loss


def eval(args, model, eval_loader, device, rh_mano):
    # validation
    emb_map=[]
    model.eval()
    emb_used=torch.zeros(7,2048)
    with torch.no_grad():
        for batch_idx, (obj_pc, hand_xyz_gt, idx) in enumerate(eval_loader):
            obj_pc, hand_xyz_gt= obj_pc.to(device), hand_xyz_gt.to(device)
            emb_idx , obj_emb= model(obj_pc, hand_xyz_gt)
            for i in range(7):
                emb_used[i][emb_idx[i].to('cpu')]+=1
            
            emb_idx=emb_idx.unsqueeze(1)
            if False:
                re=torch.cat((emb_idx[0],emb_idx[1],emb_idx[2],emb_idx[0],emb_idx[3],emb_idx[4],emb_idx[0],emb_idx[5],emb_idx[6]),0)
                reshaped_tensor = re.view(1, 3, 3)
            else:
                reshaped_tensor=torch.cat((emb_idx[0],emb_idx[1],emb_idx[2],emb_idx[3],emb_idx[4],emb_idx[5],emb_idx[6]),1)
            emb_map.append(reshaped_tensor)
            emb_used_out = f'./emb_used.npy'
            np.save(emb_used_out, emb_used)
        emb_map = torch.cat(emb_map, dim=0)
        emb_map=emb_map.cpu()
        x_out = f'./latent_e_indices.npy'
        np.save(x_out, emb_map)
        emb_used_out = f'./emb_used.npy'
        np.save(emb_used_out, emb_used)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--use_cuda", type=int, default=1)

    args = parser.parse_args()


    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    checkpoint = torch.load("your_DVQVAE_ckpt", map_location=torch.device('cpu'))['network']
    model = DVQVAE(
            obj_inchannel=4).to(device)
    model_dict =  model.state_dict()
    state_dict = {k:v for k,v in checkpoint.items() if k in model_dict.keys()}

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model = model.to(device)
    dataset = obman(mode="train", vis=True)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
    with torch.no_grad():
        rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                            model_type='mano',
                            use_pca=True,
                            num_pca_comps=45,
                            batch_size=1,
                            flat_hand_mean=True).to(device)
    rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).to(device)  
    eval(args, model, dataloader, device, rh_mano)
