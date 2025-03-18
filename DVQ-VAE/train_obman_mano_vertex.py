import os
import time
import torch
import argparse
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from collections import defaultdict
from dataset.dataset_obman_mano2 import obman
from network.DVQVAE import DVQVAE
import numpy as np
import random
from utils import utils_loss
from utils.loss import   CMap_loss3, inter_penetr_loss, CMap_consistency_loss 
from pytorch3d.loss import chamfer_distance
import mano
from torch.utils.tensorboard import SummaryWriter

def random_rotation_matrix(batch_size, device):
    theta = torch.rand(batch_size, device=device) * 2 * np.pi
    phi = torch.rand(batch_size, device=device) * np.pi
    psi = torch.rand(batch_size, device=device) * 2 * np.pi

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_psi = torch.cos(psi)
    sin_psi = torch.sin(psi)

    rotation_matrix = torch.stack([
        cos_theta * cos_psi - sin_theta * cos_phi * sin_psi,
        -cos_theta * sin_psi - sin_theta * cos_phi * cos_psi,
        sin_theta * sin_phi,
        sin_theta * cos_psi + cos_theta * cos_phi * sin_psi,
        -sin_theta * sin_psi + cos_theta * cos_phi * cos_psi,
        -cos_theta * sin_phi,
        sin_phi * sin_psi,
        sin_phi * cos_psi,
        cos_phi
    ], dim=1).view(batch_size, 3, 3)

    return rotation_matrix


def random_rotate_point_cloud(point_cloud_1,point_cloud_2):
    B, pointnum, _ = point_cloud_1.shape
    rotation_matrices = random_rotation_matrix(B, point_cloud_1.device)

    point_cloud_1 = point_cloud_1
    #print(point_cloud_1.size())
    #print(rotation_matrices.size())
    point_cloud_2 = point_cloud_2

    rotated_point_cloud_1 = torch.matmul(rotation_matrices.float(), point_cloud_1.float()).squeeze(-1)
    rotated_point_cloud_2 = torch.matmul(rotation_matrices.float(), point_cloud_2.float()).squeeze(-1)
    return rotated_point_cloud_1 , rotated_point_cloud_2


def train(args, epoch, model, train_loader, device, optimizer, log_root,checkpoint_root, rh_mano, rh_faces):
    since = time.time()
    logs = defaultdict(list)
    a, b, c, d, e ,f= args.loss_weight
    model.train()
    for batch_idx, (obj_pc, hand_param,idxx) in enumerate(train_loader):
        
        obj_pc, hand_param = obj_pc.to(device), hand_param.to(device)
        B=obj_pc.size()[0]


        gt_mano = rh_mano(betas=hand_param[:, :10], global_orient=hand_param[:, 10:13],
                          hand_pose=hand_param[:, 13:58], transl=hand_param[:, 58:])
        hand_xyz = gt_mano.vertices.to(device)  # [B,778,3]


        optimizer.zero_grad()
        recon_hand ,recon_pos, embedding_loss , perplexity = model(obj_pc, hand_xyz.permute(0,2,1))  # recon [B,61] mano params
        #print(recon_param)
        recon_param = torch.zeros((B, 61)).to(device)
        recon_param[:, 0:10] = recon_hand[:, 0:10]
        recon_param[:, 10:13] = recon_pos[:, 0:3]
        recon_param[:, 13:58] = recon_hand[:, 10:55]
        recon_param[:, 58:61] = recon_pos[:, 3:6]
        recon_mano = rh_mano(betas=recon_param[:, :10], global_orient=recon_param[:, 10:13],
                             hand_pose=recon_param[:, 13:58], transl=recon_param[:, 58:])
        recon_xyz = recon_mano.vertices.to(device)  # [B,778,3]
        # obj xyz NN dist and idx
        obj_nn_dist_gt, obj_nn_idx_gt = utils_loss.get_NN(obj_pc.permute(0,2,1)[:,:,:3], hand_xyz)

        recon_loss, _ = chamfer_distance(recon_xyz, hand_xyz, point_reduction='sum', batch_reduction='mean')
        obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(obj_pc.permute(0, 2, 1)[:, :, :3], recon_xyz)
        param_loss = torch.nn.functional.mse_loss(recon_param, hand_param, reduction='none').sum() / recon_param.size(0)

        cmap_loss = CMap_loss3(obj_pc.permute(0,2,1)[:,:,:3], recon_xyz, obj_nn_dist_recon < 0.01**2)

        consistency_loss = CMap_consistency_loss(obj_pc.permute(0,2,1)[:,:,:3], recon_xyz, hand_xyz,
                                                 obj_nn_dist_recon, obj_nn_dist_gt)
        penetr_loss = inter_penetr_loss(recon_xyz, rh_faces, obj_pc.permute(0,2,1)[:,:,:3],
                                        obj_nn_dist_recon, obj_nn_idx_recon)

        if epoch >= 5:
            loss =a* param_loss + b * embedding_loss + c* recon_loss + d * penetr_loss + e * cmap_loss +f *consistency_loss 
        else:
            loss =a * param_loss + b * embedding_loss + c * recon_loss + d * penetr_loss + f * consistency_loss 
        loss.backward()
        optimizer.step()
        logs['loss'].append(loss.item())
        logs['param_loss'].append(param_loss.item())
        logs['recon_loss'].append(recon_loss.item())
        logs['embedding_loss'].append(embedding_loss.item())
        logs['cmap_loss'].append(cmap_loss.item())
        #logs['cmap_loss_hand'].append(cmap_loss_hand.item())
        logs['penetr_loss'].append(penetr_loss.item())
        logs['cmap_consistency'].append(consistency_loss.item())
        if batch_idx % args.print_every == 0 or batch_idx == len(train_loader) - 1:
            print("Train Epoch {:02d}/{:02d}, Batch {:04d}/{:d}, Total Loss {:9.5f}, Param {:9.5f},recon_loss {:9.5f},penetr_loss {:9.5f},embedding_loss {:9.5f},consistency_loss {:9.5f},cmap_loss {:9.5f}".format(
                    epoch, args.epochs, batch_idx, len(train_loader) - 1, loss.item(),
                     param_loss.item(),recon_loss.item(),penetr_loss.item(),embedding_loss.item(),consistency_loss.item(),cmap_loss.item()))

    time_elapsed = time.time() - since
    out_str = "Epoch: {:02d}/{:02d}, train, time {:.0f}m, Mean Toal Loss {:9.5f},  recon_loss {:9.5f},embedding {:9.5f},  Param {:9.5f},  cmap_loss {:9.5f}, cmap_consistency {:9.5f},  penetr_loss {:9.5f}".format(
        epoch, args.epochs, time_elapsed // 60,
        sum(logs['loss']) / len(logs['loss']),
        sum(logs['recon_loss']) / len(logs['recon_loss']),
        sum(logs['embedding_loss']) / len(logs['embedding_loss']),
        sum(logs['param_loss']) / len(logs['param_loss']),
        sum(logs['cmap_loss']) / len(logs['cmap_loss']),
        sum(logs['cmap_consistency']) / len(logs['cmap_consistency']),
        sum(logs['penetr_loss']) / len(logs['penetr_loss']),
    )
    with open(log_root, 'a') as f:
        f.write(out_str+'\n')
    if epoch%10==0 and args.train_mode != 'Test':
        save_name = os.path.join(checkpoint_root, 'model_ {:02d}.pth'.format(epoch))
        torch.save({
            'network': model.state_dict(),
            'epoch': epoch
        }, save_name)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''experiment setting'''
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--use_cuda", type=int, default=1)
    parser.add_argument("--dataloader_workers", type=int, default=8)
    parser.add_argument("--train_mode", type=str, default='TrainTest')
    parser.add_argument("--loss_weight", type=list, default=[0.1, 10, 1, 5, 1000 , 10 ])

    args = parser.parse_args()
    # loss =a* param_loss + b * embedding_loss + c* recon_loss +d* penetr_loss + e * cmap_loss +f *consistency_loss   
    # log file
    local_time = time.localtime(time.time())
    time_str = str(local_time[1]) + '_' + str(local_time[2]) + '_' + str(local_time[3])
    model_root = os.path.join('model_path', args.model_type)
    model_info = '{}model_info'.format(str(args.loss_weight))
    save_root = os.path.join(model_root, time_str + '_' + model_info)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    log_root = save_root + '/log.txt'
    log_file = open(log_root, 'w+')
    log_file.write(str(args) + '\n')
    log_file.write('weights are {}'.format(str(args.loss_weight)) + '\n')
    log_file.close()

    # seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # device
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print("using device", device)
    device_num = 1

    # network
    model = affordanceNet(
        obj_inchannel=args.obj_inchannel).to(device)

    #checkpoint = torch.load("/home/zhaozhe/Pycode/VQVAEtmp/logs2/baseline_mano/9_7/model_ 200_test.pth", map_location=torch.device('cpu'))['network']
    #model.load_state_dict(checkpoint)
    # multi-gpu
    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = True
        device_ids = range(torch.cuda.device_count())
        print("using {} cuda".format(len(device_ids)))
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model)
            device_num = len(device_ids)

    # dataset
    if 'Train' in args.train_mode:
        train_dataset = obman(mode="train", batch_size=args.batch_size)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.dataloader_workers)

    # optimizer
    start_epoch=1

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(args.epochs * x) for x in [0.3, 0.6, 0.8, 0.9]], gamma=0.5)

    # mano hand model
    with torch.no_grad():
        rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                              model_type='mano',
                              use_pca=True,
                              num_pca_comps=45,
                              batch_size=args.batch_size,
                              flat_hand_mean=True).to(device)
    rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).contiguous() # [1, 1538, 3], face triangle indexes
    rh_faces = rh_faces.repeat(args.batch_size, 1, 1).to(device) # [N, 1538, 3]
    model.set_rh_mano(rh_mano)
    best_val_loss = float('inf')
    best_eval_loss = float('inf')
    for epoch in range(start_epoch, args.epochs+1):
        if 'Train' in args.train_mode:
            train(args, epoch, model, train_loader, device, optimizer, log_root, save_root,rh_mano, rh_faces)
            scheduler.step()
