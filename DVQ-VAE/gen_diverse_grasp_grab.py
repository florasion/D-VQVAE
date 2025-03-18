import numpy as np
import igl as igl
import time
import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
from dataset.dataset_grab import GRAB_diversity
from network.DVQVAE import DVQVAE
import numpy as np
import random
from utils import utils, utils_loss
import mano
import json
from utils.loss import TTT_loss
import trimesh
from metric.simulate import run_simulation
from network.gen_net import GenNet


def uniform_box_sampling(min_corner, max_corner, res = 0.005):
    x_min = min_corner[0] - res
    x_max = max_corner[0] + res
    y_min = min_corner[1] - res
    y_max = max_corner[1] + res
    z_min = min_corner[2] - res
    z_max = max_corner[2] + res

    h = int((x_max-x_min)/res)+1
    l = int((y_max-y_min)/res)+1
    w = int((z_max-z_min)/res)+1

    # print('Sampling size: %d x %d x %d'%(h, l, w))

    with torch.no_grad():
        xyz = x = torch.zeros(h, l, w, 3, dtype=torch.float32) + torch.tensor([x_min, y_min, z_min], dtype=torch.float32)
        for i in range(1,h):
            xyz[i,0,0] = xyz[i-1,0,0] + torch.tensor([res,0,0])
        for i in range(1,l):
            xyz[:,i,0] = xyz[:,i-1,0] + torch.tensor([0,res,0])
        for i in range(1,w):
            xyz[:,:,i] = xyz[:,:,i-1] + torch.tensor([0,0,res])
    return res, xyz



def bounding_box_intersection(min_corner0, max_corner0, min_corner1, max_corner1):
    min_x = max(min_corner0[0], min_corner1[0])
    min_y = max(min_corner0[1], min_corner1[1])
    min_z = max(min_corner0[2], min_corner1[2])

    max_x = min(max_corner0[0], max_corner1[0])
    max_y = min(max_corner0[1], max_corner1[1])
    max_z = min(max_corner0[2], max_corner1[2])

    if max_x > min_x and max_y > min_y and max_z > min_z:
        # print('Intersected bounding box size: %f x %f x %f'%(max_x - min_x, max_y - min_y, max_z - min_z))
        return np.array([min_x, min_y, min_z]), np.array([max_x, max_y, max_z])
    else:
        return np.zeros((1,3), dtype = np.float32), np.zeros((1,3), dtype = np.float32)

def writeOff(output, vertex, face):
    with open(output, 'w') as f:
        f.write("COFF\n")
        f.write("%d %d 0\n" %(vertex.shape[0], face.shape[0]))
        for row in range(0, vertex.shape[0]):
            f.write("%f %f %f\n" %(vertex[row, 0], vertex[row, 1], vertex[row, 2]))
        for row in range(0, face.shape[0]):
            f.write("3 %d %d %d\n" %(face[row, 0], face[row, 1], face[row, 2]))

def intersection_eval(mesh0, mesh1, res=0.005, scale=1., trans=None, visualize_flag=False, visualize_file='output.off'):
    '''Calculate intersection depth and volumn of the two inputs meshes.
    args:
        mesh1, mesh2 (Trimesh.trimesh): input meshes
        res (float): voxel resolustion in meter(m)
        scale (float): scaling factor
        trans (float) (1, 3): translation
    returns:
        volume (float): intersection volume in cm^3
        mesh_mesh_dist (float): maximum depth from the center of voxel to the surface of another mesh
    '''
    # mesh0 = trimesh.load(mesh_file_0, process=False)
    # mesh1 = trimesh.load(mesh_file_1, process=False)

    # scale = 1 # 10
    # res = 0.5
    mesh0.vertices = mesh0.vertices * scale
    mesh1.vertices = mesh1.vertices * scale

    S, I, C = igl.signed_distance(mesh0.vertices + 1e-10, mesh1.vertices, mesh1.faces, return_normals=False)

    mesh_mesh_distance = S.min()
    # print("dist", S)
    # print("Mesh to mesh distance: %f cm" % mesh_mesh_distance)

    #### print("Mesh to mesh distance: %f" % (max(S.min(), 0)))

    if mesh_mesh_distance > 0:
        # print('No intersection!')
        return 0, mesh_mesh_distance

    # Get bounding box for each mesh:
    min_corner0 = np.array([mesh0.vertices[:,0].min(), mesh0.vertices[:,1].min(), mesh0.vertices[:,2].min()])
    max_corner0 = np.array([mesh0.vertices[:,0].max(), mesh0.vertices[:,1].max(), mesh0.vertices[:,2].max()])

    min_corner1 = np.array([mesh1.vertices[:,0].min(), mesh1.vertices[:,1].min(), mesh1.vertices[:,2].min()])
    max_corner1 = np.array([mesh1.vertices[:,0].max(), mesh1.vertices[:,1].max(), mesh1.vertices[:,2].max()])

    # Compute the intersection of two bounding boxes:
    min_corner_i, max_corner_i = bounding_box_intersection(min_corner0, max_corner0, min_corner1, max_corner1)
    if ((min_corner_i - max_corner_i)**2).sum() == 0:
        # print('No intersection!')
        return 0, mesh_mesh_distance

    # Uniformly sample the intersection bounding box:
    _, xyz = uniform_box_sampling(min_corner_i, max_corner_i, res)
    xyz = xyz.view(-1, 3)
    xyz = xyz.detach().cpu().numpy()

    S, I, C = igl.signed_distance(xyz, mesh0.vertices, mesh0.faces, return_normals=False)

    inside_sample_index = np.argwhere(S < 0.0)
    # print("inside sample index", inside_sample_index, len(inside_sample_index))

    # Compute the signed distance for inside_samples to mesh 1:
    inside_samples = xyz[inside_sample_index[:,0], :]

    S, I, C = igl.signed_distance(inside_samples, mesh1.vertices, mesh1.faces, return_normals=False)

    inside_both_sample_index = np.argwhere(S < 0)

    # Compute intersection volume:
    i_v = inside_both_sample_index.shape[0] * (res**3)
    # print("Intersected volume: %f cm^3" % (i_v))

    # Visualize intersection volume:
    if visualize_flag:
        writeOff(visualize_file, inside_samples[inside_both_sample_index[:,0], :], np.zeros((0,3)))

    # From (m) to (cm)
    return i_v * 1e6, mesh_mesh_distance * 1e2

def seal(mesh_to_seal):
    '''
    Seal MANO hand wrist to make it wathertight.
    An average of wrist vertices is added along with its faces to other wrist vertices.
    '''
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
    center = (mesh_to_seal.vertices[circle_v_id, :]).mean(0)

    mesh_to_seal.vertices = np.vstack([mesh_to_seal.vertices, center])
    center_v_id = mesh_to_seal.vertices.shape[0] - 1

    # pylint: disable=unsubscriptable-object # pylint/issues/3139
    for i in range(circle_v_id.shape[0]):
        new_faces = [circle_v_id[i-1], circle_v_id[i], center_v_id] 
        mesh_to_seal.faces = np.vstack([mesh_to_seal.faces, new_faces])
    return mesh_to_seal



def intersect_vox(obj_mesh, hand_mesh, pitch=0.5):
    '''
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    '''
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume

def mesh_vert_int_exts(obj1_mesh, obj2_verts):
    inside = obj1_mesh.ray.contains_points(obj2_verts)
    sign = (inside.astype(int) * 2) - 1
    return sign

def fx(x, y,a):
    return (x*a + (1-a)*y)

def main(args, model, eval_loader, device, rh_mano, rh_faces):
    '''
    Generate diverse grasps for object index with args.obj_id in out-of-domain HO3D object models
    '''
    model.eval()
    rh_mano.eval()
    total_penetr_vol=0.0
    total_simu_disp=0.0
    total_num = 0
    total_contact =0
    total_q = 0
    total_time = 0.0
    vol_list = []
    mesh_dist_list = []
    for batch_idx, (obj_pc, origin_verts, origin_faces) in enumerate(eval_loader):
        obj_xyz = obj_pc.permute(0,2,1)[:,:,:3].squeeze(0).cpu().numpy()  # [3000, 3]
        origin_verts = origin_verts.squeeze(0).numpy()  # [N, 3]
        recon_params, R_list, trans_list, r_list = [], [], [], []
        
        for i in range(20):
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
            # evaluate grasp
            cam_extr = np.array([[1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., -1., 0.]]).astype(np.float32)
            obj_mesh_verts = obj_mesh_verts.dot(cam_extr[:3,:3].T)  # [N,3]
            obj_mesh = trimesh.Trimesh(vertices=obj_mesh_verts,
                                       faces=origin_faces.squeeze(0).cpu().numpy().astype(np.int32))  # obj
            #print(obj_mesh_verts.shape)
            final_mano = rh_mano(betas=recon_param[:, :10], global_orient=recon_param[:, 10:13],
                                 hand_pose=recon_param[:, 13:58], transl=recon_param[:, 58:])
            final_mano_verts = final_mano.vertices.squeeze(0).detach().cpu().numpy()  # [778, 3]
            final_mano_verts = final_mano_verts.dot(cam_extr[:3,:3].T)
            try:
                hand_mesh = trimesh.Trimesh(vertices=final_mano_verts, faces=rh_faces.cpu().numpy().reshape((-1, 3)))
            except:
                continue
            # penetration volume
            
            hand = seal(hand_mesh)
            object=obj_mesh
            trimesh.repair.fix_normals(object)
            object = trimesh.convex.convex_hull(object)
            vol, mesh_dist = intersection_eval(hand, object, res=0.001, visualize_flag=True)
            vol_list.append(vol)
            mesh_dist_list.append(mesh_dist)
             
            penetr_vol=vol
            print(" vol cm3: ", penetr_vol)
            print(" inter dist cm: ", np.mean(mesh_dist_list))
            if  penetr_vol < 1e-8:
                sample_contact = False
            else:
                sample_contact = True
            # contact
            penetration_tol = 0.005
            # simulation displacement
            ##change the path to your vhacd path
            vhacd_exe = "/home/v-hacd-master/TestVHACD"
            try:
                simu_disp = run_simulation(final_mano_verts, rh_faces.reshape((-1, 3)),
                                          obj_mesh_verts, origin_faces.cpu().numpy().astype(np.int32).reshape((-1, 3)),
                                          vhacd_exe=vhacd_exe, sample_idx=i)
                #print('run success')
            except:
                simu_disp = 0.10
            print('generate id: {}, penetr vol: {}, simu disp: {}, contact: {},saved num: {}'
                  .format(i, penetr_vol, simu_disp, sample_contact,len(r_list) ))
            total_penetr_vol+=penetr_vol
            total_simu_disp+=simu_disp
            total_num+=1
            if sample_contact:
                total_contact+=1

            total_q += fx(penetr_vol,simu_disp*100,0.301)
            print('sample num:{}, mean penetr vol:{}, mean simu disp:{}, contact ratio:{},  q:{}\n'
                .format(total_num,total_penetr_vol/total_num,total_simu_disp/total_num,total_contact/total_num,total_q/total_num))
            print('generate id {} saved'.format(i))
            recon_params.append(recon_param.detach().cpu().numpy().tolist())
            R_list.append(R.tolist())
            trans_list.append(trans.tolist())
            r_list.append(np.array([theta_x, theta_y, theta_z]).tolist())

            if len(r_list) == args.num_grasp:
                break

        save_path = './diverse_grasp/grab/obj_id_{}.json'.format(batch_idx)
        print(save_path)
        data = {
            'recon_params': recon_params,
            'R_list': R_list,
            'trans_list': trans_list,
            'r_list': r_list
        }
        with open(save_path, 'w') as f:
            json.dump(data, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''experiment setting'''
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_cuda", type=int, default=1)
    parser.add_argument("--dataloader_workers", type=int, default=32)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[1024, 512, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[1024, 256, 61])
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--obj_inchannel", type=int, default=4)
    parser.add_argument("--condition_size", type=int, default=1024)
    parser.add_argument("--num_grasp", type=int, default=20)  # number of grasps you want to generate
    args = parser.parse_args()

    # device
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda")
    print("using device", device)

    # network
    GenNet=GenNet()

    checkpoint = torch.load("./checkpoints/model_best.pth", map_location=torch.device('cpu'))['network']
    model_dict =  GenNet.state_dict()
    state_dict = {k:v for k,v in checkpoint.items() if k in model_dict.keys()}

    model_dict.update(state_dict)
    GenNet.load_state_dict(model_dict)

    print('loaded vqvae')
    pix_checkpoint = torch.load("./checkpoints/LATENT_BLOCK_pixelcnn.pt", map_location=torch.device('cpu'))
    #model_dict =  GenNet.state_dict()
    #state_dict = {k:v for k,v in pix_checkpoint.items() if k in model_dict.keys()}
    #print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
    #model_dict.update(state_dict)
    GenNet.GatedPixelCNN.load_state_dict(pix_checkpoint)
    GenNet=GenNet.to(device)
    print('loaded pixelcnn')
    # dataset
    dataset = GRAB_diversity()
    print('load data')
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
    # mano hand model
    with torch.no_grad():
        rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                            model_type='mano',
                            use_pca=True,
                            num_pca_comps=45,
                            batch_size=1,
                            flat_hand_mean=True).to(device)
    GenNet.set_rh_mano(rh_mano)
    rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).to(device)  # [1, 1538, 3], face indexes
    print("start")
    main(args, GenNet, dataloader, device, rh_mano, rh_faces)

