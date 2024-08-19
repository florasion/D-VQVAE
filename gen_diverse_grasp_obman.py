import numpy as np
import igl as igl
import time
import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import random
from utils import utils, utils_loss
import mano
import json
from utils.loss import TTT_loss
import open3d as o3d
import trimesh
from metric.simulate import run_simulation
from network.gen_net import GenNet
from dataset.dataset_obman_mano2 import obman
import open3d as o3d
import pickle
import json


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


o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

file_names = []
# change the file path to your path
with open('/data/ObMan/obman/test_to_meta.txt', 'r') as file:
    for line in file:
        # 去除每行两侧的空格和换行符
        file_name = line.strip()
        file_names.append(file_name)


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


def main(args, model, eval_loader, device, rh_mano, rh_faces):

    total_high_q = 0
    total_penetr_vol=0.0
    total_simu_disp=0.0
    total_num = 0
    total_contact =0
    total_time=0.0
    total_q=0
    vol_list = []
    mesh_dist_list = []
    id_list=[-1] * 128
    model.eval()
    rh_mano.eval()
    for batch_idx, (obj_pc, hand_param, idx) in enumerate(eval_loader):
            B=1
            obj_pc = obj_pc.to(device)
            ###change the file path to your path
            with open('/data/ObMan/obman/test/meta/'+file_names[batch_idx], 'rb') as f:
                facepkl = pickle.load(f)

            full_path = facepkl['obj_path']
            root_path = "/sequoia/data2/dataset/shapenet"
            extracted_path = full_path[31:]
            real_path = '/data/'+extracted_path
            mesh_o3d = o3d.io.read_triangle_mesh(real_path,print_progress=True)
            vertices =np.asarray(mesh_o3d.vertices)#+ np.asarray(rand_trans.cpu())
            if vertices.shape[0]>50000:
                with open('./logs/Obman_eval.txt', 'a') as file:
                    file.write('batch_idx :{}, too big\n'
                        .format(batch_idx))
                continue
            triangles =np.asarray(mesh_o3d.triangles)
            obj_pose = np.array((facepkl['affine_transform']))
            objdata = utils.vertices_transformation(vertices, obj_pose)
            obj_mesh = trimesh.Trimesh(vertices=objdata, faces=triangles)
            #trimesh.Scene(obj_mesh).show()
            recon_params, R_list, trans_list, r_list = [], [], [], []
            i = 1
            for i in range(0,1):
                time_start=time.time()

                recon_hand ,recon_pos = model.gen(obj_pc)
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
                # evaluate grasp
                cam_extr = np.array([[1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., -1., 0.]]).astype(np.float32)
                obj_mesh_verts = objdata#obj_mesh_verts.dot(cam_extr[:3,:3].T)  # [N,3]
                origin_faces = triangles
                final_mano = rh_mano(betas=recon_param[:, :10], global_orient=recon_param[:, 10:13],
                                    hand_pose=recon_param[:, 13:58], transl=recon_param[:, 58:])
                final_mano_verts = final_mano.vertices.squeeze(0).detach().cpu().numpy()  # [778, 3]
                #final_mano_verts = final_mano_verts.dot(cam_extr[:3,:3].T)
                try:
                    hand_mesh = trimesh.Trimesh(vertices=final_mano_verts, faces=rh_faces.cpu().numpy().reshape((-1, 3)))
                except:
                    continue
                # penetration volume
                
                # contact
                simu_disp = 0.0
                penetration_tol = 0.005
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
                # simulation displacement

                ##change the v-hacd path to your path
                vhacd_exe = "/home/v-hacd-master/TestVHACD"
                try:
                    simu_disp = run_simulation(final_mano_verts, rh_faces.reshape((-1, 3)),
                                            obj_mesh_verts, origin_faces,
                                            vhacd_exe=vhacd_exe)
                    print('run success')
                except:
                    simu_disp = 0.10
                print('simu_disp',simu_disp)

                total_penetr_vol+=penetr_vol
                total_simu_disp+=simu_disp
                total_num+=1
                if sample_contact:
                    total_contact+=1
                print('sample num:{}, mean penetr vol:{}, mean simu disp:{}, contact ratio:{}\n'
                    .format(total_num,total_penetr_vol/total_num,total_simu_disp/total_num,total_contact/total_num))
                with open('./logs/Obman_eval.txt', 'a') as file:
                    file.write('sample num:{}, mean penetr vol:{}, mean simu disp:{}, contact ratio:{}\n'
                            .format(total_num, total_penetr_vol / total_num, total_simu_disp / total_num, total_contact / total_num))
                if True:
                    print('generate id {} saved'.format(i))
                    recon_params.append(recon_param.detach().cpu().numpy().tolist())



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
    parser.add_argument("--num_grasp", type=int, default=1)  # number of grasps you want to generate
    args = parser.parse_args()


    # device
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
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
    dataset = obman(mode="test", vis=True)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
    print('loaded dataset')
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

    main(args, GenNet, dataloader, device, rh_mano, rh_faces )

