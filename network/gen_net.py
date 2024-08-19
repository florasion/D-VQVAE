from sympy import true
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from network.pointnet_encoder import PointNetEncoder
from network.VQVAE import VQVAE
from network.DVQVAE import Encoder,Decoder
from network.pixelcnn.models import GatedPixelCNN

class GenNet(nn.Module):
    def __init__(self):
        super(GenNet, self).__init__()
        self.obj_encoder_type = PointNetEncoder(global_feat=True, feature_transform=False, channel=4)
        self.obj_encoder_pos = PointNetEncoder(global_feat=True, feature_transform=False, channel=4)
        #self.objembnn = Encoder([1024], 1024)
        #print(self.data)
        self.vqvae0 = VQVAE(h_dim=128,res_h_dim=32,n_res_layers=2,n_embeddings=128,embedding_dim=256,beta=0.25,a=1)
        self.vqvae1 = VQVAE(h_dim=128,res_h_dim=32,n_res_layers=2,n_embeddings=128,embedding_dim=256,beta=0.25,a=1)
        self.vqvae2 = VQVAE(h_dim=128,res_h_dim=32,n_res_layers=2,n_embeddings=128,embedding_dim=256,beta=0.25,a=1)
        self.vqvae3 = VQVAE(h_dim=128,res_h_dim=32,n_res_layers=2,n_embeddings=128,embedding_dim=256,beta=0.25,a=1)
        self.vqvae4 = VQVAE(h_dim=128,res_h_dim=32,n_res_layers=2,n_embeddings=128,embedding_dim=256,beta=0.25,a=1)
        self.vqvae5 = VQVAE(h_dim=128,res_h_dim=32,n_res_layers=2,n_embeddings=128,embedding_dim=256,beta=0.25,a=1)
        self.vqvae6 = VQVAE(h_dim=128,res_h_dim=32,n_res_layers=2,n_embeddings=128,embedding_dim=1024,beta=2,a=0)
        self.decoder = Decoder(
            layer_sizes=[1024,256,55], latent_size=2560)
        self.num =0 
        self.rh_mano =None
        self.recon_encoder =  PointNetEncoder(global_feat=True, feature_transform=False, channel=3)
        self.pos_decoder = Decoder(
            layer_sizes=[1024,128,6], latent_size=2048)
        self.GatedPixelCNN =  GatedPixelCNN(512, 512, 15)


    def set_rh_mano(self,Rh_mano):
        Rh_mano.eval()
        self.rh_mano=Rh_mano

    def gen_byid(self,idx6):
        device='cuda'
        B=1
        #label = idx6
        obj_glb_feature_type = self.vqvae6.get_embbeding(idx6,1024)
        
        idx6=idx6.repeat(1, 3, 3).to(device)
        label = idx6[:,0,0].to(device)
        #print(idx6)
        x_tilde = self.GatedPixelCNN.generate(idx6,label, shape=(3,3), batch_size=B)

        #print(x_tilde)
        idx0 = x_tilde[:,0,1]
        idx1 = x_tilde[:,0,2]
        idx2 = x_tilde[:,1,1]
        idx3 = x_tilde[:,1,2]
        idx4 = x_tilde[:,2,1]
        idx5 = x_tilde[:,2,2]
        emb0 = self.vqvae0.get_embbeding(idx0,256)
        emb1 = self.vqvae1.get_embbeding(idx1,256)
        emb2 = self.vqvae2.get_embbeding(idx2,256)
        emb3 = self.vqvae3.get_embbeding(idx3,256)
        emb4 = self.vqvae4.get_embbeding(idx4,256)
        emb5 = self.vqvae5.get_embbeding(idx5,256)
        zero_feature = torch.zeros((B, 256)).to(device)
        zero_features = torch.zeros((B, 1024)).to(device)
        z_out=torch.cat((zero_feature,zero_feature,zero_feature,zero_feature,zero_feature,zero_feature,zero_features),1)
        x_hat=self.decoder(z_out)
        recon = x_hat.contiguous().view(B, 55)


        zero_params = torch.zeros((B, 6)).to(device)
        recon_pos = zero_params
        #recon = x_hat.contiguous().view(B, 61)
        return recon,recon_pos


    def gen(self,obj):
        device='cuda'

        obj_glb_feature_type, _, _ = self.obj_encoder_type(obj) # [B, 1024]
        obj_glb_feature_pos, _ , _ = self.obj_encoder_pos(obj)
        idx6,obj_emb= self.vqvae6.inference(obj_glb_feature_type)
        #label = torch.arange(200).expand(200, 200).contiguous().view(-1)*0
        B=obj.size()[0]
        #label = idx6
        #print(idx6)
        idx6=idx6.repeat(1, 3, 3).to(device)
        label = idx6[:,0,0].to(device)
        #print(idx6)
        
        x_tilde = self.GatedPixelCNN.generate(idx6,label, shape=(3,3), batch_size=B)
        p=False
        #print(x_tilde)
        idx0 = x_tilde[:,0,1]
        idx1 = x_tilde[:,0,2]
        idx2 = x_tilde[:,1,1]
        idx3 = x_tilde[:,1,2]
        idx4 = x_tilde[:,2,1]
        idx5 = x_tilde[:,2,2]
        emb0 = self.vqvae0.get_embbeding(idx0,256)
        emb1 = self.vqvae1.get_embbeding(idx1,256)
        emb2 = self.vqvae2.get_embbeding(idx2,256)
        emb3 = self.vqvae3.get_embbeding(idx3,256)
        emb4 = self.vqvae4.get_embbeding(idx4,256)
        emb5 = self.vqvae5.get_embbeding(idx5,256)
        zero_feature = torch.zeros((B, 256)).to(device)
        zero_features = torch.zeros((B, 1024)).to(device)
        z_out=torch.cat((emb0,emb1,emb2,emb3,emb4,emb5,obj_glb_feature_type),1)
        #z_out=torch.cat((emb0,emb1,emb2,emb3,emb4,emb5,zero_features),1)
        #z_out=torch.cat((zero_feature,zero_feature,zero_feature,zero_feature,zero_feature,zero_feature,obj_glb_feature_type),1)
        x_hat=self.decoder(z_out)
        recon = x_hat.contiguous().view(B, 55)


        zero_params = torch.zeros((B, 3)).to(device)
        recon_mano =  self.rh_mano(betas=recon[:, :10], global_orient=zero_params,
                                hand_pose=recon[:, 10:55], transl=zero_params).vertices  # [B,778,3]
        recon_mano_detach = recon_mano.detach()
        recon_mano_point, _, _ = self.recon_encoder(recon_mano_detach.permute(0,2,1))
        z_out_pos = torch.cat((recon_mano_point,obj_glb_feature_pos),1)
        pos=self.pos_decoder(z_out_pos)
        recon_pos = pos.contiguous().view(B, 6)
        #recon = x_hat.contiguous().view(B, 61)
        return recon,recon_pos

