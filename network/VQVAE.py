
import torch
import torch.nn as nn
import numpy as np
from network.vqvae.encoder import Encoder
from network.vqvae.quantizer import VectorQuantizer
from network.vqvae.decoder import Decoder


    
class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta,a=1, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta,al=a)
        # decode the discrete latent representation
        
        
        #self.decoderobj = Decoder(
        #    layer_sizes=[512,1024], latent_size=1024)
        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, inputs, verbose=False):

        z_e = inputs
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e,True)
        #outputs = [z_q[:,0:384],obj]
        #z_out = torch.cat(outputs, dim=1)
        #x_hat = self.decoder(z_out)
        if verbose:
        #    print('original data shape:', hand.shape,obj.shape)
        #    print('encoded data shape:', z_e.shape)
            assert False

        return embedding_loss, z_q, perplexity
    def inference(self,inputs, verbose=False):

        z_e = inputs
        #z_q, perplexity, _, _ = self.vector_quantization(
        #    z_e,False)
        z_all=self.vector_quantization(
            z_e,False)
        return z_all
    def get_embbeding(self,index,dim):
        #print(index)
        return self.vector_quantization.get_emb(index,dim)
