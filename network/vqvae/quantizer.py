import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda")


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta ,al):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.al = al
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)


    def forward(self, z, istrain):

        z_flattened = z.view(-1, self.e_dim)

        z_all=torch.Tensor([]).to(device)
        if istrain:
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1) - 2 * \
                torch.matmul(z_flattened, self.embedding.weight.t())
            min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
            min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.n_e).to(device)
            min_encodings.scatter_(1, min_encoding_indices, 1)
            z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        else :

            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1) - 2 * \
                torch.matmul(z_flattened, self.embedding.weight.t())
            min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
            min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.n_e).to(device)
            min_encodings.scatter_(1, min_encoding_indices, 1)
            z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
            return min_encoding_indices,z_q

        loss = self.al*torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))


        return loss, z_q, perplexity, min_encodings, min_encoding_indices
    def get_emb(self, min_encoding_indices,dim):
        #print(min_encoding_indices)
        min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices.unsqueeze(0), 1)
        
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(1,dim)
        return z_q