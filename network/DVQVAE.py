import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from network.pointnet_encoder import PointNetEncoder
from network.VQVAE import VQVAE

class DVQVAE(nn.Module):
    def __init__(self, obj_inchannel=4):
        super(DVQVAE, self).__init__()
        self.obj_inchannel = obj_inchannel
        self.handembnns = [Encoder([1024, 512], 256) for _ in range(6)]
        for i, emb in enumerate(self.handembnns):
            self.add_module('emb_{}'.format(i), emb)
        self.obj_encoder_type = PointNetEncoder(global_feat=True, feature_transform=False, channel=self.obj_inchannel)
        self.obj_encoder_pos = PointNetEncoder(global_feat=True, feature_transform=False, channel=self.obj_inchannel)
        self.hand_encoders = [PointNetEncoder(global_feat=True, feature_transform=False, channel=3) for _ in range(6)]
        for i, fingerEncoder in enumerate(self.hand_encoders):
            self.add_module('fing_{}'.format(i), fingerEncoder)
        self.vqvae0 = VQVAE(h_dim=128,res_h_dim=32,n_res_layers=2,n_embeddings=128,embedding_dim=256,beta=0.25,a=1)
        self.vqvae1 = VQVAE(h_dim=128,res_h_dim=32,n_res_layers=2,n_embeddings=128,embedding_dim=256,beta=0.25,a=1)
        self.vqvae2 = VQVAE(h_dim=128,res_h_dim=32,n_res_layers=2,n_embeddings=128,embedding_dim=256,beta=0.25,a=1)
        self.vqvae3 = VQVAE(h_dim=128,res_h_dim=32,n_res_layers=2,n_embeddings=128,embedding_dim=256,beta=0.25,a=1)
        self.vqvae4 = VQVAE(h_dim=128,res_h_dim=32,n_res_layers=2,n_embeddings=128,embedding_dim=256,beta=0.25,a=1)
        self.vqvae5 = VQVAE(h_dim=128,res_h_dim=32,n_res_layers=2,n_embeddings=128,embedding_dim=256,beta=0.25,a=1)
        self.vqvae6 = VQVAE(h_dim=128,res_h_dim=32,n_res_layers=2,n_embeddings=128,embedding_dim=1024,beta=2,a=0)
        self.decoder = Decoder(
            layer_sizes=[1024,256,55], latent_size=2560)
        self.rh_mano =None
        self.recon_encoder =  PointNetEncoder(global_feat=True, feature_transform=False, channel=3)
        self.pos_decoder = Decoder(
            layer_sizes=[1024,128,6], latent_size=2048)


    def set_rh_mano(self,Rh_mano):
        self.rh_mano=Rh_mano
        Rh_mano.eval()

    def forward(self, obj_pc, hand_xyz):
        '''
        :param obj_pc: [B, 3+n, N1]
        :param hand_param: [B, 61]
        :return: reconstructed hand vertex
        '''
        B = hand_xyz.size(0)
        mean_xyz = hand_xyz.mean(dim=(2)) 
        mean_xyz = mean_xyz.unsqueeze(2).repeat(1,1,778)
        hand_xyz=hand_xyz-mean_xyz


        handc = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 50, 51, 52, 53, 54, 55,
        60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
        88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 
        111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
        132, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 157, 158, 159, 178, 179, 180,
        181, 182, 183, 184, 188, 190, 191, 192, 193, 196, 197, 198, 199, 200, 203, 204, 205, 206, 207, 207, 208,
        209, 210, 211, 214, 215, 216, 217, 218, 219, 220, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 239,
        241, 242, 243, 244, 247, 254, 255, 256, 257, 264, 265, 268, 271, 275, 276, 279, 284, 285, 769, 770, 771,
        772, 773, 774, 775, 776, 777]
        f1hand = [46, 47, 48, 49, 56, 57, 58, 59, 86, 87, 133, 134, 135, 136, 137, 138, 139, 140, 155, 156, 164,
        165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 186, 189, 194, 195, 212, 213, 221, 222,
        223, 224, 225, 226, 237, 238, 245, 258, 259, 260, 261, 263, 272, 273, 274, 280, 281, 282, 283, 294, 295,
        296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316,
        317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337,
        338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355]
        f2hand = [185, 187, 246, 262, 269, 270, 277, 288, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366,
        367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387,
        388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408,
        409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429,
        430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450,
        451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467]
        f3hand = [160, 161, 162, 162, 163, 163, 290, 291, 292, 293, 468, 469, 470, 471, 472, 473, 474, 475, 476,
        477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497,
        498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518,
        519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539,
        540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560,
        561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579]
        f4hand = [201, 202, 278, 289, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594,
        595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615,
        616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636,
        637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657,
        658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678,
        679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696]


        
        obj_glb_feature_type, _, _ = self.obj_encoder_type(obj_pc) # [B, 1024]
        obj_glb_feature_pos, _, _ = self.obj_encoder_pos(obj_pc) # [B, 1024]
        finger_indices = [f0hand, f1hand, f2hand, f3hand, f4hand, handc]
        finger_features = [hand_encoder(hand_xyz[:, :, idx]) for hand_encoder, idx in zip(self.hand_encoders, finger_indices)]

        embeddings= [handembnn(feature[0]) for handembnn, feature in zip(self.handembnns, finger_features)]
        f0_emb, f1_emb, f2_emb, f3_emb, f4_emb, fc_emb = embeddings
        hand_emb=torch.cat((f0_emb,f1_emb,f2_emb,f3_emb,f4_emb,fc_emb), 1)	

        if self.training:
            embedding_loss0, x_hat0, perplexity0 = self.vqvae0(f0_emb) 
            embedding_loss1, x_hat1, perplexity1 = self.vqvae1(f1_emb)
            embedding_loss2, x_hat2, perplexity2 = self.vqvae2(f2_emb)
            embedding_loss3, x_hat3, perplexity3 = self.vqvae3(f3_emb)
            embedding_loss4, x_hat4, perplexity4 = self.vqvae4(f4_emb)
            embedding_loss5, x_hat5, perplexity5 = self.vqvae5(fc_emb)
            embedding_loss6, obj_emb, perplexity5 = self.vqvae6(obj_glb_feature_type)

            z_out=torch.cat((x_hat0,x_hat1,x_hat2,x_hat3,x_hat4,x_hat5,obj_glb_feature_type),1)



            x_hat=self.decoder(z_out)
            recon = x_hat.contiguous().view(B, 55)


            zero_params = torch.zeros((B, 3)).to('cuda')
            recon_mano =  self.rh_mano(betas=recon[:, :10], global_orient=zero_params,
                                    hand_pose=recon[:, 10:55], transl=zero_params).vertices  # [B,778,3]
            recon_mano_detach = recon_mano.detach()
            recon_mano_point, _, _ = self.recon_encoder(recon_mano_detach.permute(0,2,1))


            z_out_pos = torch.cat((recon_mano_point,obj_glb_feature_pos),1)
            pos=self.pos_decoder(z_out_pos)
            recon_pos = pos.contiguous().view(B, 6)

            embedding_loss = embedding_loss0 + embedding_loss1 + embedding_loss2 + embedding_loss3 + embedding_loss4 + embedding_loss5 + embedding_loss6
            return recon ,recon_pos, embedding_loss ,  perplexity0 #,obj_recon
        else:
            # inference
            idx0,x_hat0= self.vqvae0.inference(f0_emb) 
            idx1,x_hat1= self.vqvae1.inference(f1_emb) 
            idx2,x_hat2= self.vqvae2.inference(f2_emb) 
            idx3,x_hat3= self.vqvae3.inference(f3_emb) 
            idx4,x_hat4= self.vqvae4.inference(f4_emb) 
            idx5,x_hat5= self.vqvae5.inference(fc_emb) 
            idx6,obj_emb= self.vqvae6.inference(obj_glb_feature_type)

            emb_idx=torch.cat((idx6,idx0,idx1,idx2,idx3,idx4,idx5),0)

            return emb_idx,obj_emb


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()
        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)


    def forward(self, x):

        x = self.MLP(x)
        means = self.linear_means(x)

        return means


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):
        super().__init__()

        self.MLP = nn.Sequential()
        input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

    def forward(self, z):
        x = self.MLP(z)
        return x


