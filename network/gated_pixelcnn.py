

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from torchvision.utils import save_image
import time
import os 
import sys
"""
add vqvae and pixelcnn dirs to path
make sure you run from vqvae directory
"""
from pixelcnn.models import GatedPixelCNN
import utils

"""
Hyperparameters
"""
import argparse 
parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("-save", action="store_true")
parser.add_argument("-gen_samples", default=True, action="store_true")

parser.add_argument("--dataset",  type=str, default='LATENT_BLOCK',
    help='accepts CIFAR10 | MNIST | FashionMNIST | LATENT_BLOCK')
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--img_dim", type=int, default=8)
parser.add_argument("--input_dim", type=int, default=1,
    help='1 for grayscale 3 for rgb')
parser.add_argument("--n_embeddings", type=int, default=512,
    help='number of embeddings from VQ VAE')
parser.add_argument("--n_layers", type=int, default=15)
parser.add_argument("--learning_rate", type=float, default=3e-4)

args = parser.parse_args()

device = torch.device("cuda")

"""
data loaders
"""
if args.dataset == 'LATENT_BLOCK':
    _, _, train_loader, test_loader, _ = utils.load_data_and_data_loaders('LATENT_BLOCK', args.batch_size)
else:
    train_loader = torch.utils.data.DataLoader(
        eval('datasets.'+args.dataset)(
            '../data/{}/'.format(args.dataset), train=True, download=True,
            transform=transforms.ToTensor(),
        ), batch_size=args.batch_Size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        eval('datasets.'+args.dataset)(
            '../data/{}/'.format(args.dataset), train=False,
            transform=transforms.ToTensor(),
        ), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

model = GatedPixelCNN(args.n_embeddings, 512, args.n_layers).to(device)
criterion = nn.CrossEntropyLoss().cuda()
opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


"""
train, test, and log
"""

def train():
    train_loss = []
    for batch_idx, (x, label) in enumerate(train_loader):
        x=x.to('cuda')
        #print('x',x.size())
        #x = x.repeat(1, 1, 7)
        #print('x',x.size())
        start_time = time.time()
        #print(x.size())
        #if args.dataset == 'LATENT_BLOCK':
        #    x = (x[:, 0]).cuda()
        #else:
        #    x = (x[:, 0] * (K-1)).long().cuda()
        label = label.cuda()
       
        # Train PixelCNN with images
        logits = model(x, label)
        logits = logits.permute(0, 2, 3, 1).contiguous()
        loss = criterion(
            logits.view(-1, args.n_embeddings),
            x.view(-1)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss.append(loss.item())

        if (batch_idx + 1) % args.log_interval == 0:
            print('\tIter: [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(x), len(train_loader.dataset),
                args.log_interval * batch_idx / len(train_loader),
                np.asarray(train_loss)[-args.log_interval:].mean(0),
                time.time() - start_time
            ))


def test():
    start_time = time.time()
    val_loss = []
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            #if args.dataset == 'LATENT_BLOCK':
            #    x = (x[:, 0]).cuda()
            #else:
            #  x = (x[:, 0] * (args.n_embeddings-1)).long().cuda()
            #print(x)
            x=x.to('cuda')
            #x = x.repeat(1, 1, 7)
            label = label.cuda()

            logits = model(x, label)
            
            
            logits = logits.permute(0, 2, 3, 1).contiguous()
            #print(logits.size())
            loss = criterion(
                logits.view(-1, args.n_embeddings),
                x.view(-1)
            )
            
            val_loss.append(loss.item())

    print('Validation Completed!\tLoss: {} Time: {}'.format(
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)


def generate_samples(epoch,x_start):
    label = torch.arange(200).expand(200, 200).contiguous().view(-1)*0
    #label = torch.zeros(100, 100, dtype=torch.int).view(-1)
    B=x_start.size()[0]
    #print(x_start.size()[0])
    label = label[:B].to('cuda')
    #x = x.repeat(1, 1, 7)
    print(x_start[0])
    #print(x_start[0,0,:])
    x_tilde = model.generate(x_start,label, shape=(3,3), batch_size=B)
    print(x_tilde[0])



BEST_LOSS = 999
LAST_SAVED = -1
for epoch in range(1, args.epochs):
    print("\nEpoch {}:".format(epoch))  
    train()
    cur_loss = test()

    if args.save or cur_loss <= BEST_LOSS:
        BEST_LOSS = cur_loss
        LAST_SAVED = epoch

        print("Saving model!")
        torch.save(model.state_dict(), '/data/Model/VQVAE/results/{}_pixelcnn.pt'.format(args.dataset))
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))
    if args.gen_samples:
        for batch_idx, (x, label) in enumerate(test_loader):
            print(x.size())
            x=x.to('cuda')
            #x = x.repeat(1, 1, 7)
            generate_samples(epoch,x)
            break
