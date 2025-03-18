import os
from torch import optim, nn, utils, Tensor
import lightning as L
import pytorch_lightning as pl
from network.softNet import softNet
from datasets.dataset_HMDO_batch_fast import HMDO
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.loggers import TensorBoardLogger

autoencoder = softNet()
autoencoder.load_face( batchsize=6)
autoencoder = autoencoder.cuda()
train_dataset = HMDO(mode="train", batch_size=6)
train_loader = DataLoader(dataset=train_dataset, batch_size=6, shuffle=True,
                          num_workers=1,drop_last=True)


tb_logger = pl_loggers.TensorBoardLogger(save_dir="./logs/")
trainer = pl.Trainer( max_epochs=300,accelerator="gpu",log_every_n_steps=1,
                                               logger=tb_logger,
                                               accumulate_grad_batches=6,
                                               )
trainer.fit(model=autoencoder,train_dataloaders=train_loader)