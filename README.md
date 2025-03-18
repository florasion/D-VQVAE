# DVQ-VAE Training and Generation

This repository contains code for training the DVQ-VAE model and generating dynamic grasps.

## Usage Instructions

### 1. Train DVQ-VAE

Run the following command to train the DVQ-VAE model:

```bash
python DVQ-VAE/train_obman_mano_vertex.py
```
or download checkpoints from https://drive.google.com/drive/folders/1eVytP-yYs0F6iBbo5qu3jhCRmUwI8hVm?usp=drive_link

### 2. Generate Dynamic Grasps for Objects

To generate dynamic grasps for objects, run:

```
python DVQ-VAE/gen_HDMO_TTA.py
```

### 3. Generate Deformation GIFs

To generate deformation GIFs, run:

```
python DVQ-VAE-2/gen_deform_gif.py
```
