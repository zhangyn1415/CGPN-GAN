import os, sys
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from lib.utils import choose_model
import open_clip

###########   preparation   ############
def load_clip(clip_info, device):
    # QUILT-Net
    model, _, _ = open_clip.create_model_and_transforms(clip_info, 
        pretrained="/home/zhangyn/quilt1m-main/wisdomik/QuiltNet-B-32/open_clip_pytorch_model.bin", cache_dir=None)
    
    return model

def prepare_models(args):

    device = args.device
    CLIP4trn = load_clip(args.clip4text, device).eval().to(device)
    CLIP4evl = load_clip(args.clip4text, device).eval().to(device)
    NetG,NetD,NetC,CLIP_IMG_ENCODER,CLIP_TXT_ENCODER, patchD = choose_model(args.model)

    # image encoder
    CLIP_img_enc = CLIP_IMG_ENCODER(CLIP4trn).to(device)
    for p in CLIP_img_enc.parameters():
        p.requires_grad = False
    CLIP_img_enc.eval()

    # text encoder
    CLIP_txt_enc = CLIP_TXT_ENCODER(CLIP4trn).to(device)
    for p in CLIP_txt_enc.parameters():
        p.requires_grad = False
    CLIP_txt_enc.eval()

    # GAN models
    netG = NetG(args.nf, args.z_dim, args.cond_dim, args.imsize, args.mask_shape, args.ch_size).to(device)
    netD = NetD(args.nf, args.imsize, args.ch_size).to(device)
    netC = NetC(args.nf, args.cond_dim).to(device)
    netpatchD = patchD(args.mask_shape).to(device)

    return CLIP4trn, CLIP4evl, CLIP_img_enc, CLIP_txt_enc, netG, netD, netC, netpatchD

def prepare_dataset(args, split, transform):

    from lib.datasets import Nuclei_Text_Dataset
    dataset = Nuclei_Text_Dataset(split=split, transform=transform, args=args)
    return dataset


def prepare_datasets(args, transform):
    # train dataset
    train_dataset = prepare_dataset(args, split='train', transform=transform)
    # test dataset
    val_dataset = prepare_dataset(args, split='test', transform=transform)
    return train_dataset, val_dataset


def prepare_dataloaders(args, transform=None):

    batch_size = args.batch_size
    num_workers = args.num_workers
    train_dataset, valid_dataset = prepare_datasets(args, transform)

    # train dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True,
        num_workers=num_workers, shuffle='True')

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, drop_last=True,
        num_workers=num_workers, shuffle='False')
    
    return train_dataloader, valid_dataloader, \
            train_dataset, valid_dataset

