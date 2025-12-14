import os, sys
import os.path as osp
import time
import random
import argparse
import numpy as np
import json
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
import multiprocessing as mp

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)

from lib.utils import mkdir_p,merge_args_yaml,get_time_stamp
from lib.utils import load_models_opt,save_models_opt,get_model_size
from lib.perpare import prepare_dataloaders,prepare_models
from lib.modules import sample_one_batch as sample, train, GANLoss
from lib.datasets import get_fix_data



def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='/home/data/CgDaPn-GAN/cfg/PanNuke.yml', help='optional config file')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers(default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--stamp', type=str, default='normal', help='the stamp of model')
    parser.add_argument('--pretrained_model', type=str, default='', help='initialization')
    parser.add_argument('--log_dir', type=str, default='new', help='file path to log directory')
    parser.add_argument('--model', type=str, default='CGPNGAN', help='the model for training')
    parser.add_argument('--max_epoch', type=int, default=300, help='training epoch')
    parser.add_argument('--manual_seed', type=int, default=100, help='seed')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    parser.add_argument('--train', type=str, default='True',help='if train model')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--random_sample', action='store_true',default=True,  help='whether to sample the dataset with random sampler')
    parser.add_argument('--mask_colors', help='color for nuclei masks (backgroud)',  type=str, default="[[245, 255, 250], [205, 55, 0], [238, 118, 33], [0, 205, 102], [65, 105, 225], [238, 201, 0]]")
    parser.add_argument('--mask_shape', type=list, default=[6, 256, 256])
    args = parser.parse_args()
    return args



def main(args):

    time_stamp = get_time_stamp()
    stamp = '_'.join([str(args.model),'nf'+str(args.nf),str(args.stamp),str(args.dataset_name),str(args.imsize),time_stamp])
    args.model_save_file = osp.join(ROOT_PATH, 'saved_models', str(args.dataset_name), stamp)

    args.img_save_dir = osp.join(ROOT_PATH, 'imgs/{0}'.format(osp.join(str(args.dataset_name), 'train', stamp)))
    mkdir_p(args.model_save_file)
    mkdir_p(args.img_save_dir)

    # prepare dataloader, models, data
    train_dl, valid_dl ,train_ds, valid_ds = prepare_dataloaders(args, transform=None )
    CLIP4trn, CLIP4evl, image_encoder, text_encoder, netG, netD, netC, netpatchD = prepare_models(args)

    ganloss = GANLoss('hinge', tensor=torch.cuda.FloatTensor).to(args.device)
    featloss = torch.nn.L1Loss()
    
    print('G_size----- ',get_model_size(netG))
    print('D_size----- ',get_model_size(netD)+get_model_size(netC))

    fixed_img, fixed_mask, fixed_sent, fixed_words, fixed_z, fixed_text = get_fix_data(train_dl, valid_dl, text_encoder, args)

    img_name = 'ground_truth.png'
    img_save_path = osp.join(args.img_save_dir, img_name)
    vutils.save_image(fixed_img.data, img_save_path, nrow=8, normalize=True)
    
    with open(osp.join(args.img_save_dir, "text_description.txt"), 'w') as fp:
        for text in fixed_text:
            fp.write(text+"\n")

     ### draw nucllei masks ----------   
    mask_name = 'nuclei_mask.png'
    mask_save_path = osp.join(args.img_save_dir, mask_name)

    colors=json.loads(args.mask_colors)
    nuclei_types = len(colors)-1
    MASK_ = torch.zeros(fixed_mask.data.shape[0], 3, fixed_mask.data.shape[2], fixed_mask.data.shape[3], dtype=torch.float32).to(fixed_mask.device)
    masks = fixed_mask.permute(0,2,3,1)
    masks = masks.contiguous().view(-1, nuclei_types) 
    MASK_ = MASK_.permute(0,2,3,1)
    MASK_ = MASK_.contiguous().view(-1, 3)

    for i in range(nuclei_types):
        MASK_[masks[:,i] == 1.0,:] = torch.tensor(colors[i]).to(torch.float32).to(fixed_mask.device)
    
    MASK_ = MASK_.reshape(fixed_mask.data.shape[0], fixed_mask.data.shape[2], fixed_mask.data.shape[3], 3)
    MASK_ = MASK_.permute(0,3,1,2)
    MASK_ = MASK_ / 255.0
        
    vutils.save_image(MASK_.data, mask_save_path, nrow=8, normalize=True)
    ###### -------------------------#

    # prepare optimizer
    D_params = list(netD.parameters()) + list(netC.parameters()) + list(netpatchD.parameters())
    optimizerD = torch.optim.Adam(D_params, lr=args.lr_d, betas=(0.0, 0.9))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g, betas=(0.0, 0.9))


    # load from checkpoint
    if len( args.pretrained_model ) > 0:
        path = args.pretrained_model
        netG, netD, netC, optimizerG, optimizerD = load_models_opt(netG, netD, netC, optimizerG, optimizerD, path)

    # Start training
    test_interval, gen_interval,save_interval = args.test_interval,args.gen_interval,args.save_interval
    
    for epoch in range(args.max_epoch):

        args.current_epoch = epoch
        # training
        torch.cuda.empty_cache()
        train(train_dl, netG, netD, netC, netpatchD, ganloss, featloss, text_encoder, image_encoder, optimizerG, optimizerD, args)
        torch.cuda.empty_cache()
        # save
        if epoch%save_interval==0:
            save_models_opt(netG, netD, netC, optimizerG, optimizerD, epoch, args.model_save_file)
            torch.cuda.empty_cache()
        # sample
        if epoch%gen_interval==0:
            sample(fixed_z, fixed_mask, fixed_sent, netG, epoch, args.img_save_dir)
            torch.cuda.empty_cache()



if __name__ == "__main__": 
    
    args = merge_args_yaml(parse_args())
    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.manual_seed)
        torch.cuda.set_device(args.gpu_id)
        args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')
    main(args)

