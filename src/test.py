import os, sys
import os.path as osp
import time
import random
import argparse
import numpy as np
from PIL import Image

from tqdm import tqdm
import torch
from torchvision.utils import save_image
# from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import multiprocessing as mp

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)

from lib.utils import merge_args_yaml
from lib.utils import load_netG
from lib.perpare import prepare_models
import scipy.io as sio
import open_clip
import glob
import torch.utils.data as data
import json
from PIL import Image



def get_imgs(img_path, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB') 
    if transform is not None:
        img = transform(img)
    if normalize is not None:
        img = normalize(img)
    return img


def get_masks(img_path, transform=None):
    maps = sio.loadmat(img_path)['inst_map']
    maps = np.where(maps > 0, 1, 0)[:,:,:-1]
    labels = maps.transpose(2,0,1)

    return transform(torch.from_numpy(labels))


def get_caption(cap_path, tokenizer):
    
    tokens = tokenizer([cap_path])
    return  tokens[0]



def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

################################################################
#                    Dataset
################################################################
class Nuclei_Text_Dataset(data.Dataset):
    def __init__(self, transform=None, args=None):
        colors=json.loads(args.mask_colors)

        self.colors = np.array(colors)
        self.transform = transform
        self.clip4text = args.clip4text
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        
        path = args.data_dir +  "train/fake_masks" 
        self.imgnames, self.masknames, self.filenames = self.load_PanNukenames( data_dir = path, text_desc = args.data_dir + args.text_name )
          
        self.tokenizer = open_clip.get_tokenizer(self.clip4text)    
        self.number_example = len(self.filenames)


    def load_PanNukenames(self, data_dir, text_desc ):
    
        with open(text_desc, 'r', encoding='utf-8') as fp:
            text_description = json.load(fp)

        filepath = sorted(os.listdir(data_dir))
        filenames = []
        masknames = []
        textnames = []
        for mask_ in filepath:
            textnames.append(text_description[mask_])
            img_p = os.path.join(data_dir, mask_).replace("masks", "imgs").replace(".mat", ".jpg")
            filenames.append(img_p)
            masknames.append(os.path.join(data_dir, mask_) )

        return filenames, masknames, textnames  
    
    def __getitem__(self, index):
        #
        img_name = self.imgnames[index]
        masknames = self.masknames[index]
        text_desc = self.filenames[index]

        fake_img = img_name #.replace("imgs", "fake_imgs")
        tokens = get_caption(text_desc, self.tokenizer )
        img_masks = get_masks(masknames, self.transform )
           
        return img_masks, tokens, fake_img

    
    def __len__(self):
        return len(self.filenames)
    



def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='./cfg/PanNuke.yml', help='optional config file')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers(default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--stamp', type=str, default='normal', help='the stamp of model')
    parser.add_argument('--pretrained_model', type=str, default='/home/data/CgDaPn-GAN/saved_models/PanNuke/CGPNGAN_nf64_normal_PanNuke_256_2025_08_10_18_45_17/state_epoch_290.pth', help='initialization')
    parser.add_argument('--log_dir', type=str, default='new', help='file path to log directory')
    parser.add_argument('--model', type=str, default='CGPNGAN', help='the model for training')
    parser.add_argument('--manual_seed', type=int, default=100, help='seed')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    parser.add_argument('--train', type=str, default='True',  help='if train model')
    parser.add_argument('--gpu_id', type=int, default=0,  help='gpu id')
    parser.add_argument('--random_sample', action='store_true',default=True,  help='whether to sample the dataset with random sampler')
    parser.add_argument('--mask_colors', help='color for nuclei masks (backgroud)',  type=str, default="[[245, 255, 250], [205, 55, 0], [238, 118, 33], [0, 205, 102], [65, 105, 225], [238, 201, 0]]")
    parser.add_argument('--mask_shape', type=list, default=[6, 256, 256])
    args = parser.parse_args()
    return args


def main(args): 

    args.text_name = 'fake_PanNuke_text.txt'
    dataset = Nuclei_Text_Dataset(transform=transforms.Compose([]), args=args)
    
    valid_dl = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, drop_last=True,
            num_workers=args.num_workers, shuffle=False)  

    CLIP4trn, CLIP4evl, image_encoder, text_encoder, netG, netD, netC, _ = prepare_models(args)
    state_path = args.pretrained_model

    netG = load_netG(netG, state_path, args.train)

    netG.eval()
    
    for i, data in enumerate(tqdm(valid_dl)):

        masks, CLIP_tokens, keys = data
        masks, CLIP_tokens = masks.to(args.device).to(torch.float32), CLIP_tokens.to(args.device)
        with torch.no_grad():
            sent_emb, words_embs = text_encoder(CLIP_tokens)
            sent_emb, words_embs = sent_emb.detach(), words_embs.detach()

            noise = torch.randn(args.batch_size, args.z_dim).to(args.device)
            fake_imgs = netG(noise,sent_emb, masks).detach()
        
        for img, name in zip(fake_imgs, keys):
            IMG = tensor2im(img)
            save_image(IMG, name)


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


