import os
import numpy as np
from PIL import Image
import numpy.random as random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import open_clip
import random
import scipy.io as sio
import glob
import json

pannuke_color = np.array([[245, 255, 250], [205, 55, 0], [238, 118, 33], [0, 205, 102], [65, 105, 225], [238, 201, 0]])

def get_fix_data(train_dl, test_dl, text_encoder, args):
    fixed_image_train, fixed_mask_train, _, fixed_sent_train, fixed_word_train, fixed_key_train = get_one_batch_data(train_dl, text_encoder, args)
    fixed_image_test, fixed_mask_test, _, fixed_sent_test, fixed_word_test, fixed_key_test= get_one_batch_data(test_dl, text_encoder, args)
    fixed_image = torch.cat((fixed_image_train, fixed_image_test), dim=0)
    fixed_mask = torch.cat((fixed_mask_train, fixed_mask_test), dim=0)
    fixed_sent = torch.cat((fixed_sent_train, fixed_sent_test), dim=0)
    fixed_word = torch.cat((fixed_word_train, fixed_word_test), dim=0)
    fixed_noise = torch.randn(fixed_image.size(0), args.z_dim).to(args.device)
    
    # fixed_text = [i[i.find("_")+1:(i.rfind("frozen")-1 if "frozen" in i else i.rfind("formalin")-1)] for i in fixed_key_train+fixed_key_test]
    fixed_text = [i for i in fixed_key_train+fixed_key_test]
    id = sorted(range(len(fixed_text)), key=lambda k: fixed_text[k], reverse=False)
    
    fixed_image = torch.stack([fixed_image[k,:,:,:] for k in id], axis=0)
    fixed_mask = torch.stack([fixed_mask[k,:,:,:] for k in id], axis=0)
    fixed_sent = torch.stack([fixed_sent[k,:] for k in id], axis=0)
    fixed_word = torch.stack([fixed_word[k,:,:] for k in id], axis=0)
    fixed_noise = torch.stack([fixed_noise[k,:] for k in id], axis=0)
    
    return fixed_image, fixed_mask, fixed_sent, fixed_word, fixed_noise, sorted(fixed_text)


def get_one_batch_data(dataloader, text_encoder, args):
    data = next(iter(dataloader))
    imgs, masks, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, args.device)
    return imgs, masks, CLIP_tokens, sent_emb, words_embs, keys


def prepare_data(data, text_encoder, device):
    imgs, masks, CLIP_tokens, keys = data
    imgs, masks, CLIP_tokens = imgs.to(device), masks.to(device).to(torch.float32), CLIP_tokens.to(device)
    sent_emb, words_embs = encode_tokens(text_encoder, CLIP_tokens)
    return imgs, masks, CLIP_tokens, sent_emb, words_embs, keys


def encode_tokens(text_encoder, caption):
    # encode text
    with torch.no_grad():
        sent_emb,words_embs = text_encoder(caption)
        sent_emb,words_embs = sent_emb.detach(), words_embs.detach()
    return sent_emb, words_embs 


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


def get_transforms():
    trans = []
    trans.append(
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0)
        ])
    )
    trans.append(
        transforms.Compose([
            transforms.RandomVerticalFlip(p=1.0)
        ])        
    )
    trans.append(
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0)
        ])        
    ) 
    trans.append(
        transforms.Compose([
            transforms.RandomRotation((90,90))
        ])        
    )   
    trans.append(
        transforms.Compose([])
    )
    trans.append(
        transforms.Compose([])
    )
    trans.append(
        transforms.Compose([])
    )
    trans.append(
        transforms.Compose([])
    )

    return trans

################################################################
#                    Dataset
################################################################
class Nuclei_Text_Dataset(data.Dataset):
    def __init__(self, split="train", transform=None, args=None):
        colors=json.loads(args.mask_colors)
        self.colors = np.array(colors)
        self.spilt = split

        if transform is None:
            self.transform = get_transforms()
        else:
            self.transform = transform

        self.clip4text = args.clip4text
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        path = args.data_dir + (   "train/masks" if split == "train" else "test/masks"   )
        self.imgnames, self.masknames, self.filenames = self.load_PanNukenames( data_dir = path, text_desc = args.data_dir + args.text_name )

        self.tokenizer = open_clip.get_tokenizer(self.clip4text)  


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

        if self.spilt == 'train':
            transform = self.transform[ random.randint(0,len(self.transform)-1 ) ]
        else:
            transform = self.transform[-1]

        imgs = get_imgs(img_name, transform, normalize=self.norm )
        tokens = get_caption(text_desc, self.tokenizer )
        img_masks = get_masks(masknames, transform )
           
        return imgs, img_masks, tokens, text_desc

    def __len__(self):
        return len(self.filenames)
    

if __name__ == "__main__":
    #######PanNuke datasets....
    data = Nuclei_Text_Dataset()