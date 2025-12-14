
import os.path as osp
from scipy import linalg
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from lib.utils import transf_to_CLIP_input, dummy_context_mgr
from lib.utils import mkdir_p
from lib.datasets import prepare_data


############   GAN   ############
def train(dataloader, netG, netD, netC, netpatchD, ganloss, featloss, text_encoder, image_encoder, optimizerG, optimizerD, args):
    batch_size = args.batch_size
    device = args.device
    epoch = args.current_epoch
    max_epoch = args.max_epoch
    z_dim = args.z_dim
    netG, netD, netC, netpatchD, image_encoder = netG.train(), netD.train(), netC.train(), netpatchD.train(), image_encoder.train()

    loop = tqdm(total=len(dataloader))
    for step, data in enumerate(dataloader, 0):
        ##############
        # Train D  
        ##############
        optimizerD.zero_grad()
        with  dummy_context_mgr() as mpc:
            # prepare_data
            real, masks, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, device)
            real = real.requires_grad_()
            sent_emb = sent_emb.requires_grad_()
            words_embs = words_embs.requires_grad_()
            # predict real
            CLIP_real,real_emb = image_encoder(real)
            real_feats = netD(CLIP_real)
            pred_real, errD_real = predict_loss(netC, real_feats, sent_emb, negtive=False)
            # synthesize fake images
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = netG(noise, sent_emb, masks)
            fake_image = fake.detach()
            fake_image.requires_grad_()
            fake_c, real_c = patchGAN_loss(netpatchD, masks, fake_image, real)
            fake_loss = ganloss(fake_c, False, for_discriminator=True)
            real_loss = ganloss(real_c, True, for_discriminator=True)
            patch_loss = (fake_loss + real_loss) * 0.5
            
            CLIP_fake, fake_emb = image_encoder(fake)
            fake_feats = netD(CLIP_fake.detach())
            _, errD_fake = predict_loss(netC, fake_feats, sent_emb, negtive=True)

        # MA-GP
        errD_MAGP = MA_GP_FP32(CLIP_real, sent_emb, pred_real, args.lambda_gp)
        # whole D loss
        with  dummy_context_mgr() as mpc:
            errD = errD_real + errD_fake + errD_MAGP + patch_loss
        # update D
        errD.backward()
        optimizerD.step()
        ##############
        # Train G  
        ##############
        optimizerG.zero_grad()
        with  dummy_context_mgr() as mpc:
            fake_feats = netD(CLIP_fake)
            output = netC(fake_feats, sent_emb)
            text_img_sim = torch.cosine_similarity(fake_emb, sent_emb).mean()
            pred_fake, pred_real = patchGAN_loss(netpatchD, masks, fake, real)
            g_loss = ganloss(pred_fake, True, for_discriminator=False)

            lambda_feat = 0.0
            GAN_Feat_loss = 0.0
            
            if lambda_feat > 0:
                num_D = len(pred_fake)
                GAN_Feat_loss = torch.zeros(1).to(real.device)
                for i in range(num_D):  # for each discriminator
                    # last output is the final prediction, so we exclude it
                    num_intermediate_outputs = len(pred_fake[i]) - 1
                    for j in range(num_intermediate_outputs):  # for each layer output
                        unweighted_loss = featloss(pred_fake[i][j], pred_real[i][j].detach())
                        GAN_Feat_loss += unweighted_loss * lambda_feat / num_D
            
            errG = -output.mean() - args.lambda_cs*text_img_sim + args.lambda_fm*g_loss

        errG.backward()
        optimizerG.step()
        # update loop information
        loop.update(1)
        loop.set_description(f'Train Epoch [{epoch}/{max_epoch}]')
        loop.set_postfix()

    loop.close()


def save_model(netG, netD, netC, optG, optD, epoch, step, save_path):
    state = {'model': {'netG': netG.state_dict(), 'netD': netD.state_dict(), 'netC': netC.state_dict()}, \
            'optimizers': {'optimizer_G': optG.state_dict(), 'optimizer_D': optD.state_dict()},\
            'epoch': epoch}
    torch.save(state, '%s/state_epoch_%03d_%03d.pth' % (save_path, epoch, step))


#########   MAGP   ########
def MA_GP_FP32(img, sent, out, lambda_gp):
    grads = torch.autograd.grad(outputs=out,
                            inputs=(img, sent),
                            grad_outputs=torch.ones(out.size()).cuda(),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    grad0 = grads[0].view(grads[0].size(0), -1)
    grad1 = grads[1].view(grads[1].size(0), -1)
    grad = torch.cat((grad0,grad1),dim=1)                        
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp =  lambda_gp * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp


def sample(dataloader, netG, text_encoder, save_dir, device, z_dim):
    netG.eval()
    for step, data in enumerate(dataloader, 0):
        ######################################################
        # (1) Prepare_data
        ######################################################
        real, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, device)
        ######################################################
        # (2) Generate fake images
        ######################################################
        batch_size = sent_emb.size(0)
        with torch.no_grad():
            noise = torch.randn(batch_size, z_dim).to(device)
            fake_imgs = netG(noise, sent_emb, eval=True).float()
            fake_imgs = torch.clamp(fake_imgs, -1., 1.)

            batch_img_name = 'step_%04d.png'%(step)
            batch_img_save_dir  = osp.join(save_dir, 'batch', 'imgs')
            batch_img_save_name = osp.join(batch_img_save_dir, batch_img_name)
            batch_txt_name = 'step_%04d.txt'%(step)
            batch_txt_save_dir  = osp.join(save_dir, 'batch', 'txts')
            batch_txt_save_name = osp.join(batch_txt_save_dir, batch_txt_name)

            mkdir_p(batch_img_save_dir)
            vutils.save_image(fake_imgs.data, batch_img_save_name, nrow=8, value_range=(-1, 1), normalize=True)
            mkdir_p(batch_txt_save_dir)
            txt = open(batch_txt_save_name,'w')
            for cap in captions:
                txt.write(cap+'\n')
            txt.close()
            for j in range(batch_size):
                im = fake_imgs[j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                ######################################################
                # (3) Save fake images
                ######################################################      
                single_img_name = 'step_%04d.png'%(step)
                single_img_save_dir  = osp.join(save_dir, 'single', 'step%04d'%(step))
                single_img_save_name = osp.join(single_img_save_dir, single_img_name)   

                mkdir_p(single_img_save_dir)   
                im.save(single_img_save_name)
        print('Step: %d' % (step))


def calc_clip_sim(clip, fake, caps_clip, device):
    ''' calculate cosine similarity between fake and text features,
    '''
    # Calculate features
    fake = transf_to_CLIP_input(fake)
    fake_features = clip.encode_image(fake)
    text_features = clip.encode_text(caps_clip)
    text_img_sim = torch.cosine_similarity(fake_features, text_features).mean()
    return text_img_sim


def sample_one_batch(noise, mask, sent, netG, epoch, img_save_dir):
    netG.eval()
    with torch.no_grad():
        B = noise.size(0)
        fixed_results_train = generate_samples(noise[:B//2], mask[:B//2], sent[:B//2], netG).cpu()
        torch.cuda.empty_cache()
        fixed_results_test = generate_samples(noise[B//2:], mask[B//2:], sent[B//2:], netG).cpu()
        torch.cuda.empty_cache()
        fixed_results = torch.cat((fixed_results_train, fixed_results_test), dim=0)
    img_name = 'samples_epoch_%03d.png'%(epoch)
    img_save_path = osp.join(img_save_dir, img_name)
    vutils.save_image(fixed_results.data, img_save_path, nrow=8, value_range=(-1, 1), normalize=True)


def generate_samples(noise, mask, caption, model):
    with torch.no_grad():
        fake = model(noise, caption, mask)
    return fake

def patchGAN_loss(patchD, input_semantics, fake_image, real_image):
    
    fake_concat = torch.cat([input_semantics, fake_image], dim=1)
    real_concat = torch.cat([input_semantics, real_image], dim=1)

    # In Batch Normalization, the fake and real images are
    # recommended to be in the same batch to avoid disparate
    # statistics in fake and real images.
    # So both fake and real images are fed to D all at once.
    fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

    discriminator_out = patchD(fake_and_real)

    # so it's usually a list
    if type(discriminator_out) == list:
        fake = []
        real = []
        for p in discriminator_out:
            fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
            real.append([tensor[tensor.size(0) // 2:] for tensor in p])
    else:
        fake = discriminator_out[:discriminator_out.size(0) // 2]
        real = discriminator_out[discriminator_out.size(0) // 2:]

    return fake, real
    

def predict_loss(predictor, img_feature, text_feature, negtive):
    output = predictor(img_feature, text_feature)
    err = hinge_loss(output, negtive)
    return output,err


def hinge_loss(output, negtive):
    if negtive==False:
        err = torch.mean(F.relu(1. - output))
    else:
        err = torch.mean(F.relu(1. + output))
    return err


def logit_loss(output, negtive):
    batch_size = output.size(0)
    real_labels = torch.FloatTensor(batch_size,1).fill_(1).to(output.device)
    fake_labels = torch.FloatTensor(batch_size,1).fill_(0).to(output.device)
    output = nn.Sigmoid()(output)
    if negtive==False:
        err = nn.BCELoss()(output, real_labels)
    else:
        err = nn.BCELoss()(output, fake_labels)
    return err


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    '''
    print('&'*20)
    print(sigma1)#, sigma1.type())
    print('&'*20)
    print(sigma2)#, sigma2.type())
    '''
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

    
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)
 