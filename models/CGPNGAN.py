####### 生成器，判别器未采用残差
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from .functions import (get_activ_layer, get_downsample_x2_layer, get_norm_layer, SPADE, BaseNetwork, get_nonspade_norm_layer)
   
class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False    

class CLIP_IMG_ENCODER(nn.Module):
    def __init__(self, CLIP):
        super(CLIP_IMG_ENCODER, self).__init__()
        model = CLIP.visual
        print(model)
        self.define_module(model)
        for param in self.parameters():
            param.requires_grad = False

    def define_module(self, model):
        self.conv1 = model.conv1
        self.class_embedding = model.class_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_pre = model.ln_pre
        self.transformer = model.transformer
        self.ln_post = model.ln_post
        self.proj = model.proj

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def transf_to_CLIP_input(self,inputs):
        device = inputs.device
        if len(inputs.size()) != 4:
            raise ValueError('Expect the (B, C, X, Y) tensor.')
        else:
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])\
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            var = torch.tensor([0.26862954, 0.26130258, 0.27577711])\
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            inputs = F.interpolate(inputs*0.5+0.5, size=(224, 224))
            inputs = ((inputs+1)*0.5-mean)/var
            return inputs

    def forward(self, img: torch.Tensor):
        x = self.transf_to_CLIP_input(img)
        x = x.type(self.dtype)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        grid =  x.size(-1)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        # NLD -> LND
        x = x.permute(1, 0, 2)
        # Local features
        #selected = [1,4,7,12]
        selected = [1,4,8]
        local_features = []
        for i in range(12):
            x = self.transformer.resblocks[i](x)
            if i in selected:
                local_features.append(x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, grid, grid).contiguous().type(img.dtype))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return torch.stack(local_features, dim=1), x.type(img.dtype)


class CLIP_TXT_ENCODER(nn.Module):
    def __init__(self, CLIP):
        super(CLIP_TXT_ENCODER, self).__init__()
        self.define_module(CLIP)
        # print(model)
        for param in self.parameters():
            param.requires_grad = False

    def define_module(self, CLIP):
        self.transformer = CLIP.transformer
        self.vocab_size = CLIP.vocab_size
        self.token_embedding = CLIP.token_embedding
        self.positional_embedding = CLIP.positional_embedding
        self.ln_final = CLIP.ln_final
        self.text_projection = CLIP.text_projection

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        sent_emb = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return sent_emb, x


class CLIP_Adapter(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, G_ch, CLIP_ch, cond_dim, k, s, p, map_num):
        super(CLIP_Adapter, self).__init__()
        self.CLIP_ch = CLIP_ch
        self.FBlocks = nn.ModuleList([])
        for i in range(map_num):
            self.FBlocks.append(M_Block(out_ch, mid_ch, out_ch, cond_dim, k, s, p))
                
        self.conv_fuse = nn.Conv2d(out_ch, G_ch, 5, 1, 2)


    def forward(self, out, c):
        
        for k, FBlock in enumerate(self.FBlocks):
            out = FBlock(out, c)
                
        fuse_feat = self.conv_fuse(out)
        
        return fuse_feat


##### Mask Image encoding
class UnetBasicBlock(nn.Module):

    def __init__(
        self, in_features, out_features, activ, norm, mid_features = None
    ):
        super().__init__()

        if mid_features is None:
            mid_features = out_features

        self.block = nn.Sequential(
            get_norm_layer(norm, in_features),
            nn.Conv2d(in_features, mid_features, kernel_size = 3, padding = 1),
            get_activ_layer(activ),

            get_norm_layer(norm, mid_features),
            nn.Conv2d(
                mid_features, out_features, kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ),
        )

    def forward(self, x):
        return self.block(x)

class UNetEncBlock(nn.Module):

    def __init__(
        self, features, activ, norm, downsample, input_shape
    ):
        super().__init__()

        H, W  = input_shape, input_shape
        (output_features, input_features) = features

        self.downsample, _ = \
            get_downsample_x2_layer(downsample, output_features, output_features)

        
        self.block = UnetBasicBlock(input_features, output_features, activ, norm)

        self._output_shape = (output_features, H//2, W//2)

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        r = self.block(x)
        y = self.downsample(r)
        return (y, r)


class Encoder(torch.nn.Module):
    def __init__(self,
        feature_lists,                     # Mask image resolution
        image_shape,                       # Mask image channels
        mksize_list,                       # Mask image down-sample
        activ_func="relu",
        norm_layer="instance",
        down_sample="conv"
    ):
        super().__init__()

        self.layer_input = nn.Sequential(
            nn.Conv2d(
                image_shape[0], (feature_lists[0])[-1],
                kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ_func),
        )


        self.resblocks = torch.nn.ModuleList()
        for feat, img_shape in zip(feature_lists, mksize_list):
            block = UNetEncBlock(features=feat, activ=activ_func, norm=norm_layer,\
                                  downsample=down_sample, input_shape=img_shape)
            self.resblocks.append(block)

    def forward(self, img):

        img = self.layer_input(img)
        skips = []
        for block in self.resblocks:
            img, skip = block(img)
            skips.append(skip)
        skips = list(reversed(skips)) #[:-1]
        #skips.append(None)#
        return skips
    
class Generator(torch.nn.Module):
    def __init__(self, ngf, nz, cond_dim, imsize, mask_shape, ch_size):
        super(Generator, self).__init__()
        self.ngf = ngf
        
        mask_dim = mask_shape[0]-1
        self.code_sz, self.code_ch, self.mid_ch = 7, 64, 32
        self.CLIP_ch = 768
        self.fc_code = nn.Linear(nz, self.code_sz*self.code_sz*self.code_ch)
        
        self.mapping = CLIP_Adapter(self.code_ch, self.mid_ch, self.code_ch, ngf*8, self.CLIP_ch, cond_dim, 3, 1, 1, 4)

        # build GBlocks
        self.GBlocks = nn.ModuleList([])
        in_out_pairs = list(get_G_in_out_chs(ngf, imsize))
        mksize = []
        imsize = 4
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            if idx<(len(in_out_pairs)-1):
                imsize = imsize*2
            else:
                imsize = mask_shape[-1]
                
            self.GBlocks.append(P_Block(mask_dim, in_ch, out_ch, imsize))  

            if idx == len(in_out_pairs)-2:
                self.GBlocks.append(G_Block(cond_dim, out_ch, out_ch, imsize))
                              
            mksize.append(imsize)
        
        self.GBlocks.append(G_Block(cond_dim, out_ch, out_ch//2, imsize))  
        
        in_out_pairs = list(get_E_in_out_chs(ngf//2, imsize))

        # to RGB image
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(out_ch//2, ch_size, 3, 1, 1),
            )        


    def forward(self, noise, cond, mask):

        x = self.fc_code(noise).view(noise.size(0), self.code_ch, self.code_sz, self.code_sz)
        out = self.mapping(x, cond)

        for idx, GBlock in enumerate(self.GBlocks):

            if idx == len(self.GBlocks)-3 or idx == len(self.GBlocks)-1:
                out = GBlock(out, cond)
            else:    
                out = GBlock(out, mask)

        # convert to RGB image
        out = self.to_rgb(out)
        return out        

class NetG(nn.Module):
    def __init__(self, ngf, nz, cond_dim, imsize, mask_shape, ch_size):
        super(NetG, self).__init__()
       
        in_out_pairs = list(get_G_in_out_chs(ngf, imsize))
        mksize = []
        imsize = 4
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            if idx<(len(in_out_pairs)-1):
                imsize = imsize*2
            else:
                imsize = mask_shape[-1]

            mksize.append(imsize)
        
        in_out_pairs = list(get_E_in_out_chs(ngf//2, imsize))
        # build mask image encoding
        self.generator = Generator(ngf, nz, cond_dim, imsize, mask_shape, ch_size)


    def forward(self, noise, c, mask): # x=noise, c=ent_emb

        # fuse text and visual features
        out = self.generator(noise, c, mask)
        # convert to RGB image
        return out


class NetD(nn.Module):
    def __init__(self, ndf, imsize, ch_size):
        super(NetD, self).__init__()
        self.DBlocks = nn.ModuleList([
            D_Block(768, 768, 3, 1, 1, res=True, CLIP_feat=True),
            D_Block(768, 768, 3, 1, 1, res=True, CLIP_feat=True),
        ])
        self.main = D_Block(768, 512, 3, 1, 1, res=True, CLIP_feat=False)

    def forward(self, h):
        with dummy_context_mgr() as mpc:
            out = h[:,0]
            for idx in range(len(self.DBlocks)):
                out = self.DBlocks[idx](out, h[:,idx+1])
            out = self.main(out)
        return out


class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, mask_shape, num_D=2, netD_subarch='n_layer'):
        super().__init__()
        self.mask_shape = mask_shape
        for i in range(num_D):
            subnetD = self.create_single_discriminator(netD_subarch)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, netD_subarch):
        subarch = netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(label_nc=self.mask_shape[0]-1)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = False
        for name, D in self.named_children():
            out = D(input)
            if get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result

class NLayerDiscriminator(BaseNetwork):
    def __init__(self, ndf=64, n_layers_D=3, label_nc=6, output_nc=3, contain_dontcare_label=False, no_instance=True, no_ganFeat_loss=False):
        super().__init__()
        self.no_ganFeat_loss = no_ganFeat_loss
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = ndf
        input_nc = self.compute_D_input_nc(label_nc, output_nc, contain_dontcare_label, no_instance)

        norm_layer = get_nonspade_norm_layer()
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, label_nc, output_nc, contain_dontcare_label, no_instance):
        input_nc = label_nc + output_nc
        if contain_dontcare_label:
            input_nc += 1
        if not no_instance:
            input_nc += 1
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


class NetC(nn.Module):
    def __init__(self, ndf, cond_dim):
        super(NetC, self).__init__()
        self.cond_dim = cond_dim
        self.joint_conv = nn.Sequential(
            nn.Conv2d(512+512, 128, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
            )

    def forward(self, out, cond):
        with dummy_context_mgr() as mpc:
            cond = cond.view(-1, self.cond_dim, 1, 1)
            cond = cond.repeat(1, 1, 7, 7)
            h_c_code = torch.cat((out, cond), 1)
            out = self.joint_conv(h_c_code)
        return out


class M_Block(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, cond_dim, k, s, p):
        super(M_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, k, s, p)
        self.fuse1 = DFBLK(cond_dim, mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, k, s, p)
        self.fuse2 = DFBLK(cond_dim, out_ch)
        self.learnable_sc = in_ch != out_ch
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, text):
        h = self.conv1(h)
        h = self.fuse1(h, text)
        h = self.conv2(h)
        h = self.fuse2(h, text)
        return h

    def forward(self, h, c):
        return self.shortcut(h) + self.residual(h, c)

class N_Block(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, mask_dim, k, s, p):
        super(N_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, k, s, p)
        self.fuse1 = PDBLK(mid_ch, mid_ch, mask_dim)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, k, s, p)
        self.fuse2 = PDBLK(out_ch, out_ch, mask_dim)
        self.learnable_sc = in_ch != out_ch
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, text):
        h = self.conv1(h)
        h = self.fuse1(h, text)
        h = self.conv2(h)
        h = self.fuse2(h, text)
        return h

    def forward(self, h, c):
        return self.shortcut(h) + self.residual(h, c)
    
class G_Block(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, imsize):
        super(G_Block, self).__init__()
        self.imsize = imsize
        self.learnable_sc = in_ch != out_ch 
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.fuse1 = DFBLK(cond_dim, in_ch)
        self.fuse2 = DFBLK(cond_dim, out_ch)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, mask):
        h = self.fuse1(h, mask)
        h = self.c1(h)
        h = self.fuse2(h, mask)
        h = self.c2(h)
        return h


    def forward(self, h, mask):
        h = F.interpolate(h, size=(self.imsize, self.imsize))
        hh  = self.residual(h, mask)
        h = self.shortcut(h) + hh
        return h


class P_Block(nn.Module):
    def __init__(self, mask_dim, in_ch, out_ch, imsize):
        super(P_Block, self).__init__()
        self.imsize = imsize
        self.learnable_sc = in_ch != out_ch 
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.fuse1 = SPADE(in_ch, mask_dim, 'instance')
        self.fucs = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.fuse2 = SPADE(out_ch, mask_dim, 'instance')

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)
            self.norm_sc = SPADE(in_ch, mask_dim, 'instance')

    def shortcut(self, x, mask):
        if self.learnable_sc:
            x= self.norm_sc(x, mask)
            x = self.c_sc(x)
        return x

    def residual(self, h, mask):
        h = self.fuse1(h, mask)
        h = self.c1(h)
        h = self.fuse2(h, mask)
        h = self.c2(h)
        return h


    def forward(self, h, mask):
        h = F.interpolate(h, size=(self.imsize, self.imsize))
        hh  = self.residual(h, mask)
        h = self.shortcut(h, mask) + hh
        return h



class D_Block(nn.Module):
    def __init__(self, fin, fout, k, s, p, res, CLIP_feat):
        super(D_Block, self).__init__()
        self.res, self.CLIP_feat = res, CLIP_feat
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, k, s, p, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fout, fout, k, s, p, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        if self.res==True:
            self.gamma = nn.Parameter(torch.zeros(1))
        if self.CLIP_feat==True:
            self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, CLIP_feat=None):
        res = self.conv_r(x)
        if self.learned_shortcut:
            x = self.conv_s(x)
        if (self.res==True)and(self.CLIP_feat==True):
            return x + self.gamma*res + self.beta*CLIP_feat
        elif (self.res==True)and(self.CLIP_feat!=True):
            return x + self.gamma*res
        elif (self.res!=True)and(self.CLIP_feat==True):
            return x + self.beta*CLIP_feat
        else:
            return x


class DFBLK(nn.Module):
    def __init__(self, cond_dim, in_ch):
        super(DFBLK, self).__init__()
        self.affine0 = Affine(cond_dim, in_ch)
        self.affine1 = Affine(cond_dim, in_ch)

    def forward(self, x, y=None):
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        return h


class PDBLK(nn.Module):
    def __init__(self, in_ch, out_ch, mask_dim):
        super(PDBLK, self).__init__()
        self.sc0 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.sc1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        
        self.norm0 = SPADE(in_ch, mask_dim, "batch", nhidden_dim=32)
        self.norm1 = SPADE(out_ch, mask_dim, "batch", nhidden_dim=32)

    def forward(self, x, y=None):
        h = self.norm0(x, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.sc0(h)
        h = self.norm1(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.sc1(h)
        
        return h


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Affine(nn.Module):
    def __init__(self, cond_dim, num_features):
        super(Affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)        

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


def get_G_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    channel_nums = channel_nums[::-1]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs

def get_E_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 16) for idx in range(layer_num)]
    channel_nums = channel_nums[::-1]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs

def get_D_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs
    
