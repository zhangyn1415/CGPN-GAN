import copy
from itertools import repeat
import re
import torch.nn.functional as F
from torch import nn
import torch
import os
import numpy as np
from torch.nn import init
import torch.nn.utils.spectral_norm as spectral_norm

class ACE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, ACE_Name=None, status='train', spade_params=None, use_rgb=True):
        super().__init__()

        self.ACE_Name = ACE_Name
        self.status = status
        self.save_npy = True
        self.Spade = SPADE(*spade_params)
        self.use_rgb = use_rgb
        self.style_length = 512
        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise_var = nn.Parameter(torch.zeros(norm_nc), requires_grad=True)


        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        pw = ks // 2

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.


        if self.use_rgb:
            self.create_gamma_beta_fc_layers()

            self.conv_gamma = nn.Conv2d(self.style_length, norm_nc, kernel_size=ks, padding=pw)
            self.conv_beta = nn.Conv2d(self.style_length, norm_nc, kernel_size=ks, padding=pw)




    def forward(self, x, segmap, style_codes=None, obj_dic=None):

        # Part 1. generate parameter-free normalized activations
        added_noise = (torch.randn(x.shape[0], x.shape[3], x.shape[2], 1).cuda() * self.noise_var).transpose(1, 3)
        normalized = self.param_free_norm(x + added_noise)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        if self.use_rgb:
            [b_size, f_size, h_size, w_size] = normalized.shape
            middle_avg = torch.zeros((b_size, self.style_length, h_size, w_size), device=normalized.device)

            if self.status == 'UI_mode':
                ############## hard coding

                for i in range(1):
                    for j in range(segmap.shape[1]):

                        component_mask_area = torch.sum(segmap.bool()[i, j])

                        if component_mask_area > 0:
                            if obj_dic is None:
                                print('wrong even it is the first input')
                            else:
                                style_code_tmp = obj_dic[str(j)]['ACE']

                                middle_mu = F.relu(self.__getattr__('fc_mu' + str(j))(style_code_tmp))
                                component_mu = middle_mu.reshape(self.style_length, 1).expand(self.style_length,component_mask_area)

                                middle_avg[i].masked_scatter_(segmap.bool()[i, j], component_mu)

            else:

                for i in range(b_size):
                    for j in range(segmap.shape[1]):
                        component_mask_area = torch.sum(segmap.bool()[i, j])

                        if component_mask_area > 0:


                            middle_mu = F.relu(self.__getattr__('fc_mu' + str(j))(style_codes[i][j]))
                            component_mu = middle_mu.reshape(self.style_length, 1).expand(self.style_length, component_mask_area)

                            middle_avg[i].masked_scatter_(segmap.bool()[i, j], component_mu)


                            if self.status == 'test' and self.save_npy and self.ACE_Name=='up_2_ACE_0':
                                tmp = style_codes[i][j].cpu().numpy()
                                dir_path = 'styles_test'

                                ############### some problem with obj_dic[i]

                                im_name = os.path.basename(obj_dic[i])
                                folder_path = os.path.join(dir_path, 'style_codes', im_name, str(j))
                                if not os.path.exists(folder_path):
                                    os.makedirs(folder_path)

                                style_code_path = os.path.join(folder_path, 'ACE.npy')
                                np.save(style_code_path, tmp)


            gamma_avg = self.conv_gamma(middle_avg)
            beta_avg = self.conv_beta(middle_avg)


            gamma_spade, beta_spade = self.Spade(segmap)

            gamma_alpha = F.sigmoid(self.blending_gamma)
            beta_alpha = F.sigmoid(self.blending_beta)

            gamma_final = gamma_alpha * gamma_avg + (1 - gamma_alpha) * gamma_spade
            beta_final = beta_alpha * beta_avg + (1 - beta_alpha) * beta_spade
            out = normalized * (1 + gamma_final) + beta_final
        else:
            gamma_spade, beta_spade = self.Spade(segmap)
            gamma_final = gamma_spade
            beta_final = beta_spade
            out = normalized * (1 + gamma_final) + beta_final

        return out





    def create_gamma_beta_fc_layers(self):


        ###################  These codes should be replaced with torch.nn.ModuleList

        style_length = self.style_length

        self.fc_mu0 = nn.Linear(style_length, style_length)
        self.fc_mu1 = nn.Linear(style_length, style_length)
        self.fc_mu2 = nn.Linear(style_length, style_length)
        self.fc_mu3 = nn.Linear(style_length, style_length)
        self.fc_mu4 = nn.Linear(style_length, style_length)
        self.fc_mu5 = nn.Linear(style_length, style_length)
        self.fc_mu6 = nn.Linear(style_length, style_length)
        self.fc_mu7 = nn.Linear(style_length, style_length)
        self.fc_mu8 = nn.Linear(style_length, style_length)
        self.fc_mu9 = nn.Linear(style_length, style_length)
        self.fc_mu10 = nn.Linear(style_length, style_length)
        self.fc_mu11 = nn.Linear(style_length, style_length)
        self.fc_mu12 = nn.Linear(style_length, style_length)
        self.fc_mu13 = nn.Linear(style_length, style_length)
        self.fc_mu14 = nn.Linear(style_length, style_length)
        self.fc_mu15 = nn.Linear(style_length, style_length)
        self.fc_mu16 = nn.Linear(style_length, style_length)
        self.fc_mu17 = nn.Linear(style_length, style_length)
        self.fc_mu18 = nn.Linear(style_length, style_length)




class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, norm, pad_ks=3, nhidden_dim=128):
        super().__init__()

        if norm == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif norm == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif norm == 'group':
            self.param_free_norm = nn.GroupNorm(32, norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % norm)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = nhidden_dim

        pw = pad_ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=pad_ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=pad_ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=pad_ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
 
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
                'To see the architecture, do print(network).'
                % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)
     
def extract_name_kwargs(obj):
    if isinstance(obj, dict):
        obj    = copy.copy(obj)
        name   = obj.pop('name')
        kwargs = obj
    else:
        name   = obj
        kwargs = {}

    return (name, kwargs)

def get_norm_layer(norm, features):
    name, kwargs = extract_name_kwargs(norm)

    if name is None:
        return nn.Identity(**kwargs)

    if name == 'layer':
        return nn.LayerNorm((features,), **kwargs)

    if name == 'batch':
        return nn.BatchNorm2d(features, **kwargs)

    if name == 'instance':
        return nn.InstanceNorm2d(features, **kwargs)

    raise ValueError("Unknown Layer: '%s'" % name)

# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(norm_type='spectralinstance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer

def get_activ_layer(activ):
    name, kwargs = extract_name_kwargs(activ)

    if (name is None) or (name == 'linear'):
        return nn.Identity()

    if name == 'gelu':
        return nn.GELU(**kwargs)

    if name == 'relu':
        return nn.ReLU(inplace = True, **kwargs)

    if name == 'leakyrelu':
        return nn.LeakyReLU(inplace = True, **kwargs)

    if name == 'tanh':
        return nn.Tanh()

    if name == 'sigmoid':
        return nn.Sigmoid()

    raise ValueError("Unknown activation: '%s'" % name)


def get_downsample_x2_conv2_layer(input_features, output_features, **kwargs):
    return (
        nn.Conv2d(input_features, output_features, kernel_size = 2, stride = 2, **kwargs),
        output_features
    )

def get_downsample_x2_layer(layer, features, output_feats):
    name, kwargs = extract_name_kwargs(layer)

    if name == 'conv':
        return get_downsample_x2_conv2_layer(features, output_feats, **kwargs)

    if name == 'avgpool':
        return (nn.AvgPool2d(kernel_size = 2, stride = 2, **kwargs), features)

    if name == 'maxpool':
        return (nn.MaxPool2d(kernel_size = 2, stride = 2, **kwargs), features)

    raise ValueError("Unknown Downsample Layer: '%s'" % name)