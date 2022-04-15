import torch
from torch import nn
import torch.nn.functional as F

import os
import numpy as np
import itertools
from collections import OrderedDict

import util.util as util
import util.index as index
import models.networks as networks
import models.losses as losses
from models import arch

from .base_model import BaseModel
from PIL import Image
from os.path import join


def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy


class EdgeMap(nn.Module):
    def __init__(self, scale=1):
        super(EdgeMap, self).__init__()
        self.scale = scale
        self.requires_grad = False

    def forward(self, img):
        img = img / self.scale

        N, C, H, W = img.shape
        gradX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        
        gradx = (img[...,1:,:] - img[...,:-1,:]).abs().sum(dim=1, keepdim=True)
        grady = (img[...,1:] - img[...,:-1]).abs().sum(dim=1, keepdim=True)

        gradX[...,:-1,:] += gradx
        gradX[...,1:,:] += gradx
        gradX[...,1:-1,:] /= 2

        gradY[...,:-1] += grady
        gradY[...,1:] += grady
        gradY[...,1:-1] /= 2

        # edge = (gradX + gradY) / 2
        edge = (gradX + gradY)

        return edge


class ERRNetBase(BaseModel):
    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            util.set_opt_param(optimizer, 'weight_decay', self.opt.wd)

    def set_input(self, data, mode='train'):
        target_t = None
        target_r = None
        data_name = None
        mode = mode.lower()
        if mode == 'train':
            input, target_t, target_r = data['input'], data['target_t'], data['target_r']
        elif mode == 'eval':
            input, target_t, target_r, data_name = data['input'], data['target_t'], data['target_r'], data['fn']
        elif mode == 'test':
            input, data_name = data['input'], data['fn']
        else:
            raise NotImplementedError('Mode [%s] is not implemented' % mode)
        
        if len(self.gpu_ids) > 0:  # transfer data into gpu
            input = input.to(device=self.gpu_ids[0])
            if target_t is not None:
                target_t = target_t.to(device=self.gpu_ids[0])
            if target_r is not None:
                target_r = target_r.to(device=self.gpu_ids[0])                
        
        self.input = input
        
        self.input_edge = self.edge_map(self.input)
        self.target_t = target_t
        self.data_name = data_name

        self.issyn = False if 'real' in data else True
        self.aligned = False if 'unaligned' in data else True
        
        if target_t is not None:            
            self.target_edge = self.edge_map(self.target_t)         
            
    def eval(self, data, savedir=None, suffix=None, pieapp=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'eval')

        with torch.no_grad():
            self.forward()

            output_i = tensor2im(self.output_i)
            target = tensor2im(self.target_t)

            if self.aligned:
                res = index.quality_assess(output_i, target)
            else:
                res = {}

            if savedir is not None:
                if self.data_name is not None:
                    name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
                    if not os.path.exists(join(savedir, name)):
                        os.makedirs(join(savedir, name))
                    if suffix is not None:
                        Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name,'{}_{}.png'.format(self.opt.name, suffix)))
                    else:
                        Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name, '{}.png'.format(self.opt.name)))
                    Image.fromarray(target.astype(np.uint8)).save(join(savedir, name, 't_label.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, name, 'm_input.png'))
                else:
                    if not os.path.exists(join(savedir, 'transmission_layer')):
                        os.makedirs(join(savedir, 'transmission_layer'))
                        os.makedirs(join(savedir, 'blended'))
                    Image.fromarray(target.astype(np.uint8)).save(join(savedir, 'transmission_layer', str(self._count)+'.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, 'blended', str(self._count)+'.png'))
                    self._count += 1

            return res

    def test(self, data, savedir=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'test')

        if self.data_name is not None and savedir is not None:
            name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
            if not os.path.exists(join(savedir, name)):
                os.makedirs(join(savedir, name))

            if os.path.exists(join(savedir, name, '{}.png'.format(self.opt.name))):
                return 
        
        with torch.no_grad():
            output_i = self.forward()
            output_i = tensor2im(output_i)
            if self.data_name is not None and savedir is not None:                
                Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name, '{}.png'.format(self.opt.name)))
                Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, name, 'm_input.png'))


class ERRNetModel(ERRNetBase):
    def name(self):
        return 'errnet'
        
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_D = None

    def print_network(self):
        print('--------------------- Model ---------------------')
        print('##################### NetG #####################')
        networks.print_network(self.net_i)
        if self.isTrain and self.opt.lambda_gan > 0:
            print('##################### NetD #####################')
            networks.print_network(self.netD)

    def _eval(self):
        self.net_i.eval()

    def _train(self):
        self.net_i.train()

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        in_channels = 3
        self.vgg = None
        
        if opt.hyper:
            self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
            in_channels += 1472
        
        self.net_i = arch.__dict__[self.opt.inet](in_channels, 3).to(self.device)
        networks.init_weights(self.net_i, init_type=opt.init_type) # using default initialization as EDSR
        self.edge_map = EdgeMap(scale=1).to(self.device)

        if self.isTrain:
            # define loss functions
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['t_vgg'] = vggloss

            cxloss = losses.ContentLoss()
            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(self.vgg, weights=[0.1], indices=[opt.vgg_layer]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1], indices=[8, 13, 22]))
            elif opt.unaligned_loss == 'mse':
                cxloss.initialize(nn.MSELoss())
            elif opt.unaligned_loss == 'ctx_vgg':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1,0.1], indices=[8, 13, 22, 31], criterions=[losses.CX_loss]*3+[nn.L1Loss()]))
            else:
                raise NotImplementedError

            self.loss_dic['t_cx'] = cxloss

            # Define discriminator
            # if self.opt.lambda_gan > 0:
            self.netD = networks.define_D(opt, 3)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr, betas=(0.9, 0.999))
            self._init_optimizer([self.optimizer_D])

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.net_i.parameters(), 
                lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G])

        if opt.resume:
            self.load(self, opt.resume_epoch)
        
        if opt.no_verbose is False:
            self.print_network()

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        self.loss_D, self.pred_fake, self.pred_real = self.loss_dic['gan'].get_loss(
            self.netD, self.input, self.output_i, self.target_t)

        (self.loss_D*self.opt.lambda_gan).backward(retain_graph=True)

    def backward_G(self):
        # Make it a tiny bit faster
        for p in self.netD.parameters():
            p.requires_grad = False
        
        self.loss_G = 0
        self.loss_CX = None
        self.loss_icnn_pixel = None
        self.loss_icnn_vgg = None
        self.loss_G_GAN = None

        if self.opt.lambda_gan > 0:
            self.loss_G_GAN = self.loss_dic['gan'].get_g_loss(
                self.netD, self.input, self.output_i, self.target_t)
            self.loss_G += self.loss_G_GAN*self.opt.lambda_gan
        
        if self.aligned:
            self.loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(
                self.output_i, self.target_t)
            
            self.loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(
                self.output_i, self.target_t)

            self.loss_G += self.loss_icnn_pixel+self.loss_icnn_vgg*self.opt.lambda_vgg
        else:
            self.loss_CX = self.loss_dic['t_cx'].get_loss(self.output_i, self.target_t)
            
            self.loss_G += self.loss_CX
        
        self.loss_G.backward()

    def forward(self):
        # without edge
        input_i = self.input

        if self.vgg is not None:
            hypercolumn = self.vgg(self.input)
            _, C, H, W = self.input.shape
            hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for feature in hypercolumn]
            input_i = [input_i]
            input_i.extend(hypercolumn)
            input_i = torch.cat(input_i, dim=1)

        output_i = self.net_i(input_i)

        self.output_i = output_i

        return output_i
        
    def optimize_parameters(self):
        self._train()
        self.forward()

        if not self.opt.freeze_D and self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_icnn_pixel is not None:
            ret_errors['IPixel'] = self.loss_icnn_pixel.item()
        if self.loss_icnn_vgg is not None:
            ret_errors['VGG'] = self.loss_icnn_vgg.item()
            
        if self.loss_D is not None:
            ret_errors['D'] = self.loss_D.item()

        if self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()

        if self.loss_CX is not None:
            ret_errors['CX'] = self.loss_CX.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_i'] = tensor2im(self.output_i).astype(np.uint8)        
        ret_visuals['target'] = tensor2im(self.target_t).astype(np.uint8)
        ret_visuals['residual'] = tensor2im((self.input - self.output_i)).astype(np.uint8)

        return ret_visuals       

    @staticmethod
    def load(model, resume_epoch=None):
        icnn_path = model.opt.icnn_path
        if icnn_path is None:
            icnn_path = util.get_model_list(model.save_dir, model.name(), epoch=resume_epoch)

        state_dict = torch.load(icnn_path, map_location=torch.device('cuda:{}'.format(model.opt.gpu_ids[0])))
        model.epoch = state_dict['epoch']
        model.iterations = state_dict['iterations']
        model.net_i.load_state_dict(state_dict['icnn'])

        if model.isTrain:
            model.optimizer_G.load_state_dict(state_dict['opt_g'])
            if 'netD' in state_dict:
                print('Resume netD ...')
                model.netD.load_state_dict(state_dict['netD'])
                model.optimizer_D.load_state_dict(state_dict['opt_d'])
            
        print('Resume from epoch %d, iteration %d' % (model.epoch, model.iterations))
        return state_dict

    def state_dict(self):
        state_dict = {
            'icnn': self.net_i.state_dict(),
            'opt_g': self.optimizer_G.state_dict(), 
            'epoch': self.epoch, 'iterations': self.iterations
        }

        if self.opt.lambda_gan > 0:
            state_dict.update({
                'opt_d': self.optimizer_D.state_dict(),
                'netD': self.netD.state_dict(),
            })

        return state_dict


class ERRNetALWModel(ERRNetBase):
    def name(self):
        # errnet + Auto Loss Weighing (alw)
        return 'errnet_alw'
        
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_D = None

    def print_network(self):
        print('--------------------- Model ---------------------')
        print('##################### NetG #####################')
        networks.print_network(self.net_i)
        if self.isTrain and self.opt.lambda_gan > 0:
            print('##################### NetD #####################')
            networks.print_network(self.netD)

    def _eval(self):
        self.net_i.eval()

    def _train(self):
        self.net_i.train()

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        in_channels = 3
        self.vgg = None
        
        if opt.hyper:
            self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
            in_channels += 1472
        
        self.net_i = arch.__dict__[self.opt.inet](in_channels, 3).to(self.device)
        networks.init_weights(self.net_i, init_type=opt.init_type) # using default initialization as EDSR
        self.edge_map = EdgeMap(scale=1).to(self.device)

        if self.isTrain:
            # define loss functions
            self.loss_dic = {}
            loss_list = [loss for loss in opt.pixel_loss.split("+")]
            for loss in loss_list:
                if loss == 'mse':
                    mse_loss = losses.ContentLoss()
                    mse_loss.initialize(nn.MSELoss())
                    self.loss_dic['mse'] = mse_loss
                elif loss == 'grad':
                    gradient_loss = losses.ContentLoss()
                    gradient_loss.initialize(losses.GradientLoss())
                    self.loss_dic['grad'] = gradient_loss
                elif loss == 'ms_ssim_l1':
                    ms_ssim_loss = losses.ContentLoss()
                    ms_ssim_loss.initialize(losses.MS_SSIM_L1_Loss())
                    self.loss_dic['ms_ssim'] = ms_ssim_loss
                elif loss == 'highpass':
                    highpass_loss = losses.ContentLoss()
                    highpass_loss.initialize(losses.HighpassLoss())
                    self.loss_dic['highpass'] = highpass_loss
                else:
                    raise NotImplementedError('pixel loss {} is not implemented.'.format(loss))
            
            if opt.gan_type == 'sgan' or opt.gan_type == 'gan':
                disc_loss = losses.DiscLoss()
            elif opt.gan_type == 'rsgan':
                disc_loss = losses.DiscLossR()
            elif opt.gan_type == 'rasgan':
                disc_loss = losses.DiscLossRa()
            else:
                raise ValueError("GAN [%s] not recognized." % opt.gan_type)

            disc_loss.initialize(opt, self.Tensor)
            self.loss_dic['gan'] = disc_loss

            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['t_vgg'] = vggloss

            # initialize uncertainty parameters
            uncertainty_params = nn.ParameterDict()
            for key, val in self.loss_dic.items():
                uncertainty_params[key] = nn.Parameter(-1 * torch.ones(1).to(self.device))
            self.net_i.uncertainty_params = uncertainty_params

            # self.uncertainty_params['gan'] = nn.Parameter(4.6 * torch.ones(1))

            # at this moment do not consider unaligned losses
            cxloss = losses.ContentLoss()
            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(self.vgg, weights=[0.1], indices=[opt.vgg_layer]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1], indices=[8, 13, 22]))
            elif opt.unaligned_loss == 'mse':
                cxloss.initialize(nn.MSELoss())
            elif opt.unaligned_loss == 'ctx_vgg':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1,0.1], indices=[8, 13, 22, 31], criterions=[losses.CX_loss]*3+[nn.L1Loss()]))
            else:
                raise NotImplementedError

            self.loss_dic['t_cx'] = cxloss

            # Define discriminator
            # if self.opt.lambda_gan > 0:
            self.netD = networks.define_D(opt, 3)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr, betas=(0.9, 0.999))
            self._init_optimizer([self.optimizer_D])

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.net_i.parameters(), 
                lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G])

        if opt.resume:

            self.load(self, opt.resume_epoch)
        
        if opt.no_verbose is False:
            self.print_network()

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        self.loss_D, self.pred_fake, self.pred_real = self.loss_dic['gan'].get_loss(
            self.netD, self.input, self.output_i, self.target_t)

        (self.loss_D * torch.exp(-self.net_i.uncertainty_params['gan'])).backward(retain_graph=True)

    def backward_G(self):
        # Make it a tiny bit faster
        for p in self.netD.parameters():
            p.requires_grad = False
        
        self.loss_G = 0
        self.loss_CX = None
        self.loss_icnn_pixel = None
        self.loss_icnn_vgg = None
        self.loss_G_GAN = None

        if self.opt.lambda_gan > 0:
            self.loss_G_GAN = self.loss_dic['gan'].get_g_loss(
                self.netD, self.input, self.output_i, self.target_t)
            self.loss_G += self.loss_G_GAN * torch.exp(-self.net_i.uncertainty_params['gan'])
            self.loss_G += self.net_i.uncertainty_params['gan']
        
        if self.aligned:
            self.loss_icnn_pixel = 0
            for key, val in self.loss_dic.items():
                if key == 't_vgg' or key == 'gan' or key == 't_cx':
                    continue
                self.loss_icnn_pixel += self.loss_dic[key].get_loss(self.output_i, self.target_t) * torch.exp(-self.net_i.uncertainty_params[key])
                self.loss_G += self.net_i.uncertainty_params[key]
            
            self.loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(
                self.output_i, self.target_t)

            self.loss_G += self.loss_icnn_pixel + self.loss_icnn_vgg * torch.exp(-self.net_i.uncertainty_params['t_vgg'])
            self.loss_G += self.net_i.uncertainty_params['t_vgg']
        else:
            self.loss_CX = self.loss_dic['t_cx'].get_loss(self.output_i, self.target_t)
            self.loss_G += self.loss_CX * torch.exp(-self.net_i.uncertainty_params['t_cx'])
            self.loss_G += self.net_i.uncertainty_params['t_cx']

        self.loss_G *= 0.5
        self.loss_G.backward()

    def forward(self):
        # without edge
        input_i = self.input

        if self.vgg is not None:
            hypercolumn = self.vgg(self.input)
            _, C, H, W = self.input.shape
            hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for feature in hypercolumn]
            input_i = [input_i]
            input_i.extend(hypercolumn)
            input_i = torch.cat(input_i, dim=1)

        output_i = self.net_i(input_i)

        self.output_i = output_i

        return output_i
        
    def optimize_parameters(self):
        self._train()
        self.forward()

        if not self.opt.freeze_D and self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_icnn_pixel is not None:
            ret_errors['IPixel'] = self.loss_icnn_pixel.item()
        if self.loss_icnn_vgg is not None:
            ret_errors['VGG'] = self.loss_icnn_vgg.item()
            
        if self.loss_D is not None:
            ret_errors['D'] = self.loss_D.item()

        if self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()

        if self.loss_CX is not None:
            ret_errors['CX'] = self.loss_CX.item()

        return ret_errors
    
    def get_current_uncertainty_params(self):
        param_dict = {}
        for key, val in self.net_i.uncertainty_params.items():
            param_dict[key] = torch.exp(-val).item()
            param_dict[key] *= 0.5
        return param_dict

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_i'] = tensor2im(self.output_i).astype(np.uint8)        
        ret_visuals['target'] = tensor2im(self.target_t).astype(np.uint8)
        ret_visuals['residual'] = tensor2im((self.input - self.output_i)).astype(np.uint8)

        return ret_visuals       

    @staticmethod
    def load(model, resume_epoch=None):
        icnn_path = model.opt.icnn_path
        if icnn_path is None:
            icnn_path = util.get_model_list(model.save_dir, model.name(), epoch=resume_epoch)

        state_dict = torch.load(icnn_path, map_location=torch.device('cuda:{}'.format(model.opt.gpu_ids[0])))
        model.epoch = state_dict['epoch']
        model.iterations = state_dict['iterations']

        for k in list(state_dict['icnn']):
            if "uncertainty_params" in k:
                del state_dict['icnn'][k]
        model.net_i.load_state_dict(state_dict['icnn'])

        if model.isTrain:
            model.optimizer_G.load_state_dict(state_dict['opt_g'])
            if 'netD' in state_dict:
                print('Resume netD ...')
                model.netD.load_state_dict(state_dict['netD'])
                model.optimizer_D.load_state_dict(state_dict['opt_d'])
            
        print('Resume from epoch %d, iteration %d' % (model.epoch, model.iterations))
        return state_dict

    def state_dict(self):
        state_dict = {
            'icnn': self.net_i.state_dict(),
            'opt_g': self.optimizer_G.state_dict(), 
            'epoch': self.epoch, 'iterations': self.iterations
        }

        if self.opt.lambda_gan > 0:
            state_dict.update({
                'opt_d': self.optimizer_D.state_dict(),
                'netD': self.netD.state_dict(),
            })

        return state_dict


class NetworkWrapper(ERRNetBase):
    # You can use this class to wrap other module into our training framework (\eg BDN module)
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def print_network(self):
        print('--------------------- NetworkWrapper ---------------------')
        networks.print_network(self.net)

    def _eval(self):
        self.net.eval()

    def _train(self):
        self.net.train()

    def initialize(self, opt, net):
        BaseModel.initialize(self, opt)
        self.net = net.to(self.device)
        self.edge_map = EdgeMap(scale=1).to(self.device)
        
        if self.isTrain:
            # define loss functions
            self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['t_vgg'] = vggloss

            cxloss = losses.ContentLoss()
            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(self.vgg, weights=[0.1], indices=[31]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1], indices=[8, 13, 22]))
            elif opt.unaligned_loss == 'mse':
                cxloss.initialize(nn.MSELoss())
            elif opt.unaligned_loss == 'ctx_vgg':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1,0.1], indices=[8, 13, 22, 31], criterions=[losses.CX_loss]*3+[nn.L1Loss()]))
                
            else:
                raise NotImplementedError            
            
            self.loss_dic['t_cx'] = cxloss

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.net.parameters(), 
                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G])

            # define discriminator
            # if self.opt.lambda_gan > 0:
            self.netD = networks.define_D(opt, 3)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
            self._init_optimizer([self.optimizer_D])
        
        if opt.no_verbose is False:
            self.print_network()

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        self.loss_D, self.pred_fake, self.pred_real = self.loss_dic['gan'].get_loss(
            self.netD, self.input, self.output_i, self.target_t)

        (self.loss_D*self.opt.lambda_gan).backward(retain_graph=True)
        
    def backward_G(self):
        for p in self.netD.parameters():
            p.requires_grad = False
                    
        self.loss_G = 0
        self.loss_CX = None
        self.loss_icnn_pixel = None
        self.loss_icnn_vgg = None
        self.loss_G_GAN = None

        if self.opt.lambda_gan > 0:
            self.loss_G_GAN = self.loss_dic['gan'].get_g_loss(
                self.netD, self.input, self.output_i, self.target_t) #self.pred_real.detach())
            self.loss_G += self.loss_G_GAN*self.opt.lambda_gan
                
        if self.aligned:
            self.loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(
                self.output_i, self.target_t)
            
            self.loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(
                self.output_i, self.target_t)

            # self.loss_G += self.loss_icnn_pixel
            self.loss_G += self.loss_icnn_pixel+self.loss_icnn_vgg*self.opt.lambda_vgg
            # self.loss_G += self.loss_fm * self.opt.lambda_vgg
        else:
            self.loss_CX = self.loss_dic['t_cx'].get_loss(self.output_i, self.target_t)
            
            self.loss_G += self.loss_CX
        
        self.loss_G.backward()

    def forward(self):
        raise NotImplementedError
        
    def optimize_parameters(self):
        self._train()
        self.forward()

        if self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_icnn_pixel is not None:
            ret_errors['IPixel'] = self.loss_icnn_pixel.item()
        if self.loss_icnn_vgg is not None:
            ret_errors['VGG'] = self.loss_icnn_vgg.item()
        if self.opt.lambda_gan > 0 and self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()
            ret_errors['D'] = self.loss_D.item()
        if self.loss_CX is not None:
            ret_errors['CX'] = self.loss_CX.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_i'] = tensor2im(self.output_i).astype(np.uint8)        
        ret_visuals['target'] = tensor2im(self.target_t).astype(np.uint8)
        ret_visuals['residual'] = tensor2im((self.input - self.output_i)).astype(np.uint8)
        return ret_visuals

    def state_dict(self):
        state_dict = self.net.state_dict()
        return state_dict
