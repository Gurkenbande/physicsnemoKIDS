from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
import torch.nn.functional as F

from models.select_network import define_G, define_D
from models.model_base import ModelBase
from models.loss import GANLoss, PerceptualLoss
from models.loss_ssim import SSIMLoss
from utils.utils_model import test_mode
from physicsnemo.models.diffusion.song_unet import SongUNetPosEmbd

class ModelGAN(ModelBase):
    """Train with pixel-VGG-GAN loss"""
    def __init__(self, opt):
        super(ModelGAN, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.pos_channels = int(self.opt["netG"].get("pos_channels", 0))
        self.tp_channel_idx = int(self.opt_train.get("tp_channel_idx", 0))
        self.train_only_tp_channel = bool(
            self.opt_train.get("train_only_tp_channel", True)
        )
        if self.train_only_tp_channel:
            d_in_nc = int(self.opt.get("netD", {}).get("in_nc", 1))
            if d_in_nc != 1:
                raise ValueError(
                    "train_only_tp_channel=True requires netD.in_nc=1, "
                    f"but got netD.in_nc={d_in_nc}."
                )
        self._pos_embd_cache = {}
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.is_train:
            self.netD = define_D(opt)
            self.netD = self.model_to_device(self.netD)
            if self.opt_train['E_decay'] > 0:
                self.netE = define_G(opt).to(self.device).eval()
        sf = opt['scale']
        self.pool = torch.nn.AvgPool2d(kernel_size=sf, stride=sf)
    
    def _get_positional_embedding(self, height, width, device, dtype):
        if self.pos_channels <= 0:
            return None
        key = (height, width, str(device), str(dtype))
        if key in self._pos_embd_cache:
            return self._pos_embd_cache[key]

        dummy = type("Dummy", (), {})()
        dummy.N_grid_channels = self.pos_channels
        dummy.gridtype = "sinusoidal"
        dummy.img_shape_y = height
        dummy.img_shape_x = width
        grid = SongUNetPosEmbd._get_positional_embedding(dummy)
        if grid is None:
            raise ValueError("Positional embedding grid could not be created.")
        grid = grid.to(device=device, dtype=dtype)
        self._pos_embd_cache[key] = grid
        return grid

    def _compute_feature_loss_all_channels(self, pred, target):
        """Compute perceptual loss across all output channels.

        VGG-based perceptual loss expects 3-channel inputs. For multi-channel
        weather outputs, compute perceptual loss channel-wise by repeating each
        single channel to pseudo-RGB and averaging over channels.
        """
        # Optional memory saver for perceptual loss.
        max_size = int(self.opt_train.get("F_loss_max_size", 224) or 0)
        if (
            max_size > 0
            and (pred.shape[-2] > max_size or pred.shape[-1] > max_size)
        ):
            pred = F.interpolate(pred, size=(max_size, max_size), mode="area")
            target = F.interpolate(target, size=(max_size, max_size), mode="area")

        amp_enabled = bool(self.opt_train.get("F_loss_amp", True)) and pred.is_cuda
        num_channels = pred.shape[1]
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            if num_channels == 3:
                return self.F_lossfn(pred, target)

            loss = 0.0
            for ch in range(num_channels):
                pred_ch = pred[:, ch : ch + 1].repeat(1, 3, 1, 1)
                target_ch = target[:, ch : ch + 1].repeat(1, 3, 1, 1)
                loss = loss + self.F_lossfn(pred_ch, target_ch)
            return loss / num_channels

    def _get_tp_idx(self, tensor):
        num_channels = int(tensor.shape[1])
        if num_channels <= 0:
            raise ValueError("Expected at least one output channel.")
        return max(0, min(self.tp_channel_idx, num_channels - 1))

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.netD.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # load pre-trained G and D model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'])
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'])
            else:
                print('Copying model for E')
                self.update_E(0)
            self.netE.eval()

        load_path_D = self.opt['path']['pretrained_netD']
        if self.opt['is_train'] and load_path_D is not None:
            print('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, strict=self.opt_train['D_param_strict'])

    # ----------------------------------------
    # load optimizerG and optimizerD
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)
        load_path_optimizerD = self.opt['path']['pretrained_optimizerD']
        if load_path_optimizerD is not None and self.opt_train['D_optimizer_reuse']:
            print('Loading optimizerD [{:s}] ...'.format(load_path_optimizerD))
            self.load_optimizer(load_path_optimizerD, self.D_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        self.save_network(self.save_dir, self.netD, 'D', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)
        if self.opt_train['D_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.D_optimizer, 'optimizerD', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        self.global_lossfn_weight = self.opt_train['global_lossfn_weight']
        if self.global_lossfn_weight > 0:
             self.global_lossfn = nn.L1Loss().to(self.device)
        self.M_lossfn_weight = self.opt_train['M_lossfn_weight']
        if self.opt_train['M_lossfn_weight'] > 0:
             self.M_lossfn = nn.BCELoss().to(self.device)
        # ------------------------------------
        # 1) G_loss
        # ------------------------------------
        if self.opt_train['G_lossfn_weight'] > 0:
            G_lossfn_type = self.opt_train['G_lossfn_type']
            if G_lossfn_type == 'l1':
                self.G_lossfn = nn.L1Loss().to(self.device)
            elif G_lossfn_type == 'l2':
                self.G_lossfn = nn.MSELoss().to(self.device)
            elif G_lossfn_type == 'l2sum':
                self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
            elif G_lossfn_type == 'ssim':
                self.G_lossfn = SSIMLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
            self.G_lossfn_weight = self.opt_train['G_lossfn_weight']
        else:
            print('Do not use pixel loss.')
            self.G_lossfn = None

        # ------------------------------------
        # 2) F_loss
        # ------------------------------------
        if self.opt_train['F_lossfn_weight'] > 0:
            F_feature_layer = self.opt_train['F_feature_layer']
            F_weights = self.opt_train['F_weights']
            F_lossfn_type = self.opt_train['F_lossfn_type']
            F_use_input_norm = self.opt_train['F_use_input_norm']
            F_use_range_norm = self.opt_train['F_use_range_norm']
            if self.opt['dist']:
                self.F_lossfn = PerceptualLoss(feature_layer=F_feature_layer, weights=F_weights, lossfn_type=F_lossfn_type, use_input_norm=F_use_input_norm, use_range_norm=F_use_range_norm).to(self.device)
            else:
                self.F_lossfn = PerceptualLoss(feature_layer=F_feature_layer, weights=F_weights, lossfn_type=F_lossfn_type, use_input_norm=F_use_input_norm, use_range_norm=F_use_range_norm)
                self.F_lossfn.vgg = self.model_to_device(self.F_lossfn.vgg)
                self.F_lossfn.lossfn = self.F_lossfn.lossfn.to(self.device)
            self.F_lossfn_weight = self.opt_train['F_lossfn_weight']
        else:
            print('Do not use feature loss.')
            self.F_lossfn = None

        # ------------------------------------
        # 3) D_loss
        # ------------------------------------
        self.D_lossfn = GANLoss(self.opt_train['gan_type'], 1.0, 0.0).to(self.device)
        self.D_lossfn_weight = self.opt_train['D_lossfn_weight']

        self.D_update_ratio = self.opt_train['D_update_ratio'] if self.opt_train['D_update_ratio'] else 1
        self.D_init_iters = self.opt_train['D_init_iters'] if self.opt_train['D_init_iters'] else 0

    # ----------------------------------------
    # define optimizer, G and D
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))

        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)
        self.D_optimizer = Adam(self.netD.parameters(), lr=self.opt_train['D_optimizer_lr'], weight_decay=0)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))
        self.schedulers.append(lr_scheduler.MultiStepLR(self.D_optimizer,
                                                        self.opt_train['D_scheduler_milestones'],
                                                        self.opt_train['D_scheduler_gamma']
                                                        ))

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if self.pos_channels > 0:
            pos = self._get_positional_embedding(
                self.L.shape[-2],
                self.L.shape[-1],
                self.L.device,
                self.L.dtype,
            )
            pos = pos.unsqueeze(0).expand(self.L.shape[0], -1, -1, -1)
            self.L = torch.cat([self.L, pos], dim=1)
        if need_H:
            self.H = data['H'].to(self.device)
        #self.mask_label = torch.nan_to_num(self.pool(self.H) / self.L, nan=1.0)
        #self.mask_label = torch.where(self.mask_label<=0.01, 0, 1)
        #self.mask_label = self.mask_label.to(torch.float32)
        tp_idx = self._get_tp_idx(self.H)
        H_tp = self.H[:, tp_idx:tp_idx+1]  # (B,1,HR,HR)
        #TODO:chek warum interpolate
        self.mask_label = F.interpolate(H_tp, size=self.L.shape[-2:], mode="area")
        self.mask_label = (self.mask_label > 0).float()


    # ----------------------------------------
    # feed L to netG and get E
    # ----------------------------------------
    def netG_forward(self):
        self.E, self.supervised_nodes, self.mask = self.netG(self.L)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        # ------------------------------------
        # optimize G
        # ------------------------------------
        for p in self.netD.parameters():
            p.requires_grad = False

        self.G_optimizer.zero_grad()
        self.netG_forward()
        loss_G_total = 0
        tp_idx = self._get_tp_idx(self.H)
        E_tp = self.E[:, tp_idx : tp_idx + 1]
        H_tp = self.H[:, tp_idx : tp_idx + 1]
        if self.train_only_tp_channel:
            E_for_disc, H_for_disc = E_tp, H_tp
        else:
            E_for_disc, H_for_disc = self.E, self.H
        if self.train_only_tp_channel:
            if E_for_disc.shape[1] != 1 or H_for_disc.shape[1] != 1:
                raise RuntimeError(
                    "Single-channel training mode expected tensors with one channel."
                )

        if current_step % self.D_update_ratio == 0 and current_step > self.D_init_iters:  # updata D first
            if self.opt_train['global_lossfn_weight'] > 0:
                global_loss = self.global_lossfn_weight * self.global_lossfn(
                    E_tp.sum(dim=(2, 3)), H_tp.sum(dim=(2, 3))
                )
                loss_G_total += global_loss      
            if self.opt_train['G_lossfn_weight'] > 0:
                loss_tp   = self.G_lossfn(E_tp * self.supervised_nodes, H_tp * self.supervised_nodes)
                if self.train_only_tp_channel:
                    G_loss = self.G_lossfn_weight * loss_tp
                else:
                    rest_channels = [i for i in range(self.E.shape[1]) if i != tp_idx]
                    if rest_channels:
                        E_rest = self.E[:, rest_channels]
                        H_rest = self.H[:, rest_channels]
                        loss_rest = self.G_lossfn(E_rest, H_rest)
                    else:
                        loss_rest = torch.zeros((), device=self.device)
                    G_loss = self.G_lossfn_weight * (loss_rest + loss_tp)
                loss_G_total += G_loss                 # 1) pixel loss
            if self.opt_train['F_lossfn_weight'] > 0:
                if self.train_only_tp_channel:
                    F_loss = self.F_lossfn_weight * self._compute_feature_loss_all_channels(
                        E_tp, H_tp
                    )
                else:
                    F_loss = self.F_lossfn_weight * self._compute_feature_loss_all_channels(
                        self.E, self.H
                    )
                loss_G_total += F_loss                 # 2) VGG feature loss
            if self.opt_train['M_lossfn_weight'] > 0:
                M_loss = self.M_lossfn_weight * self.M_lossfn(self.mask, self.mask_label)
                loss_G_total += M_loss                 # 2) VGG feature loss
            if self.opt['train']['gan_type'] in ['gan', 'lsgan', 'wgan', 'softplusgan']:
                pred_g_fake = self.netD(E_for_disc)
                D_loss = self.D_lossfn_weight * self.D_lossfn(pred_g_fake, True)
            elif self.opt['train']['gan_type'] == 'ragan':
                pred_d_real = self.netD(H_for_disc).detach()
                pred_g_fake = self.netD(E_for_disc)
                D_loss = self.D_lossfn_weight * (
                        self.D_lossfn(pred_d_real - torch.mean(pred_g_fake, 0, True), False) +
                        self.D_lossfn(pred_g_fake - torch.mean(pred_d_real, 0, True), True)) / 2
            loss_G_total += D_loss                    # 3) GAN loss

            loss_G_total.backward()
            self.G_optimizer.step()

        # ------------------------------------
        # optimize D
        # ------------------------------------
        for p in self.netD.parameters():
            p.requires_grad = True

        self.D_optimizer.zero_grad()

        # In order to avoid the error in distributed training:
        # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
        # the variables needed for gradient computation has been modified by
        # an inplace operation",
        # we separate the backwards for real and fake, and also detach the
        # tensor for calculating mean.
        if self.opt_train['gan_type'] in ['gan', 'lsgan', 'wgan', 'softplusgan']:
            # real
            pred_d_real = self.netD(H_for_disc)                # 1) real data
            l_d_real = self.D_lossfn(pred_d_real, True)
            l_d_real.backward()
            # fake
            pred_d_fake = self.netD(E_for_disc.detach().clone()) # 2) fake data, detach to avoid BP to G
            l_d_fake = self.D_lossfn(pred_d_fake, False)
            l_d_fake.backward()
        elif self.opt_train['gan_type'] == 'ragan':
            # real
            pred_d_fake = self.netD(E_for_disc).detach()       # 1) fake data, detach to avoid BP to G
            pred_d_real = self.netD(H_for_disc)                # 2) real data
            l_d_real = 0.5 * self.D_lossfn(pred_d_real - torch.mean(pred_d_fake, 0, True), True)
            l_d_real.backward()
            # fake
            pred_d_fake = self.netD(E_for_disc.detach())
            l_d_fake = 0.5 * self.D_lossfn(pred_d_fake - torch.mean(pred_d_real.detach(), 0, True), False)
            l_d_fake.backward()

        self.D_optimizer.step()

        # ------------------------------------
        # record log
        # ------------------------------------
        if current_step % self.D_update_ratio == 0 and current_step > self.D_init_iters:
            if self.opt_train['G_lossfn_weight'] > 0:
                self.log_dict['G_loss'] = G_loss.item()
            if self.opt_train['global_lossfn_weight'] > 0:
                self.log_dict['global_loss'] = global_loss.item()

            if self.opt_train['M_lossfn_weight'] > 0:
                self.log_dict['M_loss'] = M_loss.item()

            if self.opt_train['F_lossfn_weight'] > 0:
                self.log_dict['F_loss'] = F_loss.item()
            self.log_dict['D_loss'] = D_loss.item()

        #self.log_dict['l_d_real'] = l_d_real.item()
        #self.log_dict['l_d_fake'] = l_d_fake.item()
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test and inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=5, sf=self.opt['scale'], modulo=1)
        self.netG.train()
    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H images
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG, netD and netF
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)
        if self.is_train:
            msg = self.describe_network(self.netD)
            print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        if self.is_train:
            msg += self.describe_network(self.netD)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
