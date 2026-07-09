"""
BatVision U-Net model (sound-only) — the reference model for this benchmark.

MODEL ONLY, ported from:
    https://github.com/AmandineBtto/Batvision-Dataset
    UNetSoundOnly/models/unetbaseline_model.py

This is the pix2pix / CycleGAN-style encoder->decoder U-Net (`unet_256`, 8 downs)
that takes the binaural cue stack `(in_ch, 256, 512)` and regresses a single-channel
ERP depth map in [0, 1]. It is a PLAIN pixel-decoder U-Net — NOT ray-conditioned —
and serves as the fixed reference that "my model" (`train.py`) must beat.

Only the network is imported. The BatVision data pipeline / losses / metrics are NOT
used — this repo's `prepare.py` remains the fixed data + evaluation harness, and
`run_base.py` trains this model under the exact same harness as `train.py` for a fair,
apples-to-apples comparison.
"""

import functools
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# ============================================================
# Verbatim BatVision model (pix2pix / CycleGAN U-Net)
# ============================================================

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights (normal | xavier | kaiming | orthogonal)."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer: batch | instance | none."""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class Identity(nn.Module):
    def forward(self, x):
        return x


def define_G(cfg, input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False,
             init_type='normal', init_gain=0.02):
    """Create + initialize a U-Net generator (unet_128 | unet_256)."""
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'unet_128':
        net = UnetGenerator(cfg, input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(cfg, input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    init_weights(net, init_type, init_gain=init_gain)
    return net


class UnetGenerator(nn.Module):
    """A Unet-based generator built recursively innermost -> outermost."""

    def __init__(self, cfg, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        # innermost
        unet_block = UnetSkipConnectionBlock(cfg, ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):          # intermediate ngf*8 layers
            unet_block = UnetSkipConnectionBlock(cfg, ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce ngf*8 -> ngf
        unet_block = UnetSkipConnectionBlock(cfg, ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(cfg, ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(cfg, ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # outermost
        self.model = UnetSkipConnectionBlock(cfg, output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Unet submodule with skip connection:
        X ------------------identity-----------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, cfg, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            if cfg.dataset.depth_norm:
                up = [uprelu, upconv, nn.Sigmoid()]
            else:
                up = [uprelu, upconv, nn.ReLU()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:                                   # skip connection
            return torch.cat([x, self.model(x)], 1)


# ============================================================
# Adapter to the auto_audio_depth_estimation training loop
# ============================================================

class BatVisionUNet(nn.Module):
    """Wrap the BatVision `UnetGenerator` to the {'D','D0','extras'} output contract
    that `run_base.py` (composite_loss / evaluate) expects.

    Input : spec (B, in_ch, H, W)   binaural spectrogram / RIR spatial feature
    Output: {"D": (B,1,H,W) in [0,1], "D0": same, "extras": {}}

    Depth is normalised to [0,1] in prepare.py, so the outermost head is Sigmoid
    (`depth_norm=True`). No coarse head -> extras is empty; composite_loss falls back
    to pooling `D` for its coarse-layout term.
    """

    def __init__(self, cfg):
        super().__init__()
        self.H, self.W = int(cfg.dataset.images_size[0]), int(cfg.dataset.images_size[1])
        in_ch = int(getattr(cfg.dataset, 'in_ch', 2))
        ngf = int(getattr(cfg.model, 'ngf', 64))
        netG = getattr(cfg.model, 'generator', 'unet_256')
        shim = SimpleNamespace(dataset=SimpleNamespace(depth_norm=True))
        self.net = define_G(shim, input_nc=in_ch, output_nc=1, ngf=ngf, netG=netG,
                            norm='batch', use_dropout=False, init_type='normal', init_gain=0.02)

    def forward(self, spec, coarse_feat=None, sh_basis=None):
        D = self.net(spec)
        if D.shape[-2:] != (self.H, self.W):
            D = F.interpolate(D, (self.H, self.W), mode='bilinear', align_corners=False)
        return {"D": D, "D0": D, "extras": {}}


def build_batvision_model(cfg):
    """Build the BatVision U-Net from the nested run_base.py cfg."""
    return BatVisionUNet(cfg)
