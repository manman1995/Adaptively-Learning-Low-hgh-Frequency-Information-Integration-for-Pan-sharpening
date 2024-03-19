# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms

from .modules import InvertibleConv1x1
from .refine import Refine
import torch.nn.init as init
from models.utils.CDC import cdcconv

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d = 1, init='xavier', gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3





class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out




class FrePreParam(nn.Module):
    def __init__(self, channels):
        super(FrePreParam, self).__init__()
        self.pan_extract  = nn.Sequential(nn.Conv2d(1,channels,3,1,1),nn.Conv2d(channels,channels,1,1,0))
        self.ms_extract = nn.Sequential(nn.Conv2d(4,channels,3,1,1),nn.Conv2d(channels,channels,1,1,0))
        self.param_predict = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                           nn.Conv2d(channels,channels//2,1,1,0),
                                           nn.LeakyReLU(0.1),
                                           nn.Conv2d(channels//2,2,1,1,0),
                                           nn.Sigmoid())

    def forward(self, ms, pan):
        ##### feature extract
        panf = self.pan_extract(pan)
        msf = self.ms_extract(ms)

        ##### FFT
        _, _, H, W = msf.shape
        msF = torch.fft.fft2(msf + 1e-8, norm='backward')
        panF = torch.fft.fft2(panf + 1e-8, norm='backward')

        msF = torch.fft.fftshift(msF)+1e-8
        panF = torch.fft.fftshift(panF)+1e-8

        msF_amp = torch.abs(msF)+1e-8
        msF_pha = torch.angle(msF)+1e-8
        panF_amp = torch.abs(panF)+1e-8
        panF_pha = torch.angle(panF)+1e-8

        #### Param Predict
        resF_amp = msF_amp-panF_amp
        param = self.param_predict(resF_amp)
        alpha,beta = param[:,0:1],param[:,1:2]

        return msf,panf,msF_amp,panF_amp,msF_pha,panF_pha,alpha,beta

import copy

class FrePreCrop(nn.Module):
    def __init__(self):
        super(FrePreCrop, self).__init__()


    def forward(self, msF_amp, panF_amp, alpha, beta):
        N,C,H,W = msF_amp.size()
        maskzero = torch.zeros_like(msF_amp)
        h,w = torch.floor(alpha*H),torch.floor(beta*W)
        maskones = torch.ones_like(msF_amp)
        # maskcropones = torch.ones(size=[N,C,h,w])

        # msF_amp_low = copy.deepcopy(msF_amp)
        if C == 4:
            maskzero[:,0,(H-h[0])//2:(H+h[0])//2,(W-w[0])//2:(W+w[0])//2]=1.0
            maskzero[:, 1, (H - h[1]) // 2:(H + h[1]) // 2, (W - w[1]) // 2:(W + w[1]) // 2] = 1.0
            maskzero[:, 2, (H - h[2]) // 2:(H + h[2]) // 2, (W - w[2]) // 2:(W + w[2]) // 2] = 1.0
            maskzero[:, 3, (H - h[3]) // 2:(H + h[3]) // 2, (W - w[3]) // 2:(W + w[3]) // 2] = 1.0
        elif C == 1:
            maskzero[:, 0, (H - h[0]) // 2:(H + h[0]) // 2, (W - w[0]) // 2:(W + w[0]) // 2] = 1.0

        mask = maskzero
        # invmask = maskones-mask

        msF_amp_low = msF_amp*mask
        msF_amp_high = msF_amp-msF_amp_low
        panF_amp_low = panF_amp * mask
        panF_amp_high = panF_amp-panF_amp_low

        return msF_amp_low.detach(),msF_amp_high.detach(),panF_amp_low.detach(),panF_amp_high.detach()



class Integrate(nn.Module):
    def __init__(self, channels):
        super(Integrate, self).__init__()
        self.spa_process = nn.Sequential(InvBlock(DenseBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels,channels,1,1,0))
        self.amp_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.cha_att = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, kernel_size=1, padding=0, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels * 2, kernel_size=1, padding=0, bias=True),
                                     nn.Sigmoid())
        self.fusion = nn.Conv2d(channels * 2, channels, 3, 1, 1)

    def forward(self, msf,panf, msF_amp,panF_amp,msF_pha,panF_pha):

        _, _, H, W = msF_pha.shape


        spafuse = self.spa_process(torch.cat([msf,panf],1))

        amp_fuse = self.amp_fuse(torch.cat([msF_amp,panF_amp],1))+1e-8
        pha_fuse = self.pha_fuse(torch.cat([msF_pha,panF_pha],1))+1e-8

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        complex1 = torch.complex(real, imag)+1e-8
        complex1 = torch.fft.ifftshift(complex1)+1e-8
        frefuse = torch.abs(torch.fft.ifft2(complex1+1e-8, s=(H, W), norm='backward'))

        cat_f = torch.cat([spafuse, frefuse], 1)
        out = self.fusion(self.cha_att(self.contrast(cat_f) + self.avgpool(cat_f))*cat_f)

        return out




class FeatureProcess(nn.Module):
    def __init__(self, channels):
        super(FeatureProcess, self).__init__()

        self.conv_p = FrePreParam(channels)
        self.frecrop = FrePreCrop()

        self.block = Integrate(channels)
        self.refine = InvBlock(DenseBlock,channels,channels//2)
        self.block1 = Integrate(channels)
        self.refine1 = InvBlock(DenseBlock, channels, channels // 2)
        self.fuse = nn.Conv2d(4*channels,channels,1,1,0)


    def forward(self, ms, pan):
        msf,panf,msF_amp,panF_amp,msF_pha,panF_pha,alpha,beta = self.conv_p(ms,pan)
        msF_amp_low, msF_amp_high, panF_amp_low, panF_amp_high = self.frecrop(msF_amp,panF_amp,alpha, beta)

        msf0 = self.block(msf, panf, msF_amp_low,panF_amp_low,msF_pha,panF_pha)
        msf01 = self.refine(msf0)

        _, _, H, W = msf.shape
        msF01 = torch.fft.fft2(msf01 + 1e-8, norm='backward')
        msF01 = torch.fft.fftshift(msF01)+1e-8
        msF01_amp = torch.abs(msF01)+ 1e-8
        msF_amp_high = msF01_amp+msF_amp_high.detach()


        msf1 = self.block1(msf01, panf, msF_amp_high,panF_amp_high,msF_pha,panF_pha)
        msf11 = self.refine1(msf1)

        msout = self.fuse(torch.cat([msf0,msf01,msf1,msf11],1))

        return msout


def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)


class AFIIN(nn.Module):
    def __init__(self, channels):
        super(AFIIN, self).__init__()
        self.process = FeatureProcess(channels)
        # self.cdc = nn.Sequential(nn.Conv2d(1, 4, 1, 1, 0), cdcconv(4, 4), cdcconv(4, 4))
        self.refine = Refine(channels, 4)

    def forward(self, ms, pan=None):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape

        mHR = upsample(ms, M, N)
        HRf = self.process(mHR, pan)
        HR = self.refine(HRf)+ mHR

        return HR



def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)