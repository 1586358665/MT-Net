import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from collections import OrderedDict
from CA.coordatt import *
from CA.cbam import *
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)


    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r

class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock1(indim, outdim)

        self.block2 = ResBlock1(outdim, outdim)
        self.attention1 =CoordAtt(outdim,outdim)#7.3
        # self.attention = CBAM(outdim)

    def forward(self, x, f16):
        x = torch.cat([x, f16], 1)
        x = self.block1(x)
        r = self.attention1(x)
        x = self.block2(x + r)

        return x







class GlobalContextQueryMV2(nn.Module):
    def __init__(self, ch_in=134, ch_val=134, ch_feat=64):
        super().__init__()

        self.conv_val = nn.Sequential(
            nn.Conv2d(ch_in, ch_feat, 1, 1, 0),
            nn.Conv2d(ch_feat, ch_feat, 5, 1,2),
            nn.LeakyReLU(0.2, inplace=True)

        )
        # self.conv_feat = nn.Sequential(
        #     nn.Conv2d(ch_val, ch_feat, 1, 1, 0),
        #     nn.Conv2d(ch_feat, ch_feat, 5, 1,2),
        #     nn.LeakyReLU(0.2, inplace=True)
        #
        # )
        self.blend = nn.Sequential(
            nn.Conv2d(2 * ch_feat, 2 * ch_feat, 5, 1,2),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * ch_feat,  ch_feat, 1, 1, 0),

        )
        # self.mish=Mish()
    def forward(self, input, gc):
        val = self.conv_val(input)
        # feat = self.conv_feat(input)

        B, K, H, W = val.shape
        val_mm = val.reshape(B, K, -1)

        gc_act = torch.bmm(gc, val_mm).reshape(B, -1, H, W)
        out = F.relu(self.blend(torch.cat([val, gc_act], dim=1)))
        # out = self.mish(self.blend(torch.cat([feat, gc_act], dim=1)))
        return(out)


class GlobalContextMemoryMV2(nn.Module):
    def __init__(self, ch_key=137, ch_val=137, ch_out=64, group_n=4):
        super().__init__()
        self.conv_key = nn.Sequential(
            nn.Conv2d(ch_key, ch_out, 1, 1, 0),
            nn.Conv2d(ch_out, ch_out, 5, 1, 2, groups=ch_out//group_n),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_val = nn.Sequential(
            nn.Conv2d(ch_val, ch_out, 1, 1, 0),
            nn.Conv2d(ch_out, ch_out, 5, 1, 2, groups=ch_out//group_n),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # self.scale_constant = 1./math.sqrt(ch_key)

    def forward(self, key, mask):

        keys = torch.cat([key, mask], dim=1)

        key = self.conv_key(keys)
        val = self.conv_val(keys)
        B, KC, H, W = key.shape
        B, KV, H, W = val.shape
        key_mm = key.reshape(B, KC, -1)
        val_mm = val.reshape(B, KV, -1).permute((0, 2, 1)).contiguous()
        out = F.softmax(torch.bmm(key_mm, val_mm).mul(1./math.sqrt(H*W)), dim=2)
        return out

#



def conv_bn(inp, oup, kernel=1, bias=False, dilation=1):

    pad = kernel // 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel, padding=pad, bias=bias, dilation=dilation),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )



class ResBlock1(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock1, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(planes)
        self.relu= nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = self.conv3(x)

        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn(out)
        out = self.relu(out)

        out += residual
        out = self.relu(out)

        return out
















