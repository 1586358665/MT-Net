import torch.nn as nn
import torch
from nets.stdcnet import STDCNet1446, STDCNet813
# from model.backbones.resnet import resnet34, resnet50, resnet101, resnet152,resnet18
# from torchsummaryX import summary
import torch.nn.functional as F
from collections import OrderedDict

# from conv_block import Conv
import functools
from functools import partial
import os, sys

# from inplace_abn import InPlaceABN, InPlaceABNSync
# from model.sync_batchnorm import SynchronizedBatchNorm2d

# BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')
# from torch.nn import SyncBatchNorm

__all__ = ['SPFNet']

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=(1, 1), group=1, bn_act=False,
                 bias=False):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=group, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            x = self.conv(x)
            x = self.bn(x)
            x = self.act(x)

            return x
        else:
            x =self.conv(x)

            return x
class SPFNet(nn.Module):
    def __init__(self, backbone='STDCNet1446', pretrain_model='STDCNet1446_76.47.tar', use_conv_last=False, *args,
                 **kwargs):

        super(SPFNet, self).__init__()

        if backbone == 'STDCNet1446':
            self.backbone = STDCNet1446(pretrain_model=pretrain_model, use_conv_last=use_conv_last)

        elif backbone == 'STDCNet813':
            self.backbone = STDCNet813(pretrain_model=pretrain_model, use_conv_last=use_conv_last)


        
        self.spfm = SPFM(1024, 1024,8)
        
     
        channels = [64, 64, 128, 256, 512,1024]
      
        self.egca1 = EGCA(channels[1])
        self.egca2 = EGCA(channels[3])
        self.egca3 = EGCA(channels[4])
        self.egca4 = EGCA(channels[5])
        
        self.adjust2 = Adjustment(64,128)
        self.adjust3 = Adjustment(256, 128)
        self.adjust4 = Adjustment(512, 128)
        self.adjust5 = Adjustment(1024, 128)

        # Multi-Scale feature fusion
        self.fuse1 = conv_block(576, 64, 3, 1, padding=1, bn_act=True)
        
        # Decoder-based subpixel convolution
        self.dsc5 = DSCModule(1024,512)
        self.dsc4 = DSCModule(512,256)
        self.dsc3 = DSCModule(256, 64)
        self.dsc2 = DSCModule(64, 64)

        self.classifier = Classifier(64, 9)

    def forward(self, x):
        B, C, H, W = x.size()
        feat2, x2, x3, x4 , x5 = self.backbone(x)

        Spfm = self.spfm(x5)

        dsc5 = self.dsc5(x5,Spfm)  
        dsc4 = self.dsc4(x4,dsc5)
        dsc3 = self.dsc3(x3,dsc4)
        dsc2 = self.dsc2(x2,dsc3)

        # Efficient global context aggregation
        gui1 = self.egca1(x2) 
        gui2 = self.egca2(x3) 
        gui3 = self.egca3(x4)  
        gui4 = self.egca4(x5) 

        adj2 = self.adjust2(gui1)
        adj3 = self.adjust3(gui2)
        adj4 = self.adjust4(gui3)
        adj5 = self.adjust5(gui4)

        adj2s = F.interpolate(adj2, size=x2.size()[2:], mode="bilinear",align_corners=True)
        adj3s = F.interpolate(adj3, size=x2.size()[2:], mode="bilinear",align_corners=True)
        adj4s = F.interpolate(adj4, size=x2.size()[2:], mode="bilinear",align_corners=True)
        adj5s = F.interpolate(adj5, size=x2.size()[2:], mode="bilinear",align_corners=True)
        dsc2s = F.interpolate(dsc2, size=x2.size()[2:], mode="bilinear",align_corners=True)
      

        msfuse = torch.cat((adj2s,adj3s,adj4s,adj5s,dsc2s), dim=1)
        msfuse = self.fuse1(msfuse)

        classifier = self.classifier(msfuse)
        classifier = F.interpolate(classifier, size=(H, W), mode='bicubic', align_corners=True)

        return adj2,adj3s,adj4s,classifier


class RPPModule(nn.Module):
    def __init__(self, in_channels: int, groups=2) -> None:
        super(RPPModule, self).__init__()
        self.groups = groups
        self.conv_dws1 = nn.Sequential(
            conv_block(in_channels, 4*in_channels, kernel_size=3, stride=1, padding=4,
                                    group=1, dilation=4, bn_act=True),
            nn.PixelShuffle(upscale_factor=2))
        self.conv_dws2 = nn.Sequential(
            conv_block(in_channels, 4*in_channels, kernel_size=3, stride=1, padding=8,
                                    group=1, dilation=8, bn_act=True),
            nn.PixelShuffle(upscale_factor=2))

        self.fusion = nn.Sequential(
            conv_block(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, group=1, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True))

        self.conv_dws3 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.avgpool=torch.nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        br1 = self.conv_dws1(x)
        b2 = self.conv_dws1(x)

        out = torch.cat((br1, b2), dim=1)
        out = self.fusion(out)

        br3 = self.conv_dws3(self.avgpool(x))
        output = br3 + out

        return output


class SPFM(nn.Module):
    def __init__(self, in_channels, out_channels, num_splits):
        super(SPFM,self).__init__()

        assert in_channels % num_splits == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_splits = num_splits
        self.subspaces = nn.ModuleList(
            [RPPModule(int(self.in_channels / self.num_splits)) for i in range(self.num_splits)])

        self.out = conv_block(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bn_act=True)

    def forward(self, x):
        group_size = int(self.in_channels / self.num_splits)
        sub_Feat = torch.chunk(x, self.num_splits, dim=1)
        out = []
        for id, l in enumerate(self.subspaces):
            out.append(self.subspaces[id](sub_Feat[id]))
        out = torch.cat(out, dim=1)
        out = self.out(out)
        return out


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)

    return x


class EGCA(nn.Module):
    def __init__(self, in_channels: int, groups=2) -> None:
        super(EGCA, self).__init__()
        self.groups = groups
        self.conv_dws1 = conv_block(in_channels // 2, in_channels, kernel_size=1, stride=1, padding=0,
                                    group=in_channels // 2, bn_act=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pw1 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.softmax = nn.Softmax(dim=1)

        self.conv_dws2 = conv_block(in_channels // 2, in_channels, kernel_size=1, stride=1, padding=0,
                                    group=in_channels // 2,
                                    bn_act=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pw2 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=False)

        self.branch3 = nn.Sequential(
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, group=in_channels, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, x1 = x.chunk(2, dim=1)
        out1 = self.conv_dws1(x0)
        out1 = self.maxpool1(out1)
        out1 = self.conv_pw1(out1)

        out2 = self.conv_dws1(x1)
        out2 = self.maxpool1(out2)
        out2 = self.conv_pw1(out2)

        out = torch.add(out1, out2)

        b, c, h, w = out.size()
        out = self.softmax(out.view(b, c, -1))
        out = out.view(b, c, h, w)

        out = out.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        out = torch.mul(out, x)
        out = torch.add(out, x)
        out = channel_shuffle(out, groups=self.groups)

        br3 = self.branch3(x)

        output = br3 + out

        return output



class DSCModule(nn.Module):
    def __init__(self, in_channels, out_channels, red=1):
        super(DSCModule, self).__init__()
     
        self.conv1 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=True)
        self.conv2 = conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bn_act=True)
        self.conv3 = nn.Sequential(
            conv_block(2 * in_channels, 4 * out_channels, kernel_size=3, stride=1, padding=1, bn_act=True),
            nn.PixelShuffle(upscale_factor=2))

    def forward(self, x_gui, y_high):
        h, w = x_gui.size(2), x_gui.size(3)

        y_high = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y_high)
        x_gui = self.conv1(x_gui)
        y_high = self.conv2(y_high)
       
        out = torch.cat([y_high, x_gui], 1)

        out = self.conv3(out)

        return out


class Classifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Classifier, self).__init__()
        self.fc = conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.fc(x)


class Adjustment(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Adjustment, self).__init__()
        self.conv = conv_block(in_channels, out_channels, 1, 1, padding=0, bn_act=True)

    def forward(self, x):
        return self.conv(x)




# if __name__ == "__main__":
#     input_tensor = torch.rand(1, 3, 240, 448)
#     model = SPFNet("STDCNet1446")
#     summary(model, input_tensor)




