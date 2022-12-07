
from libs.models.layers import *
from backbone import *
from libs.models.model_stages import *

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

class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('naf', nn.ReLU(inplace=True))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)




class REFINEV3(nn.Module):
    def __init__(self, ch_s16=128, ch_s8=64):#ch_s16=1024,
        super().__init__()

        ch_out_deconv = 2



        self.conv_s16 = ConvRelu(6+ ch_s16, 32, 3, 1, 1)

        self.blend_s161 = ConvRelu(32 + ch_s16, 16, 3, 1, 1)#32+128
        self.conv_s8 = ConvRelu(ch_s8, 16, 1, 1, 0)
        self.blend_s81 = ConvRelu(16+ ch_out_deconv, 16, 3, 1, 1)
        self.res=ResBlock(32,32)
        self.conv_s41 = ConvRelu(ch_s16, 16, 1, 1, 0)
        self.blend_s41 = ConvRelu(16+ ch_out_deconv, 16, 3, 1, 1)
        self.deconv1_1 = nn.ConvTranspose2d(16, ch_out_deconv, 4, 2, 1, bias=True)
        self.deconv1_2 = nn.ConvTranspose2d(ch_out_deconv, ch_out_deconv, 4, 2, 1, bias=True)
        self.deconv2 = nn.ConvTranspose2d(16, ch_out_deconv, 4, 2, 1, bias=True)
        self.deconv3 = nn.ConvTranspose2d(16+ 2 * ch_out_deconv, 2 * 4, 4, 2, 1, bias=True)
        self.predictor = nn.PixelShuffle(2)

        self.predictor_Diff = nn.Conv2d(16, 2, 3, 1, 1)
        self.predictor_tmp=nn.Conv2d(64, 1, 3, 1, 1)
        self.attention = CoordAtt(32, 32)

        self.cal_UpW = nn.Sequential(
            ConvRelu(18, 9, 3, 2, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(9, 1, 1, 1, 0),
            nn.Sigmoid())


    def forward(self, s16,ms8,s4, prev_seg,prev_seg_feats16):
        prev_seg = F.avg_pool2d(prev_seg, 3, 2, 1)

        shortTmp = self.conv_s16(torch.cat([prev_seg, prev_seg_feats16], dim=1))

        shortTmp = self.attention(shortTmp)
        #
        shortTmp = self.res(shortTmp)


        u16 = self.blend_s161(torch.cat([s16, shortTmp], dim=1))

        out_16 = self.deconv1_1(u16)



        u8 = torch.cat([self.conv_s8(ms8), out_16], dim=-3)

        update = self.cal_UpW(u8).squeeze(3).expand(ms8.size(0), ms8.size(1), ms8.size(1))

        u8 = self.blend_s81(u8)

        out_8 = self.deconv2(u8)


        u = torch.cat([self.conv_s41(s4), out_8], dim=1)

        out_4S = self.blend_s41(u)


        out_16=self.deconv1_2(out_16)

        out_4 = self.deconv3(torch.cat([out_16, out_8, out_4S], dim=1))

        segscore = self.predictor(out_4)

        Diff_heatmap = self.predictor_Diff(u8)


        Tmp_heatmap = torch.sigmoid(self.predictor_tmp(ms8))



        return segscore,Diff_heatmap,Tmp_heatmap,update