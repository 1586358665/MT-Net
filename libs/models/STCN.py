import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from libs.utils.loss import *
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
import matplotlib.pyplot as plt
import os
import cv2
from torchvision import models
from backbone1 import *
import numpy as np
import time
from libs.models.model_stages import *

from detail_loss import DetailAggregateLoss

from libs.models.heatmap_loss import *



class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w


        b, _, H, W = q_in.size()
        m_in=m_in.view(b, -1, H * W).permute(0, 2, 1)
        m_out = m_out.view(b, -1, H * W).permute(0, 2, 1)
        no, centers, C = m_in.size()
        _, _, vd = m_out.shape

        qi = q_in.view(-1, C, H * W)
        p = torch.bmm(m_in, qi)  # no x centers x hw 查询帧与内存帧亲和度计算
        p = p / math.sqrt(C)
        p = torch.softmax(p, dim=1)  # no x centers x hw

        mo = m_out.permute(0, 2, 1)  # no x c x centers
        mem = torch.bmm(mo, p)  # no x c x hw  与内存value进行乘积操作
        mem = mem.view(no, vd, H, W)

        # mem_out = torch.cat([mem, q_out], dim=1) #与查询帧的value进行 cat



        return mem,q_out

class ValueEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_bg = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


        resnet = models.resnet18(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 1/4, 64
        self.layer2 = resnet.layer2 # 1/8, 128
        self.layer3 = resnet.layer3 # 1/16, 256

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.attention = CoordAtt(256,256)#6.9
        self.convX=ConvBNReLU(256,256)#6.9
        self.fuser1 = FeatureFusionBlock(128+128 ,128)
        # self.attention1 = CBAM(256)

    def forward(self, in_f, in_m, in_bg,r3s):


        f = in_f
        m = torch.unsqueeze(in_m, dim=1).float()  # add channel dim
        bg = torch.unsqueeze(in_bg, dim=1).float()

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_bg(bg)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        r2 = self.layer1(x)   # 1/4, 64
        r3= self.layer2(r2) # 1/8, 128

        r4= self.layer3(r3) # 1/16, 256
        r3 = self.fuser1(r3, r3s)
        r4=self.attention(r4)
        r4= self.convX(r4)


        return r4, r3

class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()

    def get_affinity(self, mk, qk):

        o, CK, H, W = qk.shape

        qk = qk.flatten(start_dim=2)#6.8
        mk = mk.flatten(start_dim=2)
        # mk=mk.transpose(1,2)
        a = mk.pow(2).sum(1).unsqueeze(2)
        b = 2 * (mk.transpose(1, 2) @ qk)
        # this term will be cancelled out in the softmax
        # c = qk.pow(2).sum(1).unsqueeze(1)

        affinity = (-a + b) / math.sqrt(CK)  # B, THW, HW

        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum

        return affinity

    def readout(self, affinity, mv, qv):
        o, CV, H, W = qv.shape

        mo = mv.view(o, CV, H* W)
        mem = torch.bmm(mo, affinity)  # Weighted-sum B, CV, HW
        mem = mem.view(o, CV, H, W)

        # mem_out = torch.cat([mem, qv], dim=1)

        return mem,qv


def Soft_aggregation(ps, max_obj):


    num_objects, H, W = ps.shape
    em = torch.zeros(1, max_obj + 1, H, W, device=ps.device)
    # em = torch.zeros(1, max_obj + 1, H, W).to(ps.device)
    em[0, 0, :, :] = torch.prod(1 - ps, dim=0)  # bg prob
    em[0, 1:num_objects + 1, :, :] = ps  # obj prob
    em = torch.clamp(em, 1e-7, 1 - 1e-7)
    logit = torch.log((em / (1 - em)))
    return logit

def draw_features_plot(features,name):
    fig = plt.figure(figsize=(16, 16))  # 窗口上创建一个指定大小新图形
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    features = F.interpolate(features, size=[240, 427], mode="bilinear",align_corners=True)
    channel = int(features.size(1)/8)

    for j in range(8):
        # output_name = "第%s车道线" % (j + 1)
        # PATH = os.path.join("./dataset", "layermap", name, output_name)
        for i in range(channel):
            plt.subplot(1, 1,  1)
            plt.axis('off')
            img = features[j, 1, :, :]
            pmin = torch.min(img)
            pmax = torch.max(img)

            img = ((img - pmin) / (pmax - pmin + 0.00001)) * 255  # float在[0，1]之间，转换成0-255
            img = img.detach().cpu().numpy()
            img = img.astype(np.uint8)  # 转成unit8
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
            img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
            # print(img.shape)
            img = cv2.resize(img, (427, 240))
            # print(img.shape)
            plt.imshow(img)  # 这句就相当于把小图按照位置添加进去(不是交互模式，需要plt.show()才会真正显示出来)
            # output_name="第%s车道线"%(j+1)
            # PATH = os.path.join("./dataset", "layermap", name,output_name)
            output_name = "第%s_%s车道线" % (j,i)
            PATH = os.path.join("./dataset", "layermap", name, output_name)
            fig.savefig(PATH, dpi=100)
    fig.clf()  # # Clear figure
    plt.close()  # Close a figure window(plt是窗口，而fig则是1位于该窗口上的一个图形)
    # cla()   Clear axis





# 计算grad-cam并可视化
def cam_show_img(img, feature_map, grads, out_dir):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
    grads = grads.reshape([grads.shape[0],-1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							# 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heatmap + 0.7 * img

    path_cam_img = os.path.join(out_dir, "cam.jpg")
    cv2.imwrite(path_cam_img, cam_img)

class KeyValue(nn.Module):
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.Key(x), self.Value(x)


class MT(nn.Module):
    def __init__(self, keydim, valdim, phase='test', mode='recurrent', iou_threshold=0.5):
        super(MT, self).__init__()
        # self.Encoder_M = Encoder_M()
        self.Encoder_M = ValueEncoder()
        self.Encoder_Q = GlobalContext()
        self.decoder = REFINEV3()

        self.GCM1 = GlobalContextMemoryMV2()
        self.GQ = GlobalContextQueryMV2()

        self.keydim = keydim
        self.valdim = valdim

        self.arm1 = FeatureFusionBlock(128, 128)
        self.arm2 = FeatureFusionBlock(128, 128)

        self.addCoords = AddCoords(with_r=True)

        self.conv_out = BiSeNetOutput(128, 64, 128)
        self.conv_out16 = BiSeNetOutput(128, 64, 128)
        self.conv_out32 = BiSeNetOutput(128, 64, 128)

        self.ffm2 = FeatureFusionModule(192, 128)
        self.ffm8 = FeatureFusionModule(384, 128)
        self.ffm16 = FeatureFusionModule(640, 128)

        # self.KV_M_r3 = KeyValue(512, keydim=self.keydim // 2, valdim=self.keydim // 2)
        # self.KV_M_r4 = KeyValue(1024, keydim=self.keydim//2, valdim=self.keydim//2)

        self.KV_M_r31 = KeyValue(128, keydim=self.keydim // 2, valdim=self.keydim // 2)
        self.KV_M_r41 = KeyValue(256, keydim=self.keydim // 2, valdim=self.keydim // 2)

        self.KV_Q_r31 = KeyValue(128, keydim=self.keydim//2, valdim=self.keydim//2)
        self.KV_Q_r41 = KeyValue(128, keydim=self.keydim // 2, valdim=self.keydim // 2)

        self.attention = CoordAtt(128, 128)  # 6.9
        self.attention1 = CoordAtt(128, 128)



        self.convX = ConvBNReLU(128, 128)
        self.convX1 = ConvBNReLU(128, 128)# 6.9



        self.attentionr3 = CoordAtt(128, 128)
        self.convXr3 = ConvBNReLU(128, 128)

        self.detaliloss= DetailAggregateLoss()

        self.simple_out = BiSeNetOutput(128, 64, 9)
        # self.simple_out_4 = BiSeNetOutput(128, 64, 9)
        self.simple_out_5 = BiSeNetOutput(128, 64, 9)


        self.Tmp_loss = torch.nn.BCELoss().cuda()
        self.heatmap_loss = torch.nn.SmoothL1Loss().cuda()
        # self.heatmap_loss = torch.nn.MSELoss().cuda()

        self.Memory = Memory()
        # self.Memory = MemoryReader()
        # Z
        self.phase = phase
        self.mode = mode
        self.iou_threshold = iou_threshold

        self.flag = 0
        self.count = 0


        assert self.phase in ['train', 'test']



    def load_param(self, weight):

        s = self.state_dict()
        for key, val in weight.items():

            # process ckpt from parallel module
            if key[:6] == 'module':
                key = key[7:]

            if key in s and s[key].shape == val.shape:
                s[key][...] = val
            elif key not in s:
                print('ignore weight from not found key {}'.format(key))
            else:
                print('ignore weight of mistached shape in key {}'.format(key))

        self.load_state_dict(s)

    def get_init_state(self, given_seg,num_=3):

        for _ in range(num_):
            given_seg = F.avg_pool2d(given_seg, kernel_size=3, stride=2, padding=1)
        return given_seg

    def detail_loss(self, given_seg,lb):

        bec_loss2, dice_loss2 = self.detaliloss(given_seg, lb)
        return bec_loss2, dice_loss2

    def make_T(self, masks, num_objects):
        mini_mask = []
        for o in range(1, num_objects + 1):
            mini_mask.append(masks[:, o])
        mask_mini = torch.cat(mini_mask, dim=0).unsqueeze(dim=1)
        tmp = (mask_mini== 1).float().clamp(1e-7, 1 - 1e-7)
        mask=torch.cat([1 - tmp, tmp], dim=1)
        return mask, mask_mini

    def Heatmap_loss(self, pred, GT, PS,len):
        o,f,c,h,w=pred.size()
        O,F,C,H,W=GT.size()
        GT = ( GT[-len:,:, :, :, :] == 1).float().clamp(1e-7, 1 - 1e-7)
        PS =  torch.nn.Upsample(size=(h, w), mode='bilinear',align_corners=True)(PS.view(o * f, c, H, W))
        PS = PS.view(o, f, c, h, w)
        GT=GT.view(o * f, C,  H, W)
        GT = torch.nn.Upsample(size=(h, w), mode='bilinear',align_corners=True)(torch.cat([1 - GT, GT], dim=1))
        GT = GT.view(o, f, c, h, w)
        loss=self.heatmap_loss(pred, (GT - PS))
        return loss

    def tem_Loss(self,temp,mini,num_objects):
        o, f, c, h, w = temp.size()
        O, F, C, H, W = mini.size()
        mini = torch.nn.Upsample(size=(h, w), mode='bilinear',align_corners=False)(mini.view(o * f, c, H, W))
        mini= mini.view(o, f, c, h, w)
        temploss = self.Tmp_loss(temp, mini)
        return temploss

    def pre_Heatmap(self,prev_seg,pse):

        temp_seg = torch.zeros_like(pse)
        prev_segs = torch.cat([temp_seg, temp_seg, temp_seg], dim=1)

        prev_segs[:, :2, :, :] = pse
        prev_segs[:, 2:4, :, :] = prev_seg[:, :2, :, :]
        prev_segs[:, 4:, :, :] = prev_seg[:, 2:4, :, :]
        return prev_segs




    def memorize(self, frame, masks, num_objects, R3):


        # memorize a frame
        # maskb = prob[:, :num_objects, :, :]
        # make batch arg list
        frame_batch = []
        mask_batch = []
        bg_batch = []
        try:
            for o in range(1, num_objects + 1):  # 1 - no
                frame_batch.append(frame)
                mask_batch.append(masks[:, o])

            for o in range(1, num_objects + 1):
                bg_batch.append(torch.clamp(1.0 - masks[:, o], min=0.0, max=1.0)) #背景mask

            # make Batch
            frame_batch = torch.cat(frame_batch, dim=0)
            mask_batch = torch.cat(mask_batch, dim=0)
            bg_batch = torch.cat(bg_batch, dim=0)
        except RuntimeError as re:
            print(re)
            print(num_objects)
            raise re

        from matplotlib import pyplot as plt
        r4, r3 = self.Encoder_M(frame_batch, mask_batch, bg_batch, R3)  # no, c, h, w
        # r4, r3= self.Encoder_M(frame_batch, mask_batch, bg_batch,r4s)


        _, c, h, w = r4.size()
        memfeat = r4

        # k4, v4 = self.KV_M_r4(memfeat)
        # k3, v3 = self.KV_M_r3(r3)

        k4, v4 = self.KV_M_r41(memfeat)
        k3, v3 = self.KV_M_r31(r3)

        return k4, v4, r4, k3, v3,r3

    def segment(self, frame, keys, values, keys3, values3, num_objects,prev_seg):



        feat2, feat4, feat8, feat16,feat8_up, feat16_up, feat32_up = self.Encoder_Q(frame)

        r2 = self.ffm2(feat4, feat8_up)
        r3 = self.ffm8(feat8, feat16_up)
        r4 = self.ffm16(feat16, feat32_up)

        r4 = self.attention(r4)  # 3.1
        r2 = self.attention1(r2)  # 3.1

        r4 = self.convX(r4)
        r2 = self.convX1(r2)


        r2s = self.conv_out(r2)
        r3s = self.conv_out16(r3)
        r4s = self.conv_out32(r4)

        r2e = r2s.expand(num_objects, -1, -1, -1)

        r4e = r4s.expand(num_objects, -1, -1, -1)

        r3e = r3s.expand(num_objects, -1, -1, -1)

        k4, v4 = self.KV_Q_r41(r4s)
        k3, v3 = self.KV_Q_r31(r3s)

        k4e, v4e = k4.expand(num_objects, -1, -1, -1), v4.expand(num_objects, -1, -1, -1)
        k3e, v3e = k3.expand(num_objects, -1, -1, -1), v3.expand(num_objects, -1, -1, -1)

        mem1, qv1 = self.Memory(keys, values, k4e, v4e)
        mem2, qv2 = self.Memory(keys3, values3, k3e, v3e)


        m3 = self.arm1(mem2, qv2)
        m4 = self.arm2(mem1, qv1)

        # r4e = torch.cat([mem1, qv1], dim=1)
        # mem1,qv1 = self.Memory.readout(self.Memory.get_affinity(keys, k4e), values, v4e)
        # mem2,qv2= self.Memory.readout(self.Memory.get_affinity(keys3, k3e), values3, v3e)

        r3es = torch.cat([m3, prev_seg], 1)
        curr_lt = self.GCM1(self.addCoords(r3e), prev_seg)

        GQ = self.GQ(r3es,curr_lt)

        segscore,Diff_heatmap,Tmp_heatmap,update = self.decoder(r4e,  GQ, r2e, prev_seg, m4)

        pre_segemantation = F.softmax(Diff_heatmap, dim=1)  # 该帧de关于u8预测

        ps = F.softmax(segscore, dim=1)#作为记忆损失的前一帧分割
        pse = self.get_init_state(ps)

        temp_seg = self.pre_Heatmap(prev_seg,pse)


        psd = F.softmax(segscore, dim=1)[:, 1]  # no, h, w
        logit = Soft_aggregation(psd, num_objects)



        return  logit,pre_segemantation, curr_lt, ps,temp_seg,r3e,Tmp_heatmap, m4

    def segment2(self, frame, keys, values, keys3, values3, num_objects, prev_seg, key,m4s):

        feat2, feat4, feat8, feat16, feat8_up, feat16_up, feat32_up = self.Encoder_Q(frame)

        r2 = self.ffm2(feat4, feat8_up)
        r3 = self.ffm8(feat8, feat16_up)
        r4 = self.ffm16(feat16, feat32_up)

        r4 = self.attention(r4)  # 3.1
        r2 = self.attention1(r2)  # 3.1

        r4 = self.convX(r4)
        r2 = self.convX1(r2)

        r2s = self.conv_out(r2)
        r3s = self.conv_out16(r3)
        r4s = self.conv_out32(r4)

        r2e = r2s.expand(num_objects, -1, -1, -1)
        r3e = r3s.expand(num_objects, -1, -1, -1)
        r4e = r4s.expand(num_objects, -1, -1, -1)

        r3ex = r3s.expand(num_objects, -1, -1, -1)

        k4, v4 = self.KV_Q_r41(r4s)
        k3, v3 = self.KV_Q_r31(r3s)

        k4e, v4e = k4.expand(num_objects, -1, -1, -1), v4.expand(num_objects, -1, -1, -1)
        k3e, v3e = k3.expand(num_objects, -1, -1, -1), v3.expand(num_objects, -1, -1, -1)

        mem1, qv1 = self.Memory(keys, values, k4e, v4e)
        mem2, qv2 = self.Memory(keys3, values3, k3e, v3e)

        # mem1,qv1 = self.Memory.readout(self.Memory.get_affinity(keys, k4e), values, v4e)
        # mem2,qv2= self.Memory.readout(self.Memory.get_affinity(keys3, k3e), values3, v3e)

        m3 = self.arm1(mem2, qv2)
        m4 = self.arm2(mem1, qv1)

        # r4e = torch.cat([mem1, qv1], dim=1)
        r3es = torch.cat([m3, prev_seg], 1)

        GCQ = self.GQ(r3es, key)

        segscore,Diff_heatmap,Tmp_heatmap,update = self.decoder(r4e, GCQ, r2e, prev_seg, m4s)

        pre_segemantation = F.softmax(Diff_heatmap, dim=1)  # 该帧de关于u8预测

        ps = F.softmax(segscore, dim=1)  # 作为记忆损失的前一帧分割
        pse = self.get_init_state(ps)

        temp_seg = torch.zeros_like(prev_seg)
        temp_seg[:, :2, :, :] = pse
        temp_seg[:, 2:4, :, :] = prev_seg[:, :2, :, :]
        temp_seg[:, 4:, :, :] = prev_seg[:, 2:4, :, :]

        curr_gc = self.GCM1(self.addCoords(r3e), temp_seg)

        curr_lt = key * (1 / (update + 1)) + curr_gc * (update / (update + 1))

        psd = F.softmax(segscore, dim=1)[:, 1]  # no, h, w
        logit = Soft_aggregation(psd, num_objects)



        return logit, pre_segemantation, curr_lt,  ps, temp_seg, r3ex,Tmp_heatmap, m4

    # def segment_simple(self, frame, num_objects):
    #     # segment one input frame
    #
    #     feat2, feat4, feat8, feat16, feat8_up, feat16_up, feat32_up = self.Encoder_Q(frame)
    #     # r2 = self.ffm4(feat4, feat8_up)#7.7
    #     r3 = self.ffm8(feat8, feat16_up)
    #     r4 = self.ffm16(feat16, feat32_up)
    #
    #     r4 = self.attention(r4)
    #     r3s = self.conv_out16(r3)
    #
    #     r3ex = r3s.expand(num_objects, -1, -1, -1)
    #
    #     r4 = self.convX(r4)
    #
    #     simple = self.simple_out_4(r4)
    #     pred_simple = F.interpolate(simple, size=frame.shape[2:], mode='bicubic', align_corners=True)
    #     pred_simple = torch.clamp(pred_simple, 1e-7, 1 - 1e-7)
    #     pred_simple = torch.log(pred_simple / (1 - pred_simple))
    #     pred_simple = torch.softmax(pred_simple, dim=1)
    #
    #     pred_simples, _ = self.make_T(pred_simple, num_objects)
    #
    #     pred_seg = self.get_init_state(pred_simples)
    #
    #     temp_seg = torch.zeros_like(pred_seg)
    #     prev_seg = torch.cat([temp_seg, temp_seg, temp_seg], dim=1)
    #     prev_seg[:, :2, :, :] = pred_seg
    #
    #
    #
    #
    #
    #     return prev_seg,r3ex,pred_seg,pred_simples,pred_simple
    def segment_simple(self, frame,num_objects):
        # segment one input frame

        feat2, feat4, feat8, feat16, feat8_up, feat16_up, feat32_up = self.Encoder_Q(frame)

        r3 = self.ffm8(feat8, feat16_up)
        r4 = self.ffm16(feat16, feat32_up)

        r3s = self.conv_out16(r3)
        r3ex = r3s.expand(num_objects, -1, -1, -1)

        # r3 = self.attentionr3(r3)
        # r3 = self.convXr3(r3)
        # simple3 = self.simple_out_5(r3s)
        # pred_simple3 = F.interpolate(simple3, size=frame.shape[2:], mode='bicubic', align_corners=True)
        # pred_simple3 = torch.clamp(pred_simple3, 1e-7, 1 - 1e-7)
        # pred_simple3 = torch.log(pred_simple3 / (1 - pred_simple3))
        # pred_simple3 = torch.softmax(pred_simple3, dim=1)

        # r3ex = r3.expand(num_objects, -1, -1, -1)

        r4 = self.attention(r4)
        r4 = self.convX(r4)
        simple = self.simple_out(r4)
        pred_simple = F.interpolate(simple, size=frame.shape[2:], mode='bicubic', align_corners=True)
        pred_simple = torch.clamp(pred_simple, 1e-7, 1 - 1e-7)
        pred_simple = torch.log(pred_simple / (1 - pred_simple))
        pred_simple = torch.softmax(pred_simple, dim=1)
        #
        # r2 = self.attention2(r2)
        # r2 = self.convX_2(r2)
        # simple2 = self.simple_out_4(r2)
        # pred_simple2 = F.interpolate(simple2, size=frame.shape[2:], mode='bicubic', align_corners=True)
        # pred_simple2 = torch.clamp(pred_simple2, 1e-7, 1 - 1e-7)
        # pred_simple2 = torch.log(pred_simple2 / (1 - pred_simple2))
        # pred_simple2 = torch.softmax(pred_simple2, dim=1)


        pred_simples,_=self.make_T(pred_simple,num_objects)

        pred_seg = self.get_init_state(pred_simples)

        temp_seg = torch.zeros_like(pred_seg)
        prev_seg = torch.cat([temp_seg, temp_seg, temp_seg], dim=1)
        prev_seg[:, :2, :, :] = pred_seg



        return prev_seg,r3ex,pred_seg,pred_simples,pred_simple

    def forward(self, mode, *args, **kwargs):
        if mode == 'memorize':
            return self.memorize(*args, **kwargs)
        elif mode == 'segment_simple':
            return self.segment_simple(*args, **kwargs)
        elif mode == 'make_T':
            return self.make_T(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        elif mode == 'heatmap_loss':
            return self.Heatmap_loss(*args, **kwargs)
        elif mode == 'transform_heatmap':
            return self.transform_heatmap(*args, **kwargs)
        elif mode == 'tmp_loss':
            return self.tem_Loss(*args, **kwargs)
        elif mode == 'segment2':
            return self.segment2(*args, **kwargs)
        elif mode == 'pre_Heatmap':
            return self.pre_Heatmap(*args, **kwargs)
        elif mode == 'detail_loss':
            return self.detail_loss(*args, **kwargs)
        else:
            raise NotImplementedError










