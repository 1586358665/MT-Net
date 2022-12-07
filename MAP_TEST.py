#-*-coding：utf-8-*-
from libs.dataset.data import ROOT, DATA_CONTAINER, multibatch_collate_fn
from libs.dataset.transform import TrainTransform, TestTransform
from libs.utils.logger import Logger, AverageMeter
from libs.utils.loss import *

from libs.utils.utility import write_mask, save_checkpoint, adjust_learning_rate
from libs.models.STCN import MT
# from beifen_STM import STM
from libs.models.Att1 import Att
from apex import amp


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import os
import os.path as osp
import shutil
import time
import pickle
import argparse
import random
from progress.bar import Bar
from collections import OrderedDict
from random import shuffle

from options import OPTION as opt
# from ipdb import set_trace
from libs.dataset.data import ROOT, DATA_CONTAINER, multibatch_collate_fn
from libs.dataset.transform import TrainTransform, TestTransform
from libs.utils.logger import Logger, AverageMeter
from libs.utils.loss import *
from libs.utils.utility import write_mask, save_checkpoint, adjust_learning_rate, mask_iou
from libs.models.STCN import MT,draw_features_plot
# from try4 import STM
from libs.models.Att1 import Att

from libs.utils.loss import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import os
import os.path as osp
import shutil
import time
import pickle
from progress.bar import Bar
from collections import OrderedDict

from options import OPTION as opt
from matplotlib import pyplot as plt
import time
from random import shuffle
MAX_FLT = 1e6
from libs.dataset.transform import TrainTransform
from libs.dataset.data import *
from libs.dataset.try2 import get_data
import cv2
# 图片预处理



#
# fmap_block = []
# grad_block = []




# 定义获取梯度的函数
def backward_hook1(module, grad_in, grad_out):
    grad_block1.append(grad_out[0].detach())

def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output[0].detach())


# 计算grad-cam并可视化
def cam_show_img(img, feature_map, grads, out_dir,num):
    H, W,_ = img.shape
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

    path_cam_img = os.path.join(out_dir, "cam%d.jpg"%(num))
    cv2.imwrite(path_cam_img, cam_img)
def train( model, Att_model,  criterion, frames,masks,  use_cuda,objs):
    # switch to train mode
    with torch.cuda.device(0):
        objs = np.array(objs)
        objs = torch.from_numpy(objs)
        if use_cuda:
            frame = frames.cuda()
            mask = masks.cuda()
            objs= objs.cuda()
            # em = torch.zeros(1, objs + 1, H, W).cuda()



            T, C, H, W = frame.size()

            t_loss=0.0

            t_loss1 = 0.0


            num_objects = objs
            keys = []
            vals = []
            keys3 = []
            vals3 = []
            PS = []
            Pred_L = []

            last = []
            temp = []
            prev_segs_heatmap =[]
            GT =[]
            GC=[]
            M4 = []



            #pre-save
            for t in range(opt.save_freq):
                if t == 0:

                    prev_seg,r3,pred_seg,pred_simples,pred_simple2= model('segment_simple', frame=frame[t:t + 1], num_objects=num_objects)




                    key, val, _, key3, val3, _= model("memorize",frame=frame[t:t + 1], masks=mask[t:t+1],num_objects=num_objects,R3=r3)

                    keys.append(key)
                    vals.append(val)

                    keys3.append(key3)
                    vals3.append(val3)

                    prev_segs_heatmap.append(prev_seg)



                    gt = mask[t:t + 1]
                    t_loss1 = t_loss1 + criterion(pred_simple2, gt, num_objects)



                else:



                    prev_seg,r3,pred_seg,pred_simples,pred_simple2= model('segment_simple', frame=frame[t:t + 1], num_objects=num_objects)

                    pred_segs = model('pre_Heatmap',prev_seg= prev_segs_heatmap[-1], pse=pred_seg)




                    key, val, _, key3, val3, _ = model("memorize", frame=frame[t:t + 1], masks=mask[t:t + 1], num_objects=num_objects, R3=r3)

                    keys.append(key)
                    vals.append(val)

                    keys3.append(key3)
                    vals3.append(val3)

                    prev_segs_heatmap.append(pred_segs)

                    if t == opt.save_freq-1:
                        pred_simples, _ = model('make_T', masks=mask[t:t + 1], num_objects=num_objects)
                        PS.append(pred_simples)

                    gt = mask[t:t + 1]
                    t_loss1 = t_loss1 + criterion(pred_simple2, gt, num_objects)



            for t in range(opt.save_freq, T):
                tmp_key_local = torch.stack(keys[-opt.save_freq:])
                tmp_val_local = torch.stack(vals[-opt.save_freq:])

                tmp_key_local3 = torch.stack(keys3[-opt.save_freq:])
                tmp_val_local3 = torch.stack(vals3[-opt.save_freq:])

                shuffle_keys = keys.copy()
                shuffle_vals = vals.copy()
                shuffle(shuffle_keys)
                shuffle(shuffle_vals)
                tmp_key_global = torch.stack(shuffle_keys[-opt.save_freq:])
                tmp_val_global = torch.stack(shuffle_vals[-opt.save_freq:])

                shuffle_keys3 = keys3.copy()
                shuffle_vals3 = vals3.copy()
                shuffle(shuffle_keys3)
                shuffle(shuffle_vals3)
                tmp_key_global3 = torch.stack(shuffle_keys3[-opt.save_freq:])
                tmp_val_global3 = torch.stack(shuffle_vals3[-opt.save_freq:])

                #attention
                tmp_key_local = Att_model(f=tmp_key_local,tag='att_in_local')
                tmp_val_local = Att_model(f=tmp_val_local,tag='att_out_local')
                tmp_key_global = Att_model(f=tmp_key_global,tag='att_in_global')
                tmp_val_global = Att_model(f=tmp_val_global,tag='att_out_global')

                tmp_key_local3 = Att_model(f=tmp_key_local3,tag='att_in_local3')
                tmp_val_local3 = Att_model(f=tmp_val_local3,tag='att_out_local3')
                tmp_key_global3 = Att_model(f=tmp_key_global3,tag='att_in_global3')
                tmp_val_global3 = Att_model(f=tmp_val_global3,tag='att_out_global3')

                tmp_key = tmp_key_local+tmp_key_global
                tmp_val = tmp_val_local+tmp_val_global

                tmp_key3 = tmp_key_local3+tmp_key_global3
                tmp_val3 = tmp_val_local3+tmp_val_global3

                #segment
                GT_heatmap, mask_mini= model("make_T", masks=mask[t:t + 1], num_objects=num_objects)

                GT.append(mask_mini)

                if t == opt.save_freq:
                    logit,pre_segemantation, curr_lt, ps,temp_seg ,r3,Tmp_heatmap, m4= model("segment", frame=frame[t:t + 1], keys=tmp_key, values=tmp_val,
                                                      keys3=tmp_key3, values3=tmp_val3,num_objects=num_objects, prev_seg=prev_segs_heatmap[-1])

                    GC.append(curr_lt)
                    M4.append(m4)


                if t != opt.save_freq:
                    logit,pre_segemantation, curr_lt, ps,temp_seg,r3,Tmp_heatmap , m4= model("segment2", frame=frame[t:t + 1], keys=tmp_key, values=tmp_val,
                                                                                     keys3=tmp_key3, values3=tmp_val3,num_objects=num_objects,
                                                                                     prev_seg=prev_segs_heatmap[-1],key=GC[-1],m4s=M4[-1])
                    GC.append(curr_lt)
                    M4.append(m4)


                out = torch.softmax(logit, dim=1)
                Pred_L.append(pre_segemantation)

                temp.append(Tmp_heatmap)
                prev_segs_heatmap.append(temp_seg)

                # temp_batch=torch.stack(temp, dim=0)




                # memorize
                with torch.no_grad():
                    key, val, _, key3, val3, _ = model("memorize", frame=frame[t:t + 1], masks=out,
                                                       num_objects=num_objects,R3=r3)
                    keys.append(key)
                    vals.append(val)

                    keys3.append(key3)
                    vals3.append(val3)

                    if t > opt.save_freq_max:
                        keys.pop(0)
                        vals.pop(0)
                        keys3.pop(0)
                        vals3.pop(0)

                        last.pop(0)
                        prev_segs_heatmap.pop(0)

                gt = mask[t:t + 1]
                t_loss = t_loss + criterion(out, gt,num_objects)



            loss_1 = t_loss /  (T-5)


            total_loss=loss_1

        return total_loss, keys,vals,tmp_val
def parse_args():
    parser = argparse.ArgumentParser('Training Mask Segmentation')
    parser.add_argument('--gpu', default='1', type=str, help='set gpu id to train the network, split with comma')
    return parser.parse_args()

def img_preprocess(img_in):
    img = img_in.copy()
    img = img[:, :, ::-1]  # 1
    img = np.ascontiguousarray(img)  # 2
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # 3
    return img


if __name__ == '__main__':
    # path_img = './cam/bicycle.jpg'
    frame_path = ['./dataset/VIT/VIL/JPEGImages/2_Road017_Trim004_frames/00045.jpg',
                       './dataset/VIT/VIL/JPEGImages/2_Road017_Trim004_frames/00048.jpg',
                       './dataset/VIT/VIL/JPEGImages/2_Road017_Trim004_frames/00051.jpg',
                       './dataset/VIT/VIL/JPEGImages/2_Road017_Trim004_frames/00054.jpg',
                       './dataset/VIT/VIL/JPEGImages/2_Road017_Trim004_frames/00057.jpg',
                       './dataset/VIT/VIL/JPEGImages/2_Road017_Trim004_frames/00060.jpg']

    mask_path = ['./dataset/VIT/VIL/Annotations/2_Road017_Trim004_frames/00045.png',
                      './dataset/VIT/VIL/Annotations/2_Road017_Trim004_frames/00048.png',
                      './dataset/VIT/VIL/Annotations/2_Road017_Trim004_frames/00051.png',
                      './dataset/VIT/VIL/Annotations/2_Road017_Trim004_frames/00054.png',
                      './dataset/VIT/VIL/Annotations/2_Road017_Trim004_frames/00057.png',
                      './dataset/VIT/VIL/Annotations/2_Road017_Trim004_frames/00060.png']

    input_dim = opt.input_size
    train_transformer = TrainTransform(size=input_dim)
    transform = train_transformer
    frames, masks = get_data.start(transform,frame_path=frame_path, mask_path=mask_path)
    with torch.cuda.device(0):
        start_epoch = 0
        random.seed(0)

        args = parse_args()
        # Use GPU
        use_gpu = torch.cuda.is_available() and (args.gpu != '' or int(opt.gpu_id)) >= 0
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.gpu != '' else str(opt.gpu_id)
        gpu_ids = [int(val) for val in args.gpu.split(',')]

        if not os.path.isdir(opt.checkpoint):
            os.makedirs(opt.checkpoint)

        # Data
        print('==> Preparing dataset')

        input_dim = opt.input_size

        train_transformer = TrainTransform(size=input_dim)
        test_transformer = TestTransform(size=input_dim)

        try:
            if isinstance(opt.trainset, list):
                datalist = []
                for dataset, freq, max_skip in zip(opt.trainset, opt.datafreq, opt.max_skip):
                    ds = DATA_CONTAINER[dataset](
                        train=True,
                        sampled_frames=opt.sampled_frames,
                        transform=train_transformer,
                        max_skip=max_skip,
                        samples_per_video=opt.samples_per_video
                    )
                    datalist += [ds] * freq

                trainset = data.ConcatDataset(datalist)

            else:
                max_skip = opt.max_skip[0] if isinstance(opt.max_skip, list) else opt.max_skip
                trainset = DATA_CONTAINER[opt.trainset](
                    train=True,
                    sampled_frames=opt.sampled_frames,
                    transform=train_transformer,
                    max_skip=max_skip,
                    samples_per_video=opt.samples_per_video
                )
        except KeyError as ke:
            print('[ERROR] invalide dataset name is encountered. The current acceptable datasets are:')
            print(list(DATA_CONTAINER.keys()))
            exit()

        testset = DATA_CONTAINER[opt.valset](
            train=False,
            transform=test_transformer,
            samples_per_video=1
        )



        # Model


        net = MT(opt.keydim, opt.valdim, 'train',
                  mode=opt.mode, iou_threshold=opt.iou_threshold)
        net.eval()

        if use_gpu:
            net = net.cuda()

        att = Att(save_freq=opt.save_freq, keydim=opt.keydim, valdim=opt.valdim)
        att.eval()
        if use_gpu:
            att = att.cuda()


        # set training parameters
        for p in net.Encoder_M.parameters():
            p.requires_grad = True
        for p in net.Encoder_Q.parameters():
            p.requires_grad = True
        for p in net.KV_M_r41.parameters():
            p.requires_grad = True
        for p in net.KV_M_r31.parameters():
            p.requires_grad = True
        for p in net.KV_Q_r41.parameters():
            p.requires_grad = True
        for p in net.KV_Q_r31.parameters():
            p.requires_grad = True
        for p in net.decoder.parameters():
            p.requires_grad = True
        for p in att.parameters():
            p.requires_grad = True

        for p in net.ffm16.parameters():
            p.requires_grad = True
        for p in net.simple_out.parameters():
            p.requires_grad = True
        # for p in net.convX.parameters():
        #     p.requires_grad =True
        for p in net.attention.parameters():
            p.requires_grad = True
        for p in net.attention1.parameters():
            p.requires_grad = True



        for p in net.conv_out.parameters():
            p.requires_grad = True
        for p in net.conv_out16.parameters():
            p.requires_grad = True
        for p in net.conv_out32.parameters():
            p.requires_grad = True
        for p in net.GCM1.parameters():
            p.requires_grad = True
        for p in net.GQ.parameters():
            p.requires_grad = True

        criterion = None
        celoss = cross_entropy_loss

        if opt.loss == 'ce':
            criterion = celoss
        elif opt.loss == 'iou':
            criterion = mask_iou_loss
        elif opt.loss == 'both':
            criterion = lambda pred, target, obj: celoss(pred, target, obj) + mask_iou_loss(pred, target, obj)
        else:
            raise TypeError('unknown training loss %s' % opt.loss)

        if opt.solver == 'sgd':
            params = [{"params": net.parameters(), "lr": opt.learning_rate},
                      {"params": att.parameters(), "lr": opt.learning_rate}]

            optimizer = optim.SGD(params, lr=opt.learning_rate,
                                  momentum=opt.momentum[0], weight_decay=opt.weight_decay)
        elif opt.solver == 'adam':

            params = [{"params": net.parameters(), "lr": opt.learning_rate},
                      {"params": att.parameters(), "lr": opt.learning_rate}]
            optimizer = optim.Adam(params, betas=opt.momentum, weight_decay=opt.weight_decay)
        else:
            raise TypeError('unkown solver type %s' % opt.solver)

        # Resume
        title = 'STM'
        minloss = float('inf')

        opt.checkpoint_STM = osp.join(osp.join(opt.checkpoint, opt.valset, opt.setting, 'STM'))
        opt.checkpoint_att = osp.join(osp.join(opt.checkpoint, opt.valset, opt.setting, 'ATT'))
        if not osp.exists(opt.checkpoint_STM):
            os.makedirs(opt.checkpoint_STM)
        if not osp.exists(opt.checkpoint_att):
            os.makedirs(opt.checkpoint_att)

        if opt.initial_STM:
            print('==> Resuming from checkpoint {}'.format(opt.initial_STM))
            assert os.path.isfile(opt.initial_STM), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(opt.initial_STM)
            state = checkpoint['state_dict']
            net.load_param(state)
        elif opt.resume_STM:
            # Load checkpoint.
            print('==> Resuming from pretrained {}'.format(opt.resume_STM))
            assert os.path.isfile(opt.resume_STM), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(opt.resume_STM)
            net.load_state_dict(checkpoint['state_dict'], strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'])

        logger = Logger(os.path.join(opt.checkpoint, opt.mode + '_log.txt'), resume=True)

        if opt.resume_ATT:
            # Load checkpoint.
            print('==> Resuming from checkpoint {}'.format(opt.resume_ATT))
            assert os.path.isfile(opt.resume_ATT), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(opt.resume_ATT)
            minloss = checkpoint['minloss']
            start_epoch = checkpoint['epoch']
            att.load_state_dict(checkpoint['state_dict'], strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            skips = checkpoint['max_skip']


        # Train and val
        fmap_block = []
        grad_block = []
        grad_block1 = []

        # print(att.named_modules)

        # net.KV_M_r41.Value.register_forward_hook(farward_hook)  # 9
        # net.KV_M_r41.Value.register_backward_hook(backward_hook)
        # models._modules["0"]._modules.get("conv").register_forward_hook(hook_feature)
        # att.Att_out_global.register_backward_hook(backward_hook)
        att.Att_out_local3.register_backward_hook(backward_hook)

        total_loss, keys,vals,tmp_val_global3 = train(
                           model=net,
                           Att_model=att,
                           criterion=criterion,
                           frames=frames,
                           masks=masks,
                           use_cuda=use_gpu,
                           objs=MAX_OBJECT
                           )


        net.zero_grad()
        att.zero_grad()

        total_loss.backward()

    # 存放梯度和特征图



        img = cv2.imread('./dataset/VIT/VIL/JPEGImages/2_Road017_Trim004_frames/00060.jpg', 1)
        img_input = img_preprocess(img)

    # 生成cam


        grad1= grad_block[0]
        # grad3 = grad_block1[0]



        lane_grad=grad1[5]

        grad2=tmp_val_global3[5]
        # feature = vals[0]
        # grad2=feature[7]

        # grad1 =grad_block[0].reshape(-1,H,W)
        # grad2 = fmap_block[0].reshape(-1, H, W)

        grads_val = lane_grad.cpu().data.numpy().squeeze()
        fmap =  grad2.cpu().data.numpy().squeeze()



    # 保存cam图片
        num=6
        PATH = os.path.join("./dataset", "heatmap")
        cam_show_img(img, fmap, grads_val, PATH,num)

