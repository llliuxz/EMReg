#!/usr/bin/env python

from operator import truediv
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import bilinear_sampler, upflow2, corr_local
from .update import BasicUpdateBlock
from .correlation import global_correlation
import numpy as np
import random
from .attention import cbam

use_gpu = torch.cuda.is_available()

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)    # Using multi-GPU
np.random.seed(seed)                # Numpy module
random.seed(seed)                   # Python random module
torch.manual_seed(seed)

torch.backends.cudnn.enabled = True 
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

class PWCNet(nn.Module):
    def __init__(self, radius):
        super(PWCNet, self).__init__()

        class Extractor(nn.Module):
            def __init__(self):
                super().__init__()

                def convBlock(inchannel, outchannel):
                    return nn.Sequential(
                    nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=False)
                    )

                self.netOne = convBlock(1, 8)
                self.netTwo = convBlock(8, 16)
                self.netThr = convBlock(16, 32)
                self.netFou = convBlock(32, 64)
                self.netFiv = convBlock(64, 128)
                self.netSix = convBlock(128, 196)       # 1/64 of the original resolution 

                self.attention1 = cbam(8)
                self.attention2 = cbam(16)
                self.attention3 = cbam(32)
                self.attention4 = cbam(64)
                self.attention5 = cbam(128)
                self.attention6 = cbam(196)

            def forward(self, tenInput1, tenInput2):
                tenOne1 = self.netOne(tenInput1)
                tenOne2 = self.netOne(tenInput2)
                tenOne1, tenOne2 = self.attention1(torch.cat([tenOne1, tenOne2], dim=1))
                
                tenTwo1 = self.netTwo(tenOne1)
                tenTwo2 = self.netTwo(tenOne2)
                tenTwo1, tenTwo2 = self.attention2(torch.cat([tenTwo1, tenTwo2], dim=1))
                
                tenThr1 = self.netThr(tenTwo1)
                tenThr2 = self.netThr(tenTwo2)
                tenThr1, tenThr2 = self.attention3(torch.cat([tenThr1, tenThr2], dim=1))

                
                tenFou1 = self.netFou(tenThr1)
                tenFou2 = self.netFou(tenThr2)
                tenFou1, tenFou2 = self.attention4(torch.cat([tenFou1, tenFou2], dim=1))
                
                tenFiv1 = self.netFiv(tenFou1)
                tenFiv2 = self.netFiv(tenFou2)
                tenFiv1, tenFiv2 = self.attention5(torch.cat([tenFiv1, tenFiv2], dim=1))
              
                tenSix1 = self.netSix(tenFiv1)            # dimension； 196 channels  
                tenSix2 = self.netSix(tenFiv2)            # dimension； 196 channels  
                tenSix1, tenSix2 = self.attention6(torch.cat([tenSix1, tenSix2], dim=1))

                return [tenOne1, tenTwo1, tenThr1, tenFou1, tenFiv1, tenSix1], [tenOne2, tenTwo2, tenThr2, tenFou2, tenFiv2, tenSix2]
   
        class GRUUnit(nn.Module):
            def __init__(self, feature_dim, radius, up=2):
                super().__init__()
                self.gruunit = BasicUpdateBlock((2*radius + 1)**2, feature_dim, up)
                self.radius = radius

            def initialize_flow(self, img):
                N, _, H, W = img.shape

                def coords_grid(batch, ht, wd, device):
                    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
                    coords = torch.stack(coords[::-1], dim=0).float()
                    return coords[None].repeat(batch, 1, 1, 1)

                coords0 = coords_grid(N, H, W, device=img.device)
                coords1 = coords_grid(N, H, W, device=img.device)
                return coords0, coords1

            def upsample_flow(self, flow, mask):
                """ Upsample flow field [H/n, W/n, 2] -> [H, W, 2] using convex combination """
                N, _, H, W = flow.shape
                up = int((mask.size(1)//9)**0.5)
                mask = mask.view(N, 1, 9, up, up, H, W)
                mask = torch.softmax(mask, dim=2)

                up_flow = F.unfold(up * flow, [3,3], padding=1)
                up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

                up_flow = torch.sum(mask * up_flow, dim=2)
                up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
                return up_flow.reshape(N, 2, up*H, up*W)

            def lookup_correlation_global(self, coords, correlation):
                r = self.radius
                coords = coords.permute(0, 2, 3, 1) # (b,2,h,w) -> (b,h,w,2) 当前坐标，包含x和y两个方向，由 meshgrid() 函数得到
                batch, h1, w1, _ = coords.shape

                dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
                dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
                delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)    # 查找窗 (2r+1,2r+1,2)
                centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) 
                delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
                coords_lvl = centroid_lvl + delta_lvl   # (bhw,1,1,2) + (1,2r+1,2r+1,2) -> (bhw,2r+1,2r+1,2)

                # correaltion: (bhw,1,h,w) 相关性查找表
                corr = bilinear_sampler(correlation, coords_lvl)    # (bhw,1,2r+1,2r+1) 在查找表上搜索每个点的邻域特征，获得相关性图
                corr = corr.view(batch, h1, w1, -1)                 # (bhw,1,2r+1,2r+1) -> (b,h,w,(2r+1)*(2r+1))

                return corr.permute(0, 3, 1, 2).contiguous().float()

            def lookup_correlation_local(self, coords, correlation):
                radius = self.radius
                fmap1, fmap2 = correlation   
                coords = coords.permute(0, 2, 3, 1)
                batch, h1, w1, _ = coords.shape

                dx = torch.linspace(-radius, radius, 2*radius+1)
                dy = torch.linspace(-radius, radius, 2*radius+1)
                delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)
                centroid_lvl = coords.reshape(batch, h1, w1, 1, 2).clone()       
                coords_lvl = centroid_lvl + delta.view(-1, 2)
                corr = corr_local(fmap1, fmap2, coords_lvl)

                return corr.permute(0, 3, 1, 2).contiguous().float()

            def forward(self, fmap, flow_init, costvolume, iter):
                coords0, coords1 = self.initialize_flow(fmap)
                if flow_init is not None:
                    coords1 = coords1 + flow_init

                channel_dim = fmap.size(1)
                net, inp = torch.split(fmap, [channel_dim//2, channel_dim//2], dim=1)
                net = torch.tanh(net)
                inp = torch.relu(inp)

                for itr in range(iter):
                    coords1 = coords1.detach()
                    corr = self.lookup_correlation_global(coords1, costvolume)
                    flow = coords1 - coords0

                    net, up_mask, delta_flow = self.gruunit(net, inp, corr, flow)

                    coords1 = coords1 + delta_flow
                
                if up_mask is None:
                    flow_up = upflow2(coords1 - coords0)
                else:
                    flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                
                return flow_up, coords1 - coords0     

        class Refiner(nn.Module):
            def __init__(self, inchannel, outchannel):
                super().__init__()

                self.netMain = nn.Sequential(
                    nn.Conv2d(in_channels=inchannel, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=32, out_channels=outchannel, kernel_size=3, stride=1, padding=1, dilation=1)
                )

            def forward(self, tenInput):
                return self.netMain(tenInput)
           
        self.netExtractor = Extractor()

        self.gru1 = GRUUnit(196, radius)
        self.gru2 = GRUUnit(128, radius)
        self.gru3 = GRUUnit(64, radius)
        self.gru4 = GRUUnit(32, radius)
        self.gru5 = GRUUnit(16, radius)
        # self.gru6 = GRUUnit(8, radius)

        self.refine = nn.Sequential(
                    nn.Conv2d(in_channels=18, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1)
                )
        
    def compute_correlation(self, fmaps1, fmaps2):
        corr_pyramid = []

        # corr0 = local_correlation(fmaps1[0], fmaps2[0], 5, cupy_acc=False)
        # corr_pyramid.append(corr0)

        # for i in range(len(fmaps1)):
        for i in range(1, 6):
            fmap1 = fmaps1[i]
            fmap2 = fmaps2[i]
            corr = global_correlation(fmap1, fmap2)
            corr_pyramid.append(corr)

        return corr_pyramid   # (bhw,1,h,w); one, two, three...

    def forward(self, tenOne, tenTwo, iters):
        # extract the features
        tenOne, tenTwo = self.netExtractor(tenOne, tenTwo)

        # import cv2
        # for idx, ten in enumerate(tenOne):
        #     feature = ten[0][4]
        #     regimg = ((np.array(feature.cpu().data.numpy().squeeze())*0.5 + 0.5)*255).astype(np.uint8)
        #     regimg = np.asarray(regimg) 
        #     cv2.imwrite('/code/output/feature_fix_{}.png'.format(256/(2**idx)), regimg)
        
        # for idx, ten in enumerate(tenTwo):
        #     feature = ten[0][4]
        #     regimg = ((np.array(feature.cpu().data.numpy().squeeze())*0.5 + 0.5)*255).astype(np.uint8)
        #     regimg = np.asarray(regimg) 
        #     cv2.imwrite('/code/output/feature_mov_{}.png'.format(256/(2**idx)), regimg)



        # tenTwo = self.netExtractor(tenTwo)
        costvolume = self.compute_correlation(tenOne, tenTwo)

        flow6, _ = self.gru1(tenOne[5], None,  costvolume[-1], iters[0])    # 16
        flow5, _ = self.gru2(tenOne[4], flow6, costvolume[-2], iters[1])    # 32
        flow4, _ = self.gru3(tenOne[3], flow5, costvolume[-3], iters[2])    # 64
        flow3, _ = self.gru4(tenOne[2], flow4, costvolume[-4], iters[3])    # 128
        flow2, _ = self.gru5(tenOne[1], flow3, costvolume[-5], iters[4])    # 256

        flow1 = self.refine(torch.cat([tenOne[0], tenTwo[0], flow2], 1))
        flow = nn.functional.interpolate(input=flow1, size=(512, 512), mode='bilinear', align_corners=False) * 2
        
        # return flow.permute(0,2,3,1)
        # return flow.permute(0,2,3,1), flow1.permute(0,2,3,1), flow3.permute(0,2,3,1), flow4.permute(0,2,3,1), flow5.permute(0,2,3,1), flow6.permute(0,2,3,1)

        flow16 = nn.functional.interpolate(input=flow6, size=(512, 512), mode='bilinear', align_corners=False) * 32
        flow32 = nn.functional.interpolate(input=flow5, size=(512, 512), mode='bilinear', align_corners=False) * 16
        flow64 = nn.functional.interpolate(input=flow4, size=(512, 512), mode='bilinear', align_corners=False) * 8
        flow128 = nn.functional.interpolate(input=flow3, size=(512, 512), mode='bilinear', align_corners=False) * 4

        return flow.permute(0,2,3,1), flow128.permute(0,2,3,1), flow64.permute(0,2,3,1), flow32.permute(0,2,3,1), flow16.permute(0,2,3,1)

class Regis(nn.Module):
    def __init__(self, iters=1, radius=10):
        super(Regis, self).__init__()
        self.pwcnet = PWCNet(radius)
        self.iter = iters
        if use_gpu:
            self.pwcnet = self.pwcnet.cuda()
    
    def forward(self, imgFix, imgMov):
        iters = [self.iter] * 5
        # iters = [4,1,1,1,1]
        # iters = list(range(5,0, -1))
        dvf = self.pwcnet(imgFix.permute(0,3,1,2), imgMov.permute(0,3,1,2), iters)
        '''
        imgfix:     [N, 512, 512, 1]
        dvf:        [N, 512, 512, 2]
        '''
        return dvf
