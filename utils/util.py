import itertools
from operator import truediv
import numpy as np
import cv2
import os
import torch  
import torchviz
from utils import loss, grid_sample
import torch.nn as nn
import torch.nn.functional as F
from utils.tps_grid_gen import TPSGridGen

use_gpu = torch.cuda.is_available()
            
class SpatialTransformation(nn.Module):
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        super(SpatialTransformation, self).__init__()

    def meshgrid(self, height, width):
        x_t = torch.matmul(torch.ones([height, 1]), torch.transpose(torch.unsqueeze(torch.linspace(0.0, width -1.0, width), 1), 1, 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width]))

        x_t = x_t.expand([height, width])
        y_t = y_t.expand([height, width])
        if self.use_gpu==True:
            x_t = x_t.cuda()
            y_t = y_t.cuda()

        return x_t, y_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 1, 0)
        rep = rep.long()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        if self.use_gpu:
            x = x.cuda()
        return torch.squeeze(torch.reshape(x, (-1, 1)))

    def interpolate(self, im, x, y):
        im = F.pad(im, (0,0,1,1,1,1,0,0))
        batch_size, height, width, channels = im.shape
        batch_size, out_height, out_width = x.shape
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        x = x + 1
        y = y + 1
        max_x = width - 1
        max_y = height - 1
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width*height
        base = self.repeat(torch.arange(0, batch_size)*dim1, out_height*out_width)

        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = torch.reshape(im, [-1, channels])
        im_flat = im_flat.float()
        dim, _ = idx_a.transpose(1,0).shape
        Ia = torch.gather(im_flat, 0, idx_a.transpose(1,0).expand(dim, channels))
        Ib = torch.gather(im_flat, 0, idx_b.transpose(1,0).expand(dim, channels))
        Ic = torch.gather(im_flat, 0, idx_c.transpose(1,0).expand(dim, channels))
        Id = torch.gather(im_flat, 0, idx_d.transpose(1,0).expand(dim, channels))

        # and finally calculate interpolated values
        x1_f = x1.float()
        y1_f = y1.float()

        dx = x1_f - x
        dy = y1_f - y

        wa = (dx * dy).transpose(1,0)
        wb = (dx * (1-dy)).transpose(1,0)
        wc = ((1-dx) * dy).transpose(1,0)
        wd = ((1-dx) * (1-dy)).transpose(1,0)

        output = torch.sum(torch.squeeze(torch.stack([wa*Ia, wb*Ib, wc*Ic, wd*Id], dim=1)), 1)
        output = torch.reshape(output, [-1, out_height, out_width, channels])
        return output

    def forward(self, moving_image, deformation_matrix):
        dx = deformation_matrix[:, :, :, 0]
        dy = deformation_matrix[:, :, :, 1]
        batch_size, height, width = dx.shape
        x_mesh, y_mesh = self.meshgrid(height, width)
        x_mesh = x_mesh.expand([batch_size, height, width])
        y_mesh = y_mesh.expand([batch_size, height, width])
        x_new = dx + x_mesh
        y_new = dy + y_mesh

        return self.interpolate(moving_image, x_new, y_new)


def store_img(val_fixed, val_registered, val_moving, path_validation):
    for i in range(len(val_fixed)):
        fix = ((np.array(val_fixed[i].cpu().data.numpy().squeeze())*0.5 + 0.5)*255).astype(np.uint8)
        fix = np.asarray(fix)  
        reg = ((np.array(val_registered[i].cpu().data.numpy().squeeze())*0.5 + 0.5)*255).astype(np.uint8)
        reg = np.asarray(reg) 
        mov = ((np.array(val_moving[i].cpu().data.numpy().squeeze())*0.5 + 0.5)*255).astype(np.uint8)
        mov = np.asarray(mov)
        cv2.imwrite(os.path.join(path_validation, 'img_'+str(i+1)+'_fix.png'), fix)
        cv2.imwrite(os.path.join(path_validation, 'img_'+str(i+1)+'_reg.png'), reg)
        cv2.imwrite(os.path.join(path_validation, 'img_'+str(i+1)+'_mov.png'), mov)

def computation_graph(x, path):
    # store a computation graph
    dot = torchviz.make_dot(x)
    dot.render(filename=os.path.join(path, 'computation_graph'), view=False)

def create_out_path(out_path):
    testnum = 1
    org_path = os.path.join(out_path, 'train'+str(testnum))
    while os.path.exists(org_path):
        testnum += 1
        org_path = os.path.join(out_path, 'train'+str(testnum))
    return org_path, testnum

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# def store_dice(fixed_id, moving_id, warped_id, dvf):
#     registered_id = grid_sample(warped_id, dvf, mode='nearest', padding_mode='zeros')
#     registered_id = torch.round(registered_id)
#     dice_score_val = loss.dice_score(moving_id, registered_id)
#     dice_score_gt = loss.dice_score(fixed_id, registered_id)
#     return dice_score_val, dice_score_gt

def grid_warp_id(input_id, dvf):
    _, target_height, target_width = dvf.size()[:3]
    height = 5
    step = 1.
    offset = 0.00001

    source_control_points = torch.Tensor(list(itertools.product(
    torch.arange(-step, step+offset, 2*step / (height-1)),
    torch.arange(-step, step+offset, 2*step / (height-1)),
    )))   
    if use_gpu:  
        source_control_points = source_control_points.cuda()

    tps = TPSGridGen(target_height, target_width, source_control_points, use_gpu)      
    source_coordinate = tps(source_control_points.unsqueeze(0))
    source_grid = source_coordinate.view(1, target_height, target_width, 2)
    delta_grid = dvf*2/(target_height-1)
    grid = source_grid + delta_grid
    registered_id = F.grid_sample(input_id, grid, mode='nearest', padding_mode='zeros')
    
    return registered_id
    
def store_dice_stn(fixed_id, moving_id, warped_id, dvf):
    stn = SpatialTransformation(use_gpu)
    if use_gpu:
        stn = stn.cuda()

    registered_id = stn(warped_id.permute(0,2,3,1), torch.round(dvf)).permute(0,3,1,2) 
    # registered_id = grid_warp_id(warped_id, dvf)

    dice_score_mov = loss.dice_score(moving_id, registered_id)
    dice_score_fix = loss.dice_score(fixed_id, registered_id)

    return dice_score_mov, dice_score_fix

def store_dice_stn_twice(fixed_id, moving_id, warped_id, dvf1, dvf2):
    stn = SpatialTransformation(use_gpu)
    if use_gpu:
        stn = stn.cuda()

    registered_id0 = stn(warped_id.permute(0,2,3,1), torch.round(dvf1)).permute(0,3,1,2) 
    registered_id = stn(registered_id0.permute(0,2,3,1), torch.round(dvf2)).permute(0,3,1,2) 
    
    # registered_id0 = grid_warp_id(warped_id, dvf1)
    # registered_id  = grid_warp_id(registered_id0, dvf2)
    
    dice_score_mov = loss.dice_score(moving_id, registered_id)
    dice_score_fix = loss.dice_score(fixed_id, registered_id)

    return dice_score_mov, dice_score_fix

def weights_init(m):
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    #     nn.init.normal_(m.weight.data, 0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    #     nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     nn.init.constant_(m.bias.data, 0)

    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    #     nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
    # elif classname.find('BatchNorm') != -1:
    #     nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     nn.init.constant_(m.bias.data, 0.0)

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def upflow2(flow, mode='bilinear'):
    new_size = (2 * flow.shape[2], 2 * flow.shape[3])
    return  2 * F.interpolate(flow, size=new_size, mode=mode, align_corners=False)

def corr_local(fmap1, fmap2, coords):
    _, D, H, W = fmap2.shape

    # map grid coordinates to [-1,1]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)

    output_corr = []
    for grid_slice in grid.unbind(3):
        fmapw_mini = F.grid_sample(fmap2, grid_slice, align_corners=True)
        corr = torch.sum(fmapw_mini * fmap1, dim=1)
        output_corr.append(corr)
    corr = torch.stack(output_corr, dim=1).permute(0,2,3,1)

    return corr / torch.sqrt(torch.tensor(D).float())    

def img_warp_grid(img, dvf):
    '''img size should be [N, C, H, W], dvf size should be [N, H, W, C]'''
    '''registered img size [N, C, H, W]'''
    batch_size, target_height, target_width = dvf.size()[:3]
    height = 5
    step = 1.
    offset = 0.00001

    source_control_points = torch.Tensor(list(itertools.product(
    torch.arange(-step, step+offset, 2*step / (height-1)),
    torch.arange(-step, step+offset, 2*step / (height-1)),
    )))   
    if use_gpu:  
        source_control_points = source_control_points.cuda()

    tps = TPSGridGen(target_height, target_width, source_control_points, use_gpu)      
    source_coordinate = tps(source_control_points.unsqueeze(0))
    source_grid = source_coordinate.view(1, target_height, target_width, 2)
    delta_grid = dvf*2/(target_height-1)
    grid = source_grid + delta_grid

    canvas = torch.Tensor(batch_size, 1, target_height, target_width).fill_(-1).cuda()
    registered_img = grid_sample.grid_sample(img, grid, canvas=canvas, mode='bilinear', padding_mode='zeros')
    return registered_img

def freeze(module):
    """
    Freeze module's parameters.
    """
    
    for parameter in module.parameters():
        parameter.requires_grad = False
            
def unfreeze(module):
    """
    Unfreeze module's parameters.
    """
    
    for parameter in module.parameters():
        parameter.requires_grad = True
        
def get_freezed_parameters(module):
    """
    Returns names of freezed parameters of the given module.
    """
    
    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)
            
    return freezed_parameters