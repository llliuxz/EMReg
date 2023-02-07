# encoding: utf-8

import h5py 
import numpy as np
import math
import torch
import itertools
from grid_sample import grid_sample
from torch.autograd import Variable
from tps_grid_gen import TPSGridGen
from collections import defaultdict
import torch.nn.functional as F

def create_flitter(control_points, flitter):
    height = int(math.sqrt(control_points.size(0)))
    control_points = control_points.view(-1, height, 2)
    flitter_metric = torch.Tensor(control_points.size(0)-2, control_points.size(1)-2, 2).uniform_(-flitter, flitter)
    background = torch.Tensor(control_points.size()).uniform_(0, 0)
    background[1:-1,1:-1,:] = flitter_metric
    return background.view(-1, 2)

grid_width = grid_height = 4
flitter = 0.2

step = 1.
repeat_num = 5
offset = 0.00001
bounded_stn = False

target_control_points = torch.Tensor(list(itertools.product(
torch.arange(-step, step + offset, 2.*step / grid_height),
torch.arange(-step, step + offset, 2.*step / grid_width),
)))


file = '/data/liuxz/cremi_regis/sample_A.hdf'

img = defaultdict(list)
label = defaultdict(list)
offset = []

with h5py.File(file,'r') as f: 
        target_height, target_width = 512, 512
        tps = TPSGridGen(target_height, target_width, target_control_points, False)

	    # for idx, img in  enumerate(f['volumes']['labels']['neuron_ids']):
        for idx, fixed_img in enumerate(f['volumes']['raw']):
            if idx+1 == len(f['volumes']['raw']):
                break
            
            new_anchors = np.random.randint(0, 624, (repeat_num, 2))
            for j, anchor in enumerate(new_anchors):
                fixed_patch = fixed_img[anchor[0]:anchor[0]+512, 
                                        anchor[1]:anchor[1]+512]   # 512,512

                reference_img = f['volumes']['raw'][idx+1][anchor[0]:anchor[0]+512, 
                                                           anchor[1]:anchor[1]+512]   # 512,512
                
                moving_img = np.array(reference_img).astype('float32')
                moving_img = np.expand_dims(moving_img, 2)
                moving_img = np.expand_dims(moving_img.swapaxes(2, 1).swapaxes(1, 0), 0)
                moving_img = Variable(torch.from_numpy(moving_img))
            
                fixed_label = f['volumes']['labels']['neuron_ids'][idx][anchor[0]:anchor[0]+512, 
                                                                        anchor[1]:anchor[1]+512]   # 512,512

                reference_label = f['volumes']['labels']['neuron_ids'][idx+1][anchor[0]:anchor[0]+512, 
                                                                              anchor[1]:anchor[1]+512]   # 512,512
                moving_label = np.array(reference_label).astype('float32')
                moving_label = np.expand_dims(moving_label, 2)
                moving_label = np.expand_dims(moving_label.swapaxes(2, 1).swapaxes(1, 0), 0)
                moving_label = Variable(torch.from_numpy(moving_label))


        
                if bounded_stn:
                    flit = create_flitter(target_control_points, flitter)
                    source_control_points = target_control_points + flit
                else:
                    flit = torch.Tensor(4, 2).uniform_(-flitter, flitter)
                   
                    flit_resize = flit.view(-1)
                   
                    dx = flit_resize[0::2].reshape(2,2)
                    dy = flit_resize[1::2].reshape(2,2)
                   
                    dx_resize = F.interpolate(dx.unsqueeze(0).unsqueeze(0), (5, 5), mode='bilinear', align_corners=True).squeeze()
                    dy_resize = F.interpolate(dy.unsqueeze(0).unsqueeze(0), (5, 5), mode='bilinear', align_corners=True).squeeze()
                    
                    z = torch.Tensor(5,5,2)
                    z[:,:,0] = dx_resize
                    z[:,:,1] = dy_resize
                    flit = z.view(-1,2)

                    flit_tps = torch.Tensor(target_control_points.size()).uniform_(-0.05, 0.05)


                    source_control_points = target_control_points + flit + flit_tps
                source_coordinate = tps(Variable(torch.unsqueeze(source_control_points, 0)))
                grid = source_coordinate.view(1, target_height, target_width, 2)

            
                warped_image = torch.clamp(torch.round(grid_sample(moving_img, grid, mode='bilinear', padding_mode='zeros')), 0, 255).squeeze().data.numpy().astype('uint64')


                warped_label = grid_sample(moving_label, grid, mode='nearest', padding_mode='zeros').squeeze().data.numpy().astype('uint64')

                # plt.imshow(fixed_img, cmap='gray')
                # plt.axis(False)
                # plt.savefig('./fixed.png')

                # plt.imshow(warped_image, cmap='gray')
                # plt.axis(False)
                # plt.savefig('./warped.png')
                
                # plt.imshow(reference_img, cmap='gray')
                # plt.axis(False)
                # plt.savefig('./reference.png')
                # exit()
                
                img['fixed'].append(fixed_patch)
                img['moving'].append(reference_img)
                img['warped'].append(warped_image)

                label['fixed'].append(fixed_label)
                label['moving'].append(reference_label)
                label['warped'].append(warped_label)
                offset.append(flit.tolist())

            print('img %d warp finished...'%(idx+1))

with h5py.File('cremi/train_B_affine_tps.h5','w') as f5:
    f5.create_dataset('dvf', data=offset)      

    imgs = f5.create_group('image')
    labels = f5.create_group('label')
    
    imgs.create_dataset('fixed', data=img['fixed'])
    imgs.create_dataset('moving', data=img['moving'])
    imgs.create_dataset('warped', data=img['warped'])
    
    labels.create_dataset('fixed', data=label['fixed'])
    labels.create_dataset('moving', data=label['moving'])
    labels.create_dataset('warped', data=label['warped'])      
    

    f5.close()
    print('\nall finished......')
