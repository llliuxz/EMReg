# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import random
import pandas as pd
import argparse
from torch.utils.tensorboard import SummaryWriter
from dataset.data_generator import PrepareH5Dataset as h5dataset
from utils import util, loss
from model.em_reg import Regis

use_gpu = torch.cuda.is_available()
print('use gpu: {}'.format(use_gpu))

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

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=10, help='batch size (default: 5)')
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=500, help='number of training epochs (default: 500)')
    parser.add_argument('--no-shuffle', action="store_true")
    parser.add_argument('--weight-smooth', type=float, default=1., help='weight of smoothing loss (default: 1.)')
    parser.add_argument('--tps-lr', type=float, default=0.1, help='learning rate of tps network parameters (default: 0.1)')
    parser.add_argument('--valnum', type=int, default=50, help='validation data num (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--loss', type=str, default='mse', help="G loss type, the implemented types are \
        'mse', 'mae', 'huber', 'kldive' and 'smoothl1' (default: mse)")
    parser.add_argument('--beta', type=float, default=0.9, help='param for adam optimizer')
    parser.add_argument('--activation', type=str, default='relu', help="activation type, the implemented types are \
        'relu', 'prelu', 'lrelu', 'elu' and 'tanh', the default type is 'relu'")
    parser.add_argument('--init', action="store_true", help='apply the initial weights')
    parser.add_argument('--grid', type=int, default=8, help='grid num')
    parser.add_argument('--worker', type=int, default=4, help='num_worker')
    parser.add_argument('--noload', action="store_false")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--data', default="cremi")
    parser.add_argument('--label', action="store_true")
    # parser.add_argument('--thre_train', type=float, default=0.85, help='dice threshold for saving models when training')
    parser.add_argument('--thre_test', type=float, default=0.85, help='dice threshold for saving models when testing')
    parser.add_argument('--thre_test_ssim', type=float, default=0.60, help='ssim threshold for saving models when testing')
    parser.add_argument('--ending', type=float, default=0.98, help='end training when "score_val >= ending"')
    parser.add_argument('--optim', type=str, default='no', help='optimizer type,the implemented types are \
        "step", "expo", "cosine", "lambda", "no"')    
    
    global args
    args = parser.parse_args()
    print(args)

if __name__ == '__main__':  
    args_parse()
    
    if args.data == "fib":
        DATA_PATH = '/data/liuxz/cremi_regis/fib25/data_fib25.h5'
    elif args.data == "fafb":
        DATA_PATH = '/data/liuxz/cremi_regis/fafb/fafb_x1074.h5'
    elif args.data == "cremi":
        DATA_PATH = '/data/liuxz/cremi_regis/train_A_8.h5'
    elif args.data == "cremi0.05":
        DATA_PATH = '/data/liuxz/cremi_regis/large_deformation/train_A_0.05.h5'
    elif args.data == "cremi0.1":
        DATA_PATH = '/data/liuxz/cremi_regis/large_deformation/train_A_0.1.h5'
    elif args.data == "cremi0.15":
        DATA_PATH = '/data/liuxz/cremi_regis/large_deformation/train_A_0.15.h5'
    elif args.data == "cremi0.2":
        DATA_PATH = '/data/liuxz/cremi_regis/large_deformation/train_A_0.2.h5'
    elif args.data == "cremi0.25":
        DATA_PATH = '/data/liuxz/cremi_regis/large_deformation/train_A_0.25.h5'
        
    out_path = '/output'
    # model_g = '/data/liuxz/cremi_regis/weights/fib1.pth'
    model_g = None
    
    org_path, _ = util.create_out_path(out_path)
    dir_checkpoint = util.create_path(org_path+'/checkpoints')
    dir_loss = util.create_path(org_path+'/log')
    dir_validation = util.create_path(org_path+'/validate')
    dir_train = util.create_path(org_path+'/train')
    dir_test = util.create_path(org_path+'/test')


    # data generator
    H5Data = h5dataset(DATA_PATH, args)
    training_generator, validation_generator, test_generator = H5Data.create()

    # Generator
    G = Regis(iters=args.iter)

    
    if use_gpu:
        G = nn.DataParallel(G).cuda()

    # load model params
    if model_g is not None:
        best_model = torch.load(model_g)
        G.load_state_dict(best_model)
        print('generator model loaded...')
    else:
        G.apply(util.weights_init)
        best_model = G.state_dict()
        print('generator model initialized...')

    G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta, 0.999))
    if args.optim == 'step':
        scheduler = optim.lr_scheduler.StepLR(G_optimizer,step_size=10, gamma=0.9)
    if args.optim == 'multi-step':
        scheduler = optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[100,150,200], gamma=0.5)
    elif args.optim == 'lambda':
        scheduler = optim.lr_scheduler.LambdaLR(G_optimizer, lambda epoch: epoch // 10)
    elif args.optim == 'expo':
        scheduler = optim.lr_scheduler.ExponentialLR(G_optimizer, gamma=0.9)
    elif args.optim == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(G_optimizer, T_max=64)
    elif args.optim == 'no':
        scheduler = None    

    if args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'mae':
        criterion = nn.L1Loss()
    elif args.loss == 'huber':
        criterion = nn.HuberLoss()
    elif args.loss == 'smoothl1':
        criterion = nn.SmoothL1Loss()
    elif args.loss == 'kldive':
        criterion = nn.KLDivLoss()
    else:
        raise Exception('Not supported loss type.')

    epochs = args.epoch
    writer_train = SummaryWriter(log_dir=util.create_path(os.path.join(dir_loss, 'train')))
    writer_val = SummaryWriter(log_dir=util.create_path(os.path.join(dir_loss, 'validation(reg-mov)')))
    writer_gt = SummaryWriter(log_dir=util.create_path(os.path.join(dir_loss, 'validation(reg-fix)')))

    ###################################################################
    # train and validate
    thre_test = args.thre_test
    thre_test_ssim = args.thre_test_ssim

    for epoch in range(epochs):
        smooth_loss, pixel_loss, loss_epoch = [], [], []
        train_mov, train_fix = [], []
        ssim_f_print, ssim_m_print = [], []

        # train the generator 
        G.train()
        time_train = 0
        time_s = time.time() 
        for data in training_generator:
            time_gpu = time.time()

            G_optimizer.zero_grad()
            if args.label:
                image, label = data
                batch_fixed, batch_reference, batch_moving = image
                fixed_id, moving_id, warped_id = label

                fixed_id = fixed_id.cuda()
                moving_id = moving_id.cuda()
                warped_id = warped_id.cuda()
            else:
                batch_fixed, batch_reference, batch_moving = data
            
            batch_fixed = batch_fixed.cuda()
            batch_reference = batch_reference.cuda()
            batch_moving = batch_moving.cuda()

            batch_fixed = batch_fixed.permute(0, 2, 3, 1)
            batch_moving = batch_moving.permute(0, 2, 3, 1) 
            
            dvf_grid, dvf_grid128, dvf_grid64, dvf_grid32, dvf_grid16 = G(batch_fixed,batch_moving)
            # dvf_grid = G(batch_fixed,batch_moving)
            img_grid = util.img_warp_grid(batch_moving.permute(0,3,1,2), dvf_grid).permute(0, 2, 3, 1)
            
            img_grid128 = util.img_warp_grid(batch_moving.permute(0,3,1,2), dvf_grid128).permute(0, 2, 3, 1)
            img_grid64 = util.img_warp_grid(batch_moving.permute(0,3,1,2), dvf_grid64).permute(0, 2, 3, 1)
            img_grid32 = util.img_warp_grid(batch_moving.permute(0,3,1,2), dvf_grid32).permute(0, 2, 3, 1)
            img_grid16 = util.img_warp_grid(batch_moving.permute(0,3,1,2), dvf_grid16).permute(0, 2, 3, 1)
            loss_pixel_grid128 = 4*criterion(img_grid128, batch_fixed)
            loss_pixel_grid64 = 4*criterion(img_grid64, batch_fixed)
            loss_pixel_grid32 = 4*criterion(img_grid32, batch_fixed)
            loss_pixel_grid16 = 4*criterion(img_grid16, batch_fixed)
            loss_smooth_grid128 = 2*loss.smoothloss(dvf_grid128)
            loss_smooth_grid64 = 2*loss.smoothloss(dvf_grid64)
            loss_smooth_grid32 = 2*loss.smoothloss(dvf_grid32)
            loss_smooth_grid16 = 2*loss.smoothloss(dvf_grid16)
            loss_pixel_grid = (loss_pixel_grid128 + 4*criterion(img_grid, batch_fixed) + loss_pixel_grid64 + loss_pixel_grid32 + loss_pixel_grid16)/5
            loss_smooth_grid = (loss_smooth_grid128+loss_smooth_grid64+loss_smooth_grid32+loss_smooth_grid16+2*loss.smoothloss(dvf_grid))/5


            loss_smooth_grid = 2*loss.smoothloss(dvf_grid)
            loss_pixel_grid = 4*criterion(img_grid, batch_fixed)

            G_train_loss_total = loss_smooth_grid + loss_pixel_grid

            G_train_loss_total.backward()
            G_optimizer.step()

            loss_epoch.append(G_train_loss_total.data.item())
            smooth_loss.append(loss_smooth_grid.data.item())
            pixel_loss.append(loss_pixel_grid.data.item())

            if args.label:                             
                score_mov, score_fix = util.store_dice_stn(fixed_id, moving_id, warped_id, dvf_grid)
                train_mov.append(score_mov)
                train_fix.append(score_fix)
            else:
                regis_permute_print = img_grid.permute(0,3,1,2)
                SSIM = loss.SSIM()
                ssim_1_ = SSIM(regis_permute_print, batch_reference)
                # ssim_1_ = SSIM(batch_fixed.permute(0,3,1,2), batch_reference)
                ssim_2_ = SSIM(regis_permute_print, batch_fixed.permute(0,3,1,2))
                if not np.isnan(ssim_1_.data.item()) and not np.isnan(ssim_2_.data.item()):
                    ssim_f_print.append(ssim_1_.data.item())
                    ssim_m_print.append(ssim_2_.data.item())
            
            time_train += (time.time()-time_gpu) 
        time_e = time.time() - time_s
        if args.label:
            print('dice (mov, fix): ({}, {})'.format(np.mean(train_mov), np.mean(train_fix)))
        else:
            print('ssim (mov, fix): ({}, {})'.format(np.mean(ssim_f_print), np.mean(ssim_m_print)))
       
        # # visualize images to test
        # test_path = './img'
        # if not os.path.exists(test_path):
        #     os.makedirs(test_path) 

        # mov = ((np.array(batch_moving[0].cpu().data.numpy().squeeze())*0.5 + 0.5)*255).astype(np.uint8)
        # mov = np.asarray(mov) 
        # cv2.imwrite(os.path.join(test_path, 'img_{}_mov.png'.format(epoch+1)), mov)

        # fix = ((np.array(batch_fixed[0].cpu().data.numpy().squeeze())*0.5 + 0.5)*255).astype(np.uint8)
        # fix = np.asarray(fix) 
        # cv2.imwrite(os.path.join(test_path, 'img_{}_fix.png'.format(epoch+1)), fix)

        # regimg = ((np.array(img_grid[0].cpu().data.numpy().squeeze())*0.5 + 0.5)*255).astype(np.uint8)
        # regimg = np.asarray(regimg) 
        # cv2.imwrite(os.path.join(test_path, 'img_{}_reg.png'.format(epoch+1)), regimg)

        if not scheduler == None:
            scheduler.step()

        writer_train.add_scalar("G loss", torch.mean(torch.FloatTensor(loss_epoch)), epoch+1)
        if args.label:
            writer_train.add_scalar("G dice", np.mean(train_mov), epoch+1)
        print('[%d/%d] - Loss_Generator_Total: %.3f, Loss_pixel: %.3f, Loss_smooth: %.3f\nTime_total: %.3fs, Time_train: %.3fs' % ((epoch + 1), epochs,
                                    torch.mean(torch.FloatTensor(loss_epoch)),
                                    torch.mean(torch.FloatTensor(pixel_loss)),
                                    torch.mean(torch.FloatTensor(smooth_loss)),
                                    time_e,
                                    time_train))
        
        # store training example images
        if epoch+1 == epochs or (epoch+1) %10 == 0:  
            vfd = batch_fixed[-5:, :, :, :]
            vmd = batch_moving[-5:, :, :, :]
            registered_image = img_grid[-5:, :, :, :]
            util.store_img(vfd, registered_image, vmd, dir_test)
            torch.save(G.state_dict(), os.path.join(dir_checkpoint, 'model_epoch{}.pth'.format(epoch+1)))
        
        if not args.test:
            score_val, score_gt = [], []
            ssim_f, ssim_m = [], []
            loss_sum_val = []
            G.eval()
            with torch.no_grad():
                for data in validation_generator:
                    if args.label:
                        val_image, val_label = data
                        val_fixed, val_reference, val_moving = val_image
                        fixed_id, moving_id, warped_id = val_label

                        fixed_id = fixed_id.cuda()
                        moving_id = moving_id.cuda()
                        warped_id = warped_id.cuda()
                    else:
                        val_fixed, val_reference, val_moving = data

                    val_fixed = val_fixed.cuda()
                    val_reference = val_reference.cuda()
                    val_moving = val_moving.cuda()
                    

                    val_fixed = val_fixed.permute(0, 2, 3, 1)
                    val_moving = val_moving.permute(0, 2, 3, 1)
                    val_dvf, _, _, _, _ = G(val_fixed, val_moving)
                    # val_dvf = G(val_fixed, val_moving)
                    val_registered = util.img_warp_grid(val_moving.permute(0,3,1,2), val_dvf).permute(0, 2, 3, 1)
                
                    loss_smooth_val = 2*loss.smoothloss(val_dvf)
                    loss_pixel_val = 2*criterion(val_registered, val_fixed)
                    G_eval_loss_total = loss_smooth_val + loss_pixel_val 
    
                    loss_sum_val.append(G_eval_loss_total.data.item())
                    if args.label:
                        val_score, gt_score = util.store_dice_stn(fixed_id, moving_id, warped_id, val_dvf)         
                        score_gt.append(gt_score)
                        score_val.append(val_score)
                    else:
                        regis_permute = val_registered.permute(0,3,1,2)
                        SSIM = loss.SSIM()
                        ssim_1 = SSIM(regis_permute, val_reference)
                        ssim_2 = SSIM(regis_permute, val_fixed.permute(0,3,1,2))
                        if not np.isnan(ssim_1.data.item()) and not np.isnan(ssim_2.data.item()):
                            ssim_f.append(ssim_1.data.item())
                            ssim_m.append(ssim_2.data.item())
                
            writer_val.add_scalar("G loss", torch.mean(torch.FloatTensor(loss_sum_val)), epoch+1)
            if args.label:
                writer_val.add_scalar("G dice", np.mean(score_val), epoch+1)
                writer_gt.add_scalar("G dice", np.mean(score_gt), epoch+1)
            else:
                writer_val.add_scalar("G ssim", np.mean(ssim_f), epoch+1)
                writer_gt.add_scalar("G ssim", np.mean(ssim_m), epoch+1)

            # store model parameters
            if args.label:
                if  np.mean(score_gt)>thre_test:
                    thre_test  = np.maximum(np.mean(score_gt), thre_test)
                    best_model = G.state_dict()
                    
                    # store validation images
                    util.store_img(val_fixed, val_registered, val_moving, dir_validation)
                    
                    torch.save(best_model, os.path.join(dir_checkpoint, 'epoch'+str(epoch+1)+'_fullmodel.pth'))

                if np.mean(score_gt)>=args.ending:
                    break  
            else:
                if  np.mean(ssim_m)>thre_test:
                    thre_test  = np.maximum(np.mean(ssim_m), thre_test)
                    best_model = G.state_dict()
                    
                    # store validation images
                    util.store_img(val_fixed, val_registered, val_moving, dir_validation)
                    
                    torch.save(best_model, os.path.join(dir_checkpoint, 'epoch'+str(epoch+1)+'_fullmodel.pth'))

                if np.mean(ssim_m)>=args.ending:
                    break       
    torch.save(best_model, os.path.join(dir_checkpoint, 'best_fullmodel.pth'))

    # train and validate ending
    ###################################################################


    ###################################################################
    # test 
    score_val_sum = []
    index = []    
    score_val, score_gt = [], []
    ncc_f, ncc_m = [], []
    ssim_f, ssim_m = [], []

    G.load_state_dict(best_model)
    print('best model loaded...')
    G.eval()
    with torch.no_grad():
        for data in training_generator:
            if args.label:
                image, label = data
                test_fixed, test_reference, test_moving = image
                fixed_id, moving_id, warped_id = label

                fixed_id = fixed_id.cuda()
                moving_id = moving_id.cuda()
                warped_id = warped_id.cuda()
            else:
                test_fixed, test_reference, test_moving = data
                
            test_fixed = test_fixed.cuda()
            test_reference = test_reference.cuda()
            test_moving = test_moving.cuda()
            
            test_fixed = test_fixed.permute(0, 2, 3, 1)
            test_moving = test_moving.permute(0, 2, 3, 1)
            test_dvf, _, _, _, _ = G(test_fixed, test_moving)  
            # test_dvf = G(test_fixed, test_moving)  
            val_registered = util.img_warp_grid(test_moving.permute(0,3,1,2), test_dvf).permute(0, 2, 3, 1)

            regis_permute = val_registered.permute(0,3,1,2)
            NCC = loss.NCC() 
            ncc_score1 = NCC(regis_permute, test_reference)
            ncc_score2 = NCC(regis_permute, test_fixed.permute(0,3,1,2))   
            ncc_f.append(ncc_score1.data.item())
            ncc_m.append(ncc_score2.data.item())

            SSIM = loss.SSIM()
            ssim_1 = SSIM(regis_permute, test_reference)
            ssim_2 = SSIM(regis_permute, test_fixed.permute(0,3,1,2))
            if not np.isnan(ssim_1.data.item()) and not np.isnan(ssim_2.data.item()):
                ssim_f.append(ssim_1.data.item())
                ssim_m.append(ssim_2.data.item())
            
            if args.label:
                val_score, gt_score = util.store_dice_stn(fixed_id, moving_id, warped_id, test_dvf) 
                score_val.append(val_score)
                score_gt.append(gt_score)
        
        # store test image examples
        util.store_img(test_fixed, val_registered, test_moving, dir_train)
            
    if args.label:
        score_val_sum.append((np.mean(score_val), np.mean(score_gt)))
    else:
        score_val_sum.append((0, 0))
    score_val_sum.append((np.mean(ncc_f), np.mean(ncc_m)))
    score_val_sum.append((np.mean(ssim_f), np.mean(ssim_m)))

    index.append('dice')
    index.append('ncc')
    index.append('ssim')

    # store dice scores     
    pd_train = pd.DataFrame(score_val_sum, index=index, columns=['Test', 'GT'])
    pd_train.plot(kind='bar', ylim=[0,1], rot=0)
    plt.legend(loc='best')
    plt.title("Dice scores")
    plt.savefig(org_path+'/dice_bar.png')
    plt.clf()
    score_round = np.round(score_val_sum, 5)
    table = plt.table(cellText = score_round,
    cellLoc = 'center',
    cellColours = None,
    rowLabels = index,
    rowColours = plt.cm.Blues(np.linspace(0, 0.5,5))[::-1], 
    colLabels = ['Reg-Mov', 'Reg-Fix'],
    colColours = plt.cm.PuBu(np.linspace(0, 0.5,5))[::-1], 
    rowLoc='center',
    loc='center')
    table.scale(0.7, 2)
    table.set_fontsize(13)
    plt.axis('off')
    plt.savefig(org_path+'/dice_table.png')

    # test ending
    ######################################################################




    # f.close()
    writer_train.close()
    writer_val.close()
    writer_gt.close()
    print('All jobs finished. \nResults stored in "{}".'.format(os.path.abspath(org_path)))
    sys.exit(0)
