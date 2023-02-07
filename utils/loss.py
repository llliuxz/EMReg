import torch
import torch.nn.functional as F
import numpy as np
import math
from math import exp

use_gpu = torch.cuda.is_available()

def dice_score(label1, label2):
    batch_size = label1.size(0)
    label = (label1==label2).int().view(batch_size, -1)
    score = []
    for i in range(batch_size):
        a = label[i,:].sum(0)
        b = label[i,:].size(0)
        score.append((a/b).cpu().numpy())
    return np.mean(score)

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def __call__(self, y_true, y_pred):
        Ii = torch.round((y_true*0.5 + 0.5)*255)
        Ji = torch.round((y_pred*0.5 + 0.5)*255)

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [128] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return torch.mean(cc)

def smoothloss(y_pred):
    dy = torch.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
    dx = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    # dxy = torch.abs(y_pred[:, 1:, 1:, :] - y_pred[:, :-1, :-1, :])

    dx = torch.mul(dx, dx)
    dy = torch.mul(dy, dy)
    # dxy = torch.mul(dxy, dxy)
    d = torch.mean(dx) + torch.mean(dy)
    return d/2.0
 
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
 
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
 
def ssim(img1, img2, window_size=11, window=None, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
 
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)  # u(X)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)  # u(y)
 
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq   
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
 
    # C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    v2 = sigma1_sq + sigma2_sq + C2
  
    # lumi = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contr = (2 * (sigma1_sq**0.5) * (sigma2_sq**0.5) + C2) / v2
    si = (sigma12 + C2) / ((sigma1_sq**0.5) * (sigma2_sq**0.5) + C2)
    
    res2, res3 = contr.mean(), si.mean()**0.5

    return res2*res3
 
class SSIM(torch.nn.Module):
    def __init__(self, window_size=64):
        super(SSIM, self).__init__()
        self.window_size = window_size
        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)
 
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
 
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
 
        return ssim(img1, img2, window=window.to(img1.device), window_size=self.window_size)