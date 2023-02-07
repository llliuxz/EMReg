from operator import truediv
import torch
import torch.nn.functional as F
from .correlation_cupy import correlation_cupy_acc

def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid

def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)],
                          )
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid

def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]

def global_correlation(feature0, feature1):
    # global correlation
    b, c, h, w = feature0.shape
    feature0 = feature0.view(b, c, -1)                      # [B, C, H*W]
    feature1 = feature1.view(b, c, -1)                      # [B, C, H*W]
    correlation = torch.matmul(feature0.transpose(1,2), feature1).reshape(b*h*w, 1, h, w)  / torch.sqrt(torch.tensor(c).float())     # [B, H, W, H, W]
    return correlation

def local_correlation(feature0, feature1, local_radius=4, padding_mode='zeros', cupy_acc=False):
    if cupy_acc:
        corr = correlation_cupy_acc.FunctionCorrelation(feature0, feature1)
        b, window, h, w = corr.size()
        corr = corr.permute(0,2,3,1).reshape(b*h*w, 1, -1)
        radius = int(window ** 0.5)
        return corr.reshape(b*h*w, 1, radius, -1)
    else:
        b, c, h, w = feature0.size()
        coords_init = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
        
        coords = coords_init.view(b, 2, -1).permute(0, 2, 1)    # [B, H*W, 2]

        local_h = 2 * local_radius + 1
        local_w = 2 * local_radius + 1

        window_grid = generate_window_grid(-local_radius, local_radius,
                                        -local_radius, local_radius,
                                        local_h, local_w, device=feature0.device)    # [2R+1, 2R+1, 2]
        window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)                     # [B, 1, (2R+1)^2, 2]
        sample_coords = coords.unsqueeze(-2) + window_grid                              # [B, H*W, (2R+1)^2, 2]

        # exclude coords that are out of image space
        valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)    # [B, H*W, (2R+1)^2]
        valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)    # [B, H*W, (2R+1)^2]
        valid = valid_x & valid_y        # [B, H*W, (2R+1)^2], used to mask out invalid values when softmax

        # normalize coordinates to [-1, 1]
        sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
        window_feature = F.grid_sample(feature1, sample_coords_norm,
                                    padding_mode=padding_mode, align_corners=True
                                    ).permute(0, 2, 1, 3)  # [B, H*W, C, (2R+1)^2]
        feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]
        corr = torch.matmul(feature0_view, window_feature).view(b,h*w,-1) / (c ** 0.5)  # [B, H*W, (2R+1)^2]

        # mask invalid locations
        corr[~valid] = 0
        corr = corr.reshape(b*h*w, 1, local_h, local_w)

        return corr