import torch
from torch import nn
 
class spatial_attention(nn.Module):
    def __init__(self, kernel_size):
        super(spatial_attention, self).__init__()
        
        padding = kernel_size // 2

        self.conv_f = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=kernel_size, padding=padding, bias=False), 
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))

        self.conv_m = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=kernel_size, padding=padding, bias=False), 
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))

        self.sigmoid = nn.Sigmoid()

    def pooling(self, input_fmap):
        x_maxpool, _ = torch.max(input_fmap, dim=1, keepdim=True)
        x_avgpool = torch.mean(input_fmap, dim=1, keepdim=True)
        x = torch.cat([x_maxpool, x_avgpool], dim=1)
        return x

    
    def forward(self, input_cross, input_f, input_m):
        pooling_cross = self.pooling(input_cross)
        pooling_f = self.pooling(input_f)
        pooling_m = self.pooling(input_m)

        cat_f = torch.cat([pooling_f, pooling_cross], dim=1)
        cat_m = torch.cat([pooling_m, pooling_cross], dim=1)

        weight_f = self.conv_f(cat_f)
        prob_f = self.sigmoid(weight_f)

        weight_m = self.conv_m(cat_m)
        prob_m = self.sigmoid(weight_m)
        
        return prob_f, prob_m
 
class cbam(nn.Module):
    def __init__(self, in_channel, ratio=4, kernel_size=5):
        super(cbam, self).__init__()
        self.inchannel = in_channel
        self.spatial_attention = spatial_attention(kernel_size=kernel_size)
        self.squeeze_dim = nn.Sequential(
                    nn.Conv2d(in_channels=in_channel*2, out_channels=in_channel, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=False)
                    )
    
    def forward(self, inputs):
        fix_fmaps, mov_fmaps = torch.split(inputs, [self.inchannel, self.inchannel], dim=1)
        cross_fmaps = self.squeeze_dim(inputs)

        prob_f, prob_m = self.spatial_attention(cross_fmaps, fix_fmaps, mov_fmaps)  
        weighted_fix = fix_fmaps*prob_f
        weighted_mov = mov_fmaps*prob_m     
        return weighted_fix, weighted_mov
