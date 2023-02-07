import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class UnitGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(UnitGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepUnitGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, motion_dim=128):
        super(SepUnitGRU, self).__init__()
        self.convz1 = nn.Conv2d(motion_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(motion_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(motion_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(motion_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(motion_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(motion_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q           # [N, dim(feature)/2, H, W]

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q           # [N, dim(feature)/2, H, W]

        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicMotionEncoder(nn.Module):
    def __init__(self, inchannel):
        super(BasicMotionEncoder, self).__init__()
        # layers for correlation
        self.convc1 = nn.Conv2d(inchannel, 256, 1, padding=0)          # outsize = insize
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)                 # outsize = insize

        # layers for flow
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)           # outsize = insize
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)          # outsize = insize

        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)      # outsize = insize

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))      # dimension 192

        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))      # dimension 64

        cor_flo = torch.cat([cor, flo], dim=1)  # dimension 192+64
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = UnitGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow

class BasicUpdateBlock(nn.Module):
    def __init__(self, inchannel, feature_dim, up):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(inchannel)
        self.gru = SepUnitGRU(hidden_dim=feature_dim//2, input_dim=feature_dim)
        self.flow_head = FlowHead(feature_dim//2, hidden_dim=feature_dim)

        self.mask = nn.Sequential(
            nn.Conv2d(feature_dim//2, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, up*up*9, 1, padding=0))

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)      # [N, 128, H, W]
        inp = torch.cat([inp, motion_features], dim=1)  # 128 + dim(feature)/2

        # inp: 128 + dim(feature)/2;   net: dim(feature)/2
        net = self.gru(net, inp)        # [N, dim(feature)/2, H, W]
        delta_flow = self.flow_head(net)        # [N, 2, H, W]

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow



