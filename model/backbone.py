from torch import nn
from torch.nn import functional as F
import functools
import torch
from torch.autograd import Variable
from model.self_attention import SelfAttention

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class Block(nn.Module):

    def __init__(self, inplanes, planes, downsample):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        return out


class ResNet12(nn.Module):

    def __init__(self, opt, channels):
        super().__init__()
        self.weight = opt.loss_weight
        self.inplanes = 3

        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])
        # self.self_attention = SelfAttention(8, 512, 512, 0., 0.)
        self.out_dim = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = Block(self.inplanes, planes, downsample)
        self.inplanes = planes
        return block

    def get_grads(self):
        grads = []
        for p in self.layer1.parameters():
            grads.append(p.grad.data.clone().flatten())
        for p in self.layer2.parameters():
            grads.append(p.grad.data.clone().flatten())
        for p in self.layer3.parameters():
            grads.append(p.grad.data.clone().flatten())
        for p in self.layer4.parameters():
            grads.append(p.grad.data.clone().flatten())
        return torch.cat(grads)

    def proj_grad(self, domain_grads):
        new_grads = []
        start = 0
        for k, p in enumerate(self.layer1.parameters()):
            dims = p.shape
            end = start + dims.numel()
            classigy_grads = domain_grads[0][start:end]
            da_grads = domain_grads[1][start:end]
            inner_product = torch.sum(classigy_grads*da_grads)
            proj_direction = inner_product / (torch.sum(
                da_grads*da_grads)+1e-12)
            da_grads = da_grads - torch.min(
                proj_direction, torch.zeros_like(proj_direction)) * da_grads         
            new_grads.append(self.weight*classigy_grads + (1-self.weight)*da_grads)
            start = end
        for k, p in enumerate(self.layer2.parameters()):
            dims = p.shape
            end = start + dims.numel()
            classigy_grads = domain_grads[0][start:end]
            da_grads = domain_grads[1][start:end]
            inner_product = torch.sum(classigy_grads*da_grads)
            proj_direction = inner_product / (torch.sum(
                da_grads*da_grads)+1e-12)
            da_grads = da_grads - torch.min(
                proj_direction, torch.zeros_like(proj_direction)) * da_grads
            new_grads.append(self.weight*classigy_grads + (1-self.weight)*da_grads)
            start = end
        for k, p in enumerate(self.layer3.parameters()):
            dims = p.shape
            end = start + dims.numel()
            classigy_grads = domain_grads[0][start:end]
            da_grads = domain_grads[1][start:end]
            inner_product = torch.sum(classigy_grads*da_grads)
            proj_direction = inner_product / (torch.sum(
                da_grads*da_grads)+1e-12)
            da_grads = da_grads - torch.min(
                proj_direction, torch.zeros_like(proj_direction)) * da_grads  
            new_grads.append(self.weight*classigy_grads + (1-self.weight)*da_grads)
            start = end
        for k, p in enumerate(self.layer4.parameters()):
            dims = p.shape
            end = start + dims.numel()
            classigy_grads = domain_grads[0][start:end]
            da_grads = domain_grads[1][start:end]
            inner_product = torch.sum(classigy_grads*da_grads)
            proj_direction = inner_product / (torch.sum(
                da_grads*da_grads)+1e-12)
            da_grads = da_grads - torch.min(
                proj_direction, torch.zeros_like(proj_direction)) * da_grads    
            new_grads.append(self.weight*classigy_grads + (1-self.weight)*da_grads)
            start = end
        return torch.cat(new_grads)

    def set_grads(self, new_grads):
        start = 0
        for k, p in enumerate(self.layer1.parameters()):
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end
        for k, p in enumerate(self.layer2.parameters()):
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end
        for k, p in enumerate(self.layer3.parameters()):
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end
        for k, p in enumerate(self.layer4.parameters()):
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end
        # for k, p in enumerate(self.self_attention.parameters()):
        #     dims = p.shape
        #     end = start + dims.numel()
        #     p.grad.data = new_grads[start:end].reshape(dims)
        #     start = end

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.out_dim)
        # x = x.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        return x # self.self_attention(x).view(x.size(0), -1) #mean(x, dim=1)


def conv_block_relu(in_channels, out_channels, downsample=True):
    if downsample: return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


def conv_block_leakyrelu(in_channels, out_channels, downsample=True):
    if downsample: return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
        nn.MaxPool2d(2)
    )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )


class Conv64F(nn.Module):
    def __init__(self, leakyrelu=True):
        super(Conv64F, self).__init__()
        self.hid_dim = self.z_dim = 64
        conv_block = conv_block_leakyrelu if leakyrelu else conv_block_relu
        self.features = nn.Sequential(
            conv_block(3, self.hid_dim),
            conv_block(self.hid_dim, self.hid_dim),
            conv_block(self.hid_dim, self.hid_dim),
            conv_block(self.hid_dim, self.z_dim),
        )
        self.self_attention = SelfAttention(8, 64, 64, 0., 0.)

    def forward(self, input1,):
        # extract features of input1--query image
        x = self.features(input1).permute(0, 2, 3, 1).contiguous().view(input1.size(0), -1, self.z_dim)
        return self.self_attention(x).view(x.size(0), -1)


class Conv512F(nn.Module):
    def __init__(self, leakyrelu=True):
        super(Conv512F, self).__init__()
        conv_block = conv_block_leakyrelu if leakyrelu else conv_block_relu
        self.features = nn.Sequential(
            conv_block(3, 96),
            conv_block(96, 192),
            conv_block(192, 384),
            conv_block(384, 512),
        )

    def forward(self, input1,):
        # extract features of input1--query image
        x = self.features(input1)
        # x = x.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        return x
