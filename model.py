import torch
import torch.nn as nn
from scipy.stats import truncnorm
import math

expansion = 1

def Conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=False)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=True)
        self.conv = nn.Sequential(conv, )
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=2, padding=1, output_padding=1, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class FastGuide(nn.Module):
    def __init__(self, input_planes, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.expansion_ratio = 3
        self.conv1 = Basic2d(input_planes, input_planes, None)      
        self.weight_expansion = Basic2d(input_planes, input_planes * self.expansion_ratio, norm_layer, kernel_size=1, padding=0)

        self.conv2 = Basic2d(input_planes, input_planes, norm_layer, kernel_size=1, padding=0)
        self.conv3 = Basic2d(input_planes, input_planes)

    def forward(self, input, weight):
        weight = self.conv1(weight)
        weight = self.weight_expansion(weight)

        kernels = torch.chunk(weight, self.expansion_ratio, 1)
        splits = []

        for i in range(self.expansion_ratio):
            splits.append(input*kernels[i])
        out = sum(splits)
        out = self.conv2(out)

        avg_out = torch.mean(weight, dim=1, keepdim=True)
        out = self.conv3(out * avg_out)
        
        return out


class BasicBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, act=True):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.act = act

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if self.act:
            out = self.relu(out)
        return out


class CHNet(nn.Module):
    def __init__(self, block=BasicBlock, bc=16, img_layers=[2, 2, 2, 2, 2],
                 depth_layers=[2, 2, 2, 2, 2], norm_layer=nn.BatchNorm2d):
        super().__init__()
        self._norm_layer = norm_layer

        self.conv_img = Basic2d(3, bc * 2, norm_layer=norm_layer, kernel_size=5, padding=2)
        in_channels = bc * 2
        self.inplanes = in_channels
        self.layer1_img = self._make_layer(block, in_channels * 2, img_layers[0], stride=2)

        self.guide1 = FastGuide(in_channels * 2, norm_layer)
        self.inplanes = in_channels * 2 * expansion
        self.layer2_img = self._make_layer(block, in_channels * 4, img_layers[1], stride=2)

        self.guide2 = FastGuide(in_channels * 4, norm_layer)
        self.inplanes = in_channels * 4 * expansion
        self.layer3_img = self._make_layer(block, in_channels * 8, img_layers[2], stride=2)

        self.guide3 = FastGuide(in_channels * 8, norm_layer)
        self.inplanes = in_channels * 8 * expansion
        self.layer4_img = self._make_layer(block, in_channels * 8, img_layers[3], stride=2)

        self.guide4 = FastGuide(in_channels * 8, norm_layer)

        self.conv_lidar = Basic2d(1, bc * 2, norm_layer=None, kernel_size=5, padding=2)

        self.inplanes = in_channels
        self.layer1_lidar = self._make_layer(block, in_channels * 2, depth_layers[0], stride=2)
        self.inplanes = in_channels * 2 * expansion
        self.layer2_lidar = self._make_layer(block, in_channels * 4, depth_layers[1], stride=2)
        self.inplanes = in_channels * 4 * expansion
        self.layer3_lidar = self._make_layer(block, in_channels * 8, depth_layers[2], stride=2)
        self.inplanes = in_channels * 8 * expansion
        self.layer4_lidar = self._make_layer(block, in_channels * 8, depth_layers[3], stride=2)

        self.layer1d = Basic2dTrans(in_channels * 2, in_channels, norm_layer) 
        self.layer2d = Basic2dTrans(in_channels * 4, in_channels * 2, norm_layer)
        self.layer3d = Basic2dTrans(in_channels * 8, in_channels * 4, norm_layer)
        self.layer4d = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)

        self.conv_ob = nn.Sequential(block(bc * 2, bc * 2, norm_layer=norm_layer, act=False),
                                       nn.Conv2d(bc * 2, 1, kernel_size=3, stride=1, padding=1))
        self.conv_unob = nn.Sequential(block(bc * 2, bc * 2, norm_layer=norm_layer, act=False),
                                       nn.Conv2d(bc * 2, 1, kernel_size=3, stride=1, padding=1))
        self.ref = block(bc * 2, bc * 2, norm_layer=norm_layer, act=False)

        self._initialize_weights()

    def forward(self, x):
        img = x['rgb']
        lidar = x['d']

        lidar_mask = (lidar > 0).detach()

        c0_img = self.conv_img(img)
        c0_lidar = self.conv_lidar(lidar)

        c1_img = self.layer1_img(c0_img)
        c1_lidar = self.layer1_lidar(c0_lidar)
        c1_lidar = self.guide1(c1_lidar, c1_img)

        c2_img = self.layer2_img(c1_img)
        c2_lidar = self.layer2_lidar(c1_lidar)
        c2_lidar = self.guide2(c2_lidar, c2_img)

        c3_img = self.layer3_img(c2_img)
        c3_lidar = self.layer3_lidar(c2_lidar)
        c3_lidar = self.guide3(c3_lidar, c3_img)

        c4_img = self.layer4_img(c3_img)
        c4_lidar = self.layer4_lidar(c3_lidar)
        c4_lidar = self.guide4(c4_lidar, c4_img)

        de2 = self.layer4d(c4_lidar)
        de2 = de2 + c3_lidar

        de3 = self.layer3d(de2)
        de3 = de3 + c2_lidar

        de4 = self.layer2d(de3)
        de4 = de4 + c1_lidar

        de5 = self.layer1d(de4)
        de5 = de5 + c0_lidar  

        output = self.ref(de5)

        output_ob = self.conv_ob(output)
        output_unob = self.conv_unob(output)

        output = lidar_mask * output_ob + ~lidar_mask * output_unob

        return output, output_ob, output_unob


    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * expansion, stride),
                norm_layer(planes * expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        def truncated_normal_(num, mean=0., std=1.):
            lower = -2 * std
            upper = 2 * std
            X = truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std)
            samples = X.rvs(num)
            output = torch.from_numpy(samples)
            return output

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                data = truncated_normal_(m.weight.nelement(), mean=0, std=math.sqrt(1.3 * 2. / n))
                data = data.type_as(m.weight.data)
                m.weight.data = data.view_as(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)