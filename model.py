import torch.nn as nn
import torch
from torchvision.models import resnet18
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import deform_conv2d
from torch.nn import functional as F
import math
from util import BezierCurve

"""
Base convolution layer module including option for dilated, batchnorm and ReLU
"""
class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, init_weight=False, bn=True, relu=True, kernel_size=3, stride=1, padding=0, bias=False, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn2d = nn.BatchNorm2d(out_channels)
        self.bn = bn
        if init_weight:
            self.init_weight()
        self.relu = relu


    def forward(self, x):
        out = self.conv(x)
        if self.bn:
            out = self.bn2d(out)
        if self.relu:
            out = F.relu(out, inplace=True)
        
        return out
    
    def init_weight(self):
        nn.init.kaiming_normal_(self.conv.weight, a=1)
        if not self.conv.bias is None: 
            nn.init.constant_(self.conv.bias, 0)


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn = True, relu=True, kernel_size=3, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn1d = nn.BatchNorm1d(out_channels)
        self.bn = bn
        self.relu = relu


    def forward(self, x):
        out = self.conv(x)
        if self.bn:
            out = self.bn1d(out)
        if self.relu:
            out = F.relu(out, inplace=True)
        
        return out


"""
Feature Flip Fusion block
"""
class FeatureFlipFusion(nn.Module):
    def __init__(self, channels, kernel_size=(3, 3), groups=1, deform_groups=1) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = Conv2DBlock(channels, channels, relu=False, kernel_size=1, padding=0)
        self.norm = nn.BatchNorm2d(channels)
        self.conv_offset = nn.Conv2d(
            channels * 2,
            deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            padding=1,
            bias=True)

        self.weight = nn.Parameter(torch.Tensor(channels, channels // groups, *kernel_size))
        self.bias = nn.Parameter(torch.Tensor(channels))

        self.init_weights()

    def init_weights(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        n = self.channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()


    def forward(self, x):
        flip = x.flip(-1) # 256 * 23 * 40

        x = self.conv(x) # 256 * 23 * 40

        # deformable
        concat = torch.cat([flip, x], dim=1) # 512 * 23 * 40
        out = self.conv_offset(concat) # 27 * 21 * 38
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1) # 18 * 21 * 38
        mask = torch.sigmoid(mask) # 9 * 21 * 38
        flip = deform_conv2d(flip, offset, self.weight, self.bias, mask=mask) # 256 * 21 * 38

        return F.relu(self.norm(flip) + x)


"""
Dilated block 
"""
class DilatedBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, dilation=[4, 8]) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            Conv2DBlock(in_channels, hidden_channels, bn=True, relu=True, kernel_size=1, padding=0),
            Conv2DBlock(hidden_channels, hidden_channels, bn=True, relu=True, kernel_size=3, dilation=dilation[0], padding=dilation[0]),
            Conv2DBlock(hidden_channels, in_channels, bn=True, relu=True, kernel_size=1, padding=0)
        )   
        self.block2 = nn.Sequential(
            Conv2DBlock(in_channels, hidden_channels, bn=True, relu=True, kernel_size=1, padding=0),
            Conv2DBlock(hidden_channels, hidden_channels, bn=True, relu=True, kernel_size=3, dilation=dilation[1], padding=dilation[1]),
            Conv2DBlock(hidden_channels, in_channels, bn=True, relu=True, kernel_size=1, padding=0)
        )   

    def forward(self, x):
        
        identity = x
        out = identity + self.block1(x)
        identity = out
        out = identity + self.block1(out)
        return out 

def lane_pruning(existence, existence_conf, max_lane):
    # Prune lanes based on confidence (a max number constrain for lanes in an image)
    # Maybe too slow (but should be faster than topk/sort),
    # consider batch size >> max number of lanes
    while (existence.sum(dim=1) > max_lane).sum() > 0:
        indices = (existence.sum(dim=1, keepdim=True) > max_lane).expand_as(existence) * \
                  (existence_conf == existence_conf.min(dim=1, keepdim=True).values)
        existence[indices] = 0
        existence_conf[indices] = 1.1  # So we can keep using min

    return existence, existence_conf

    
class BenizerNet(nn.Module):
    def __init__(self, image_height=360, global_stride=16, regression_target=8, final_channels=256, hidden_channels=64, thresh=0.5, local_maximum_window_size=9):
        super().__init__()
        self.thresh = thresh
        self.local_maximum_window_size = local_maximum_window_size

        # Pretrained Resnet
        self.resnet = resnet18(pretrained=True)
        self.resnet = IntermediateLayerGetter(self.resnet, {'layer3': 'feat2'})

        # Dilated RESA module
        self.dilate_block = DilatedBlock(final_channels, hidden_channels)

        # Segmentation head
        self.seg_head = nn.ModuleList()
        self.seg_head.append(Conv2DBlock(final_channels, hidden_channels, init_weight=True, padding=1))
        self.seg_head.append(Conv2DBlock(hidden_channels, 1, init_weight=True, bn=False, relu=False, padding=1))

        # Feature Flip Fusion
        self.fff = FeatureFlipFusion(final_channels)

        self.aggregator = nn.AvgPool2d(kernel_size=((image_height - 1) // global_stride + 1, 1), stride=1, padding=0)
        self.regression_head = nn.ModuleList()
        self.regression_head.append(Conv1DBlock(final_channels, final_channels, bn=True, relu=True, kernel_size=3, bias=True, padding=2))
        self.regression_head.append(Conv1DBlock(final_channels, final_channels, bn=True, relu=True, kernel_size=3, bias=True, padding=2))

        self.proj_classification = Conv1DBlock(final_channels, 1, bn=False, relu=False, kernel_size=1, bias=True)
        self.proj_regression = Conv1DBlock(final_channels, regression_target, bn=False, relu=False, kernel_size=1, bias=True)


    def forward(self, x):
        x = self.resnet(x)['feat2']
        segmentations = x
        for model in self.seg_head:
            segmentations = model(segmentations)

        x = self.dilate_block(x)
        x = self.fff(x)
        x = self.aggregator(x)[:, :, 0, :]

        for model in self.regression_head:
            x = model(x)

        logits = self.proj_classification(x).squeeze(1)
        curves = self.proj_regression(x)

        return [logits, curves, segmentations]

    # Run model inference to get x coordinates that match the h_samples from labeling
    def infer(self, input, max_lane=5, gap=10, points=56):
        pred = self.forward(input)
        pred_conf = pred[0].sigmoid() 
        pred_select = pred_conf > self.thresh

        _, max_indices = torch.nn.functional.max_pool1d(pred_conf.unsqueeze(1),
                                                            kernel_size=self.local_maximum_window_size, stride=1,
                                                            padding=(self.local_maximum_window_size - 1) // 2,
                                                            return_indices=True)
        max_indices = max_indices.squeeze(1)  # B x Q
        indices = torch.arange(0, pred_conf.shape[1], dtype=pred_conf.dtype).unsqueeze(0).expand_as(max_indices)
        local_maxima = max_indices == indices
        pred_select *= local_maxima

        control_points = pred[1]

        pred_select, _ = lane_pruning(pred_select, pred_conf, max_lane=max_lane)

        existence = existence.cpu().numpy()
        control_points = control_points.cpu().numpy()
        bezier = BezierCurve(order=3, num_sample_points=self.image_height)

        lane_coordinates = []
        for j in range(existence.shape[0]):
            lane_coordinates.append(bezier.bezier_to_coordinates(control_points=control_points[j], existence=existence[j],
                                                               resize_shape=(360, 640), gap=gap, ppl=points))

        return lane_coordinates