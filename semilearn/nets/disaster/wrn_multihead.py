# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn
import torch.nn.functional as F

from semilearn.nets.wrn.wrn import BasicBlock, NetworkBlock

class WideResNetMultihead(nn.Module):
    def __init__(self, first_stride, num_classes, num_heads=3, depth=28, widen_factor=2, drop_rate=0.0, **kwargs):
        super(WideResNetMultihead, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock

        # number of heads used in multihead cotraining
        self.num_heads = num_heads

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = self.multihead_constructor(lambda: NetworkBlock(n, channels[2], channels[3], block, 2, drop_rate))

        # global average pooling and classifier
        self.bn1 = self.multihead_constructor(lambda: nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001))
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.classifier = self.multihead_constructor(lambda: nn.Linear(channels[3], num_classes))
        self.channels = channels[3]
        self.num_features = channels[3]

        # rot_classifier for Remix Match
        # self.is_remix = is_remix
        # if is_remix:
        #     self.rot_classifier = nn.Linear(self.channels, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.extract(x)
        for i in range(len(out)):
            out[i] = F.adaptive_avg_pool2d(out[i], 1)
            out[i] = out[i].view(-1, self.channels)

        output = [head_classifier(head_out) for head_classifier, head_out in zip(self.classifier, out)]

        result_dict = {'logits': output, 'feat': out}

        return result_dict

    def extract(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        return [self.relu(head_bn1(head_block3(out))) for head_block3, head_bn1 in zip(self.block3, self.bn1)]

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem=r'^{}conv1'.format(prefix), blocks=r'^{}block(\d+)'.format(prefix) if coarse else r'^{}block(\d+)\.layer.(\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        nwd = []
        for n, _ in self.named_parameters():
            if 'bn' in n or 'bias' in n:
                nwd.append(n)
        return nwd
    
    def multihead_constructor(self, constructor):
        return [constructor() for _ in range(self.num_heads)]
    
    # @overrides
    def modules(self):
        from itertools import chain
        return chain(super().modules(), self.block3, self.bn1, self.classifier)


def wrn_multihead(num_classes, num_heads, depth, widen_factor):
    return WideResNetMultihead(first_stride=1, num_classes=num_classes, num_heads=num_heads, depth=depth, widen_factor=widen_factor)

def wrn_multihead_28_2(args):
    return wrn_multihead(args.num_classes, args.num_heads, 28, 2)

def wrn_multihead_28_8(args):
    return wrn_multihead(args.num_classes, args.num_heads, 28, 8)
