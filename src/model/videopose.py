# Based on https://github.com/facebookresearch/VideoPose3D
#
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn


class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """

    def __init__(self, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels, layernorm):
        super().__init__()

        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'

        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths
        self.layernorm = layernorm
        self.channels = channels

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.pad = [filter_widths[0] // 2]  # list of padding sizes
        self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)

    def set_bn_momentum(self, momentum):
        # if not self.layernorm:
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def create_norm_layer(self, frame_num):
        """ frame_num is the spatial dimension """
        if self.layernorm:
            # return nn.LayerNorm([self.channels, frame_num], elementwise_affine=False)
            return nn.InstanceNorm1d(self.channels, momentum=0.1, affine=True)
        else:
            return nn.BatchNorm1d(self.channels, momentum=0.1)

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def forward(self, x):
        assert len(x.shape) == 3, x.shape
        assert x.shape[-1] == self.in_features

        # sz = x.shape
        # x = x.view(x.shape[0], x.shape[1], -1)  # (nBatch,nFrames,nJoints*2) - unroll a single pose
        x = x.permute(0, 2, 1)  # (nBatch, nFeatures, nFrames)

        x = self._forward_blocks(x)

        x = x.permute(0, 2, 1)  # (nBatch, nFrames, nFeatures)
        # x = x.view(sz[0], -1, self.num_joints_out, 3)

        return x


class TemporalModel(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False, layernorm=False):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(in_features, num_joints_out, filter_widths, causal, dropout, channels, layernorm)

        self.expand_conv = nn.Conv1d(in_features, channels, filter_widths[0], bias=False)
        conv_num_frames = 1  # spatial dimension of the output of the conv layer; works only for [3,3,3,...shaped]
        for f in filter_widths:
            conv_num_frames *= f
        conv_num_frames = conv_num_frames - (filter_widths[0]-1)
        self.expand_bn = self.create_norm_layer(conv_num_frames)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]  # nonzero only for causal model
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation) if causal else 0)
            conv_num_frames = conv_num_frames - (filter_widths[i] - 1) * next_dilation

            layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2 * self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            layers_bn.append(self.create_norm_layer(conv_num_frames))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(self.create_norm_layer(conv_num_frames))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            res = x[:, :, pad + shift: x.shape[2] - pad + shift]

            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))

        x = self.shrink(x)
        return x


class TemporalModelOptimized1f(TemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.

    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """

    def __init__(self, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, layernorm=False):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__(in_features, num_joints_out, filter_widths, causal, dropout, channels, layernorm)

        self.expand_conv = nn.Conv1d(in_features, channels, filter_widths[0], stride=filter_widths[0], bias=False)
        conv_num_frames = 1  # spatial dimesnsion of the output of the conv layer; works only for [3,3,3,...shaped]
        for f in filter_widths[1:]:
            conv_num_frames *= f
        self.expand_bn = self.create_norm_layer(conv_num_frames)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0] // 2) if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2) if causal else 0)
            conv_num_frames = conv_num_frames // filter_widths[i]

            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn.append(self.create_norm_layer(conv_num_frames))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(self.create_norm_layer(conv_num_frames))
            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i + 1] + self.filter_widths[i + 1] // 2:: self.filter_widths[i + 1]]

            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))

        x = self.shrink(x)
        return x
