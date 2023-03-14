"""
95% of this file is shamelessly stolen from https://github.com/openai/DALL-E/tree/master/dall_e
"""


import attr
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections  import OrderedDict
from functools    import partial


def conv_layer(n_in, n_out, k, device=torch.device('cpu'), requires_grad=True):
    assert n_in >= 1 and n_out >= 1
    assert k >= 1 and k % 2 == 1
    pad = (k - 1) // 2
    conv = nn.Conv2d(n_in, n_out, k, padding=pad, device=device)
    conv.requires_grad_(requires_grad)
    return conv


@attr.s(eq=False, repr=False)
class EncoderBlock(nn.Module):
    n_in:     int = attr.ib(validator=lambda i, a, x: x >= 1)
    n_out:    int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 4 ==0)
    n_layers: int = attr.ib(validator=lambda i, a, x: x >= 1)

    device:        torch.device = attr.ib(default=None)
    requires_grad: bool         = attr.ib(default=False)

    def __attrs_post_init__(self) -> None:
        super().__init__()
        self.n_hid = self.n_out // 4
        self.post_gain = 1 / (self.n_layers ** 2)

        make_conv     = partial(conv_layer, device=self.device, requires_grad=self.requires_grad)
        self.id_path  = make_conv(self.n_in, self.n_out, 1) if self.n_in != self.n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([
                ('relu_1', nn.ReLU()),
                ('conv_1', make_conv(self.n_in,  self.n_hid, 3)),
                ('relu_2', nn.ReLU()),
                ('conv_2', make_conv(self.n_hid, self.n_hid, 3)),
                ('relu_3', nn.ReLU()),
                ('conv_3', make_conv(self.n_hid, self.n_hid, 3)),
                ('relu_4', nn.ReLU()),
                ('conv_4', make_conv(self.n_hid, self.n_out, 1)),]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)


@attr.s(eq=False, repr=False)
class Encoder(nn.Module):
    group_count:     int = 4
    n_hid:           int = 256
    n_blk_per_group: int = 2
    input_channels:  int = 3
    output_channels: int = 10

    device:          torch.device = torch.device('cpu')
    requires_grad:   bool         = False

    def __attrs_post_init__(self) -> None:
        super().__init__()

        blk_range  = range(self.n_blk_per_group)
        n_layers   = self.group_count * self.n_blk_per_group
        make_conv  = partial(conv_layer, device=self.device, requires_grad=self.requires_grad)
        make_blk   = partial(EncoderBlock, n_layers=n_layers, device=self.device,
                requires_grad=self.requires_grad)

        self.blocks = nn.Sequential(OrderedDict([
            ('input', make_conv(self.input_channels, 1 * self.n_hid, 7)),
            ('group_1', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(1 * self.n_hid, 1 * self.n_hid)) for i in blk_range],
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ]))),
            ('group_2', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(1 * self.n_hid if i == 0 else 2 * self.n_hid, 2 * self.n_hid)) for i in blk_range],
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ]))),
            ('group_3', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(2 * self.n_hid if i == 0 else 4 * self.n_hid, 4 * self.n_hid)) for i in blk_range],
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ]))),
            ('group_4', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(4 * self.n_hid if i == 0 else 8 * self.n_hid, 8 * self.n_hid)) for i in blk_range],
            ]))),
            ('output', nn.Sequential(OrderedDict([
                ('relu', nn.ReLU()),
                ('conv', make_conv(8 * self.n_hid, self.output_channels, 1)),
            ]))),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 4:
            raise ValueError(f'input shape {x.shape} is not 4d')
        if x.shape[1] != self.input_channels:
            raise ValueError(f'input has {x.shape[1]} channels but model built for {self.input_channels}')
        if x.dtype != torch.float32:
            raise ValueError('input must have dtype torch.float32')

        return self.blocks(x)


@attr.s(eq=False, repr=False)
class DecoderBlock(nn.Module):
    n_in:     int = attr.ib(validator=lambda i, a, x: x >= 1)
    n_out:    int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 4 ==0)
    n_layers: int = attr.ib(validator=lambda i, a, x: x >= 1)

    device:        torch.device = attr.ib(default=None)
    requires_grad: bool         = attr.ib(default=False)

    def __attrs_post_init__(self) -> None:
        super().__init__()
        self.n_hid = self.n_out // 4
        self.post_gain = 1 / (self.n_layers ** 2)

        make_conv     = partial(conv_layer, device=self.device, requires_grad=self.requires_grad)
        self.id_path  = make_conv(self.n_in, self.n_out, 1) if self.n_in != self.n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([
                ('relu_1', nn.ReLU()),
                ('conv_1', make_conv(self.n_in,  self.n_hid, 1)),
                ('relu_2', nn.ReLU()),
                ('conv_2', make_conv(self.n_hid, self.n_hid, 3)),
                ('relu_3', nn.ReLU()),
                ('conv_3', make_conv(self.n_hid, self.n_hid, 3)),
                ('relu_4', nn.ReLU()),
                ('conv_4', make_conv(self.n_hid, self.n_out, 3)),]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)



@attr.s(eq=False, repr=False)
class Decoder(nn.Module):
    group_count:     int = 4
    n_init:          int = 128
    n_hid:           int = 256
    n_blk_per_group: int = 2
    output_channels: int = 3
    input_channels:  int = 10

    device:              torch.device = torch.device('cpu')
    requires_grad:       bool         = False
    use_mixed_precision: bool         = True

    def __attrs_post_init__(self) -> None:
        super().__init__()

        blk_range  = range(self.n_blk_per_group)
        n_layers   = self.group_count * self.n_blk_per_group
        make_conv  = partial(conv_layer, device=self.device, requires_grad=self.requires_grad)
        make_blk   = partial(DecoderBlock, n_layers=n_layers, device=self.device,
                requires_grad=self.requires_grad)

        self.blocks = nn.Sequential(OrderedDict([
            ('input', make_conv(self.input_channels, self.n_init, 1)),
            ('group_1', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(self.n_init if i == 0 else 8 * self.n_hid, 8 * self.n_hid)) for i in blk_range],
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            ]))),
            ('group_2', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(8 * self.n_hid if i == 0 else 4 * self.n_hid, 4 * self.n_hid)) for i in blk_range],
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            ]))),
            ('group_3', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(4 * self.n_hid if i == 0 else 2 * self.n_hid, 2 * self.n_hid)) for i in blk_range],
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            ]))),
            ('group_4', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(2 * self.n_hid if i == 0 else 1 * self.n_hid, 1 * self.n_hid)) for i in blk_range],
            ]))),
            ('output', nn.Sequential(OrderedDict([
                ('relu', nn.ReLU()),
                ('conv', make_conv(self.n_hid, self.output_channels, 1)),
            ]))),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 4:
            raise ValueError(f'input shape {x.shape} is not 4d')
        if x.shape[1] != self.input_channels:
            raise ValueError(f'input has {x.shape[1]} channels but model built for {self.input_channels}')
        if x.dtype != torch.float32:
            raise ValueError('input must have dtype torch.float32')

        return self.blocks(x)
