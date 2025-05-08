import torch
import torch.nn as nn
from torchinfo import summary

torch.set_float32_matmul_precision('high')


class ConvNNBuilder:
    def __init__(self, input_shape):
        """
        Args:
            input_shape: (channels, height, width)
        """
        self.layers: list[nn.Module] = []
        self._built = False
        C, H, W = input_shape
        self.prev_channels = C
        self.H, self.W = H, W

    def _check_not_built(self):
        if self._built:
            raise RuntimeError("Cannot modify builder after output layer is added.")

    def _add_conv(
        self,
        out_channels,
        kernel_size,
        stride,
        padding,
        activation
    ):

        self._check_not_built()
        conv = nn.Conv2d(self.prev_channels, out_channels, kernel_size, stride, padding)
        self.layers.append(conv)

        k_h, k_w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        s_h, s_w = stride if isinstance(stride, tuple) else (stride, stride)
        p_h, p_w = padding if isinstance(padding, tuple) else (padding, padding)
        self.H = (self.H + 2*p_h - k_h)//s_h + 1
        self.W = (self.W + 2*p_w - k_w)//s_w + 1

        self.layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            self.layers.append(activation())
        self.prev_channels = out_channels
        return self

    def _add_pool(
        self,
        kernel_size,
        stride,
        pool_type
    ):

        self._check_not_built()
        if pool_type == 'max':
            pool = nn.MaxPool2d(kernel_size, stride)
        else:
            pool = nn.AvgPool2d(kernel_size, stride)
        self.layers.append(pool)

        k_h, k_w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        s_h, s_w = stride if isinstance(stride, tuple) else (stride or kernel_size, stride or kernel_size)
        self.H = (self.H - k_h)//s_h + 1
        self.W = (self.W - k_w)//s_w + 1
        return self

    def _add_dropout2d(self, p):
        self._check_not_built()
        self.layers.append(nn.Dropout2d(p))
        return self

    def _add_flatten(self):
        self._check_not_built()
        self.layers.append(nn.Flatten())

        self._flat_dim = self.prev_channels * self.H * self.W
        return self

    def _add_dense(
        self,
        hidden_dim,
        activation
    ):

        self._check_not_built()
        if not hasattr(self, '_flat_dim'):
            raise RuntimeError("Must call add_flatten() before dense layers.")
        self.layers.append(nn.Linear(self._flat_dim, hidden_dim))
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(activation())

        self._flat_dim = hidden_dim
        return self

    def _add_dropout(self, p):
        self._check_not_built()
        self.layers.append(nn.Dropout(p))
        return self

    def _add_output_layer(
        self,
        output_dim,
        activation
    ):

        self._check_not_built()
        if not hasattr(self, '_flat_dim'):
            raise RuntimeError("Must call _add_flatten() and optionally dense layers before output.")
        self.layers.append(nn.Linear(self._flat_dim, output_dim))
        if activation:
            self.layers.append(activation())
        model = nn.Sequential(*self.layers)
        self._built = True
        return model


def invoke(input_dim):
    builder = ConvNNBuilder(input_dim)
    return builder


meta = {
    "name": "CNN Input Layer",
    "min_args": 1,
    "max_args": 1
}
