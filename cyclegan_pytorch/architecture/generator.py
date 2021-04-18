"""
From the paper, Appendix, 7.2. Network architectures:
"We  adopt  our  architectures from  Johnson  et  al. [23].
We  use 6 residual  blocks  for 128×128 training images,
and 9 residual blocks for 256×256 or higher-resolution
training images. Below, we followthe naming convention
used in the Johnson et al.’s Github repository.
Let c7s1-k denote a 7×7 Convolution-InstanceNorm-ReLU layer
with k filters and stride 1. dk denotes a 3×3 Convolution-InstanceNorm-ReLU 
layer with k filters and stride 2. Reflection padding was
used to reduce artifacts. Rk denotes a residual block that
contains two 3×3 convolutional layers with the same number
of filters on both layer. uk denotes a 3×3 fractional-strided-Convolution-InstanceNorm-ReLU
layer with k filters and stride 1/2. 
The network with 6 residual blocks consists of:
c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3
The network with 9 residual blocks consists of:
c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128u64,c7s1-3"
"""
from torch import nn

from .residual_block import ResidualBlock


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape: list, num_residual_blocks: int = 9):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        # c7s1-64
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        # d128, d256
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        # R256 * <num_residual_blocks> (6 or 9 in the paper)
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        # u128, # u64
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        # c7s1-3
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features, channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
