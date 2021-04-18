"""
From the paper, Appendix, 7.2. Network architectures:
"For discriminator networks, we use 70×70 PatchGAN [22].
Let Ck denote a 4×4 Convolution-InstanceNorm-LeakyReLU
layer with k filters and stride 2. After the last layer,
we apply a convolution to produce a1-dimensional output.
We do not use InstanceNorm for the first C64 layer.
We use leaky ReLUs with a slope of 0.2.
The discriminator architecture is:
C64-C128-C256-C512
"""
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, input_shape: list):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            # C64
            *discriminator_block(64, 128),
            # C128
            *discriminator_block(128, 256),
            # C256
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            # C512
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
