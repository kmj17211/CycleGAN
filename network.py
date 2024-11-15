import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.conv_block = nn.Sequential(nn.ReflectionPad2d(1),
                                        nn.Conv2d(in_channel, in_channel, 3),
                                        nn.InstanceNorm2d(in_channel),
                                        nn.ReLU(inplace=True),
                                        nn.ReflectionPad2d(1),
                                        nn.Conv2d(in_channel, in_channel, 3),
                                        nn.InstanceNorm2d(in_channel))
        
    def forward(self, x):
        return x + self.conv_block(x)
    
class Generator(nn.Module):
    def __init__(self, in_channel = 1, out_channel = 1, n_residual_block = 6, bias = True):
        super().__init__()

        conv_dim = 64

        # Down Sampling Layer
        # 128 x 128 x 1
        down_layers = []
        down_layers.append(nn.ReflectionPad2d(3))
        down_layers.append(nn.Conv2d(in_channel, conv_dim, kernel_size=7, bias=bias))
        down_layers.append(nn.InstanceNorm2d(conv_dim))
        down_layers.append(nn.ReLU(inplace=True))
        
        # 128 x 128 x 64
        down_layers.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=3, stride=2, padding=1, bias=bias))
        down_layers.append(nn.InstanceNorm2d(conv_dim*2))
        down_layers.append(nn.ReLU(inplace=True))

        # 64 x 64 x 128
        down_layers.append(nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=3, stride=2, padding=1, bias=bias))
        down_layers.append(nn.InstanceNorm2d(conv_dim*4))
        down_layers.append(nn.ReLU(inplace=True))

        # Bottleneck Layer
        # 32 x 32 x 256
        bottle_layer = []
        for i in range(n_residual_block):
            bottle_layer.append(ResidualBlock(conv_dim*4))

        # Up Sampling Layer
        # 32 x 32 x 256
        up_layer = []
        up_layer.append(nn.ConvTranspose2d(conv_dim*4, conv_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias))
        up_layer.append(nn.InstanceNorm2d(conv_dim*2))
        up_layer.append(nn.ReLU(inplace=True))

        # 64 x 64 x 128
        up_layer.append(nn.ConvTranspose2d(conv_dim*2, conv_dim, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias))
        up_layer.append(nn.InstanceNorm2d(conv_dim))
        up_layer.append(nn.ReLU(inplace=True))

        # 128 x 128 x 64
        up_layer.append(nn.ReflectionPad2d(3))
        up_layer.append(nn.Conv2d(conv_dim, out_channel, kernel_size=7, bias=bias))
        up_layer.append(nn.Tanh()) # Tanh() -> ReLU()

        # 128 x 128 x 1
        self.down = nn.Sequential(*down_layers)
        self.bottle = nn.Sequential(*bottle_layer)
        self.up = nn.Sequential(*up_layer)

    def forward(self, x):
        x = self.down(x)
        x = self.bottle(x)
        x = self.up(x)

        return x
    
class Discriminator(nn.Module):
    def __init__(self, in_channel=1):
        super().__init__()

        conv_dim = 64

        # 128 x 128 x 1
        model = []
        model.append(nn.Conv2d(in_channel, conv_dim, kernel_size=4, stride=2, padding=1))
        model.append(nn.LeakyReLU(0.2, inplace=True))

        # 64 x 64 x 64
        model.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1))
        model.append(nn.InstanceNorm2d(conv_dim*2))
        model.append(nn.LeakyReLU(0.2, inplace=True))

        # 32 x 32 x 128
        model.append(nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=4, stride=2, padding=1))
        model.append(nn.InstanceNorm2d(conv_dim*4))
        model.append(nn.LeakyReLU(0.2, inplace=True))

        # 16 x 16 x 256
        model.append(nn.Conv2d(conv_dim*4, 1, kernel_size=3))

        # 15 x 15 x 1
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x