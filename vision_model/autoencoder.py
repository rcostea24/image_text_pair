import torch
import torch.nn as nn

class Downsample(nn.Module):
    # downsample class
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        
        # conv layer with maxpool before
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # forward pass
        return self.conv(self.maxpool(x))

class Upsample(nn.Module):
    # upsample class
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        
        # upsample layer with factor of 2
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        # conv layer
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # dropout
        self.use_dropout = dropout
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # forward pass
        out = self.conv(self.up(x))
        return self.dropout(out) if self.use_dropout else out

class VisionModel(nn.Module):
    # class for autoencoder
    def __init__(self, vision_params):
        super(VisionModel, self).__init__()

        # init conv
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )

        # downsample layers
        self.down1 = Downsample(in_channels=4, out_channels=4)
        self.down2 = Downsample(in_channels=4, out_channels=4)
        self.down3 = Downsample(in_channels=4, out_channels=4)

        # upsample layers
        self.up1 = Upsample(in_channels=4, out_channels=4, dropout=True)
        self.up2 = Upsample(in_channels=4, out_channels=4, dropout=True)
        self.up3 = Upsample(in_channels=4, out_channels=4, dropout=False)

        # last conv
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # forward pass

        # init conv
        x = self.init_conv(x)

        # downsample the input
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        # save downsampled features
        features = x
        
        # reconstruct the image with upsample layers
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        
        # last conv
        x = self.last_conv(x)
        return [x, features]
        
        