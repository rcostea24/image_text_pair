import torch
import torch.nn as nn

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
            
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(self.maxpool(x))

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.use_dropout = dropout
            
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        out = self.conv(self.up(x))
        return self.dropout(out) if self.use_dropout else out

class VisionModel(nn.Module):
    def __init__(self, vision_params):
        super(VisionModel, self).__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )

        self.down1 = Downsample(in_channels=4, out_channels=4)
        self.down2 = Downsample(in_channels=4, out_channels=4)
        self.down3 = Downsample(in_channels=4, out_channels=4)

        self.up1 = Upsample(in_channels=4, out_channels=4, dropout=True)
        self.up2 = Upsample(in_channels=4, out_channels=4, dropout=True)
        self.up3 = Upsample(in_channels=4, out_channels=4, dropout=False)

        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # print(x.shape)
        x = self.init_conv(x)
        # print(x.shape)
        x = self.down1(x)
        # print(x.shape)
        x = self.down2(x)
        # print(x.shape)
        x = self.down3(x)
        # print(x.shape)
        # print(x.shape)

        
        features = x
        
        x = self.up1(x)
        # print(x.shape)
        x = self.up2(x)
        # print(x.shape)
        x = self.up3(x)
        # print(x.shape)
        # print(x.shape)
        
        x = self.last_conv(x)
        # print(x.shape)
        return [x, features]
        
        