import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super(CNNBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        

    def forward(self, x):
        residual = self.residual_conv(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out
        
class VisionModel(nn.Module):
    # Vision model inspired by ResNet18
    def __init__(self, params):
        super(VisionModel, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = nn.Sequential(
            CNNBlock(64, 64, False),
            CNNBlock(64, 64, False)
        )
        self.layer2 = nn.Sequential(
            CNNBlock(64, 128, False),
            CNNBlock(128, 128, True)
        )
        self.layer3 = nn.Sequential(
            CNNBlock(128, 256, False),
            CNNBlock(256, 256, True)
        )
        self.layer4 = nn.Sequential(
            CNNBlock(256, 512, False),
            CNNBlock(512, 512, True)
        )

        self.avg_pool = nn.AvgPool2d(7)

        self.flatten = nn.Flatten()

    def forward(self, x):
        # print(x.shape)
        x = self.first_layer(x)
        x = self.maxpool(x)
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)

        x = self.avg_pool(x)
        # print(x.shape)

        x = self.flatten(x)

        return x