import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    # class for cnn block
    def __init__(self, in_channels, out_channels, downsample):
        super(CNNBlock, self).__init__()

        # set the first conv layer
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # set the second conv layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # activation function
        self.relu = nn.ReLU()
        
        # convolution for skip connection
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        

    def forward(self, x):
        # forward pass

        # apply convolution for skip connection
        residual = self.residual_conv(x)
        
        # first convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # second convolution
        out = self.conv2(out)
        out = self.bn2(out)
        
        # skip-connection
        out += residual
        out = self.relu(out)
        
        return out
        
class VisionModel(nn.Module):
    # vision model inspired by ResNet18
    def __init__(self, params):
        super(VisionModel, self).__init__()

        # initial conv layer
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # first block
        self.layer1 = nn.Sequential(
            CNNBlock(64, 64, False),
            CNNBlock(64, 64, False)
        )

        # second block
        self.layer2 = nn.Sequential(
            CNNBlock(64, 128, False),
            CNNBlock(128, 128, True)
        )

        # third block
        self.layer3 = nn.Sequential(
            CNNBlock(128, 256, False),
            CNNBlock(256, 256, True)
        )

        # forth block
        self.layer4 = nn.Sequential(
            CNNBlock(256, 512, False),
            CNNBlock(512, 512, True)
        )

        # avg pool and flatten layers
        self.avg_pool = nn.AvgPool2d(7)

        self.flatten = nn.Flatten()

    def forward(self, x):
        # init conv with maxpool
        x = self.first_layer(x)
        x = self.maxpool(x)

        # forward all convolutional blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # avg pool and flatten
        x = self.avg_pool(x)
        x = self.flatten(x)

        return x