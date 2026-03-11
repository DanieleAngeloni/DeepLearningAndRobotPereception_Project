import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()

        # Encoder  - base ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        # Livelli intermedi con skip connections
        self.layer1 = resnet.layer1  # Output: 256 canali
        self.layer2 = resnet.layer2  # Output: 512 canali
        self.layer3 = resnet.layer3  # Output: 1024 canali
        self.layer4 = resnet.layer4  # Output: 2048 canali

        # Decoder   - con skip connections + BatchNorm
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(1024 + 1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.upsample = nn.Upsample(size=(144, 256), mode="bilinear", align_corners=True)
        self.conv6 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.initial(x)
        x1 = self.layer1(x)  # 256 canali
        x2 = self.layer2(x1)  # 512 canali
        x3 = self.layer3(x2)  # 1024 canali
        x4 = self.layer4(x3)  # 2048 canali

        # Decoder con skip connections
        d1 = self.conv1(x4)  # 1024 canali

        d2 = F.interpolate(d1, size=x3.shape[2:], mode="bilinear", align_corners=True)
        d2 = self.conv2(torch.cat([d2, x3], dim=1))

        d3 = F.interpolate(d2, size=x2.shape[2:], mode="bilinear", align_corners=True)
        d3 = self.conv3(torch.cat([d3, x2], dim=1))

        d4 = F.interpolate(d3, size=x1.shape[2:], mode="bilinear", align_corners=True)
        d4 = self.conv4(torch.cat([d4, x1], dim=1))

        d5 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=True)
        d5 = self.conv5(d5)

        d5 = self.upsample(d5)
        output = self.conv6(d5)  # 1 canale

        return output
