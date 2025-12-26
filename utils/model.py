
import torch
from torch import nn
import torchvision.models as models


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            SEBlock(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNetColorizationNet(nn.Module):
    def __init__(self):
        super(UNetColorizationNet, self).__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        self.encoder_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool

        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2
        self.encoder_layer3 = resnet.layer3
        self.encoder_layer4 = resnet.layer4
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.dec4 = DecoderBlock(in_channels=512 + 256, mid_channels=512, out_channels=256)

        self.dec3 = DecoderBlock(in_channels=256 + 128, mid_channels=256, out_channels=128)
        
        self.dec2 = DecoderBlock(in_channels=128 + 64, mid_channels=128, out_channels=64)

        self.dec1 = DecoderBlock(in_channels=64 + 64, mid_channels=64, out_channels=64)

        self.dec0 = DecoderBlock(in_channels=64 + 1, mid_channels=32, out_channels=32)

        self.output_head = nn.Conv2d(32, 313, kernel_size=3, padding=1)

    def forward(self, x_in):
        x_skip1 = self.encoder_conv1(x_in)
        x_skip1 = self.encoder_bn1(x_skip1)
        x_skip1 = self.encoder_relu(x_skip1) 
        
        x_pool = self.encoder_maxpool(x_skip1)
        x_skip2 = self.encoder_layer1(x_pool)
        
        x_skip3 = self.encoder_layer2(x_skip2)
        
        x_skip4 = self.encoder_layer3(x_skip3)
        
        x_bottle = self.encoder_layer4(x_skip4)
        
        x = self.upsample(x_bottle)
        x = torch.cat([x, x_skip4], dim=1)
        x = self.dec4(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_skip3], dim=1)
        x = self.dec3(x)

        x = self.upsample(x)
        x = torch.cat([x, x_skip2], dim=1)
        x = self.dec2(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_skip1], dim=1)
        x = self.dec1(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_in], dim=1)
        x = self.dec0(x)
        
        logits = self.output_head(x)
        
        return logits
    
