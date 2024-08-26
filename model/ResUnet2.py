import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

# 定义解码器中的卷积块
class expansive_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()

        # 卷积块的结构
        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, d):
        # 上采样
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.block(d)
        return out

# 定义最后一层卷积块
def final_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    )
    return block

# 定义 Resnet34_Unet 类
class Resnet34_Unet(nn.Module):
    # 定义初始化函数
    def __init__(self, in_channel, out_channel, pretrained=False):
        # 调用 nn.Module 的初始化函数
        super(Resnet34_Unet, self).__init__()

        # 创建 ResNet34 模型
        self.resnet = models.resnet34(weights=None if not pretrained else 'IMAGENET1K_V1')
        # 修改第一层卷积层以接受单通道输入，并减少通道数
        self.resnet.conv1 = nn.Conv2d(in_channel, 8, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.bn1 = nn.BatchNorm2d(8)

        # 定义 layer0，包括 ResNet34 的第一层卷积、批归一化、ReLU 和最大池化层
        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )

        # 定义 Encode 部分，包括 ResNet34 的 layer1、layer2、layer3 和 layer4，减少通道数
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 定义 Bottleneck 部分，包括两个卷积层、ReLU 和批归一化层
        self.bottleneck = torch.nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=128, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        # 定义 Decode 部分，包括四个 expansive_block 和一个 final_block
        self.conv_decode3 = expansive_block(128, 64, 64)
        self.conv_decode2 = expansive_block(64, 32, 32)
        self.conv_decode1 = expansive_block(32, 16, 16)
        self.conv_decode0 = expansive_block(16, 8, 8)
        self.final_layer = final_block(8, out_channel)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 10)

    # 定义前向传播函数
    def forward(self, x):
        # 执行 layer0
        x = self.layer0(x)
        # 执行 Encode
        encode_block1 = self.layer1(x)
        encode_block2 = self.layer2(encode_block1)
        encode_block3 = self.layer3(encode_block2)
        encode_block4 = self.layer4(encode_block3)

        # 执行 Bottleneck
        bottleneck = self.bottleneck(encode_block4)

        # 执行 Decode
        decode_block3 = self.conv_decode3(bottleneck)
        decode_block2 = self.conv_decode2(decode_block3)
        decode_block1 = self.conv_decode1(decode_block2)
        decode_block0 = self.conv_decode0(decode_block1)
        final_layer = self.final_layer(decode_block0)

        x = self.avgpool(final_layer)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

# 测试网络
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Resnet34_Unet(in_channel=1, out_channel=8, pretrained=False).to(device)
summary(model, input_size=(1, 28, 28))
