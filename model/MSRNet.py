import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ConvLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=True):
        super(ConvLayer2D, self).__init__()
        self.activation = activation
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                              padding=self.padding, bias=True)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            return F.relu(x)
        else:
            return x


class MultiScaleResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleResidualBlock2D, self).__init__()
        self.conv5_1 = ConvLayer2D(in_channels=in_channels, out_channels=out_channels, kernel_size=5)
        self.conv3_1 = ConvLayer2D(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.conv5_2 = ConvLayer2D(in_channels=in_channels * 2, out_channels=out_channels * 2, kernel_size=5)
        self.conv3_2 = ConvLayer2D(in_channels=in_channels * 2, out_channels=out_channels * 2, kernel_size=3)
        self.bottleneck = ConvLayer2D(in_channels=in_channels * 4, out_channels=out_channels, kernel_size=1, activation=False)

    def forward(self, x):
        P1 = self.conv5_1(x)
        S1 = self.conv3_1(x)
        P2 = self.conv5_2(torch.cat([P1, S1], 1))
        S2 = self.conv3_2(torch.cat([P1, S1], 1))
        S = self.bottleneck(torch.cat([P2, S2], 1))
        return x + S


class MSRNet_2D(nn.Module):
    def __init__(self, num_blocks, block_channels, input_size):
        super(MSRNet_2D, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=block_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.PReLU()
        self.residual_modules = nn.ModuleList([MultiScaleResidualBlock2D(in_channels=block_channels, out_channels=block_channels) for _ in range(num_blocks)])
        self.feature_fusion = ConvLayer2D(in_channels=block_channels * (num_blocks + 1), out_channels=block_channels,
                                          kernel_size=1, activation=False)
        self.conv2 = nn.Conv2d(in_channels=block_channels, out_channels=block_channels, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(block_channels, 10)  # 添加一个全连接层作为分类器

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu(output)
        residual_conv = output
        block_outputs = [residual_conv]
        for module in self.residual_modules:
            output = module(output)
            block_outputs.append(output)
        output = self.feature_fusion(torch.cat(block_outputs, 1))
        output = self.conv2(output)
        output = self.global_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)  # 添加全连接层
        return output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class MSRNet2(nn.Module):
    def __init__(self, num_blocks, block_channels, input_size):
        super(MSRNet2, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=block_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.residual_modules = nn.ModuleList([MultiScaleResidualBlock2D(in_channels=block_channels, out_channels=block_channels) for _ in range(num_blocks)])
        self.feature_fusion = ConvLayer2D(in_channels=block_channels * (num_blocks + 1), out_channels=block_channels,
                                          kernel_size=1, activation=False)
        self.conv2 = nn.Conv2d(in_channels=block_channels, out_channels=block_channels, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 分类器
        self.classifier = nn.Sequential(
            # 16*4*4 -> 120
            nn.Linear(in_features=block_channels, out_features=120),
            nn.ReLU(),
            # 120 -> 84
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            # 84 -> num_classes
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu(output)
        residual_conv = output
        block_outputs = [residual_conv]
        for module in self.residual_modules:
            output = module(output)
            block_outputs.append(output)
        output = self.feature_fusion(torch.cat(block_outputs, 1))
        output = self.conv2(output)
        output = self.global_pool(output)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)  # 添加全连接层
        return output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


if __name__ == "__main__":
    num_blocks = 2
    model = MSRNet2(num_blocks=num_blocks, block_channels=16, input_size=(28, 28))
    model.initialize_weights()
    model.to(torch.device("cuda:0"), dtype=torch.float32)
    print(model)
    summary(model, (1, 28, 28))
