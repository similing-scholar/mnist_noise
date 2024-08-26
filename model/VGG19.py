import torch
import torch.nn as nn
from torchsummary import summary


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(VGGBlock, self).__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGG19(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG19, self).__init__()
        self.features = nn.Sequential(
            VGGBlock(1, 16, 2),  # conv block 1
            VGGBlock(16, 32, 2),  # conv block 2
            VGGBlock(32, 64, 4),  # conv block 3
            VGGBlock(64, 128, 4)  # conv block 4
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = VGG19(num_classes=10)  # 适用于MNIST数据集的10分类任务
    model.initialize_weights()
    model.to(torch.device("cuda:0"), dtype=torch.float32)
    print(model)
    summary(model, (1, 28, 28))  # MNIST数据集的输入尺寸
