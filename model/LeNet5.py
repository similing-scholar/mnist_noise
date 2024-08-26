import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, grayscale=True):
        '''
        :param num_classes: 分类的类别数量
        :param grayscale: 是否为灰度图
        假设输入进来的图片是1,28,28
        '''
        super(LeNet5, self).__init__()
        self.num_classes = num_classes
        self.grayscale = grayscale
        self.in_channels = 1 if grayscale else 3

        # 卷积网络
        self.extract_feature = nn.Sequential(
            # 1,28,28 -> 6,24,24
            nn.Conv2d(in_channels=self.in_channels, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            # 6,24,24 -> 6,12,12
            nn.MaxPool2d(kernel_size=2),
            # 6,12,12 -> 16,8,8
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            # 16,8,8 -> 16,4,4
            nn.MaxPool2d(kernel_size=2),
        )

        # 分类器
        self.classifier = nn.Sequential(
            # 16*4*4 -> 120
            nn.Linear(in_features=16*4*4, out_features=120),
            nn.ReLU(),
            # 120 -> 84
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            # 84 -> num_classes
            nn.Linear(in_features=84, out_features=self.num_classes),
        )

    def forward(self, inputs):
        # 提取特征
        x = self.extract_feature(inputs)
        # 展平,[注意1]第0维是batch_size,默认从第1维开始;[注意2]forward使用函数，而不是层nn.Flatten
        x = torch.flatten(x, 1)
        # 分类
        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


if __name__ == "__main__":
    model = LeNet5(grayscale=True)  # 确保在初始化时设置为灰度图像
    model.initialize_weights()
    model.to(torch.device("cuda:0"), dtype=torch.float32)
    print(model)
    summary(model, (1, 28, 28))  # 输入的通道数应与模型期望的通道数一致
