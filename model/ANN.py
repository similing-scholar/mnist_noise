import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ANN(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        '''
        :param input_size: 输入层的神经元数量
        :param num_classes: 分类的类别数量
        假设输入进来的图片是28*28（flatten之后是784）
        '''
        super(ANN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        # 全连接网络
        self.classifier = nn.Sequential(
            # input_size -> 512
            nn.Linear(in_features=self.input_size, out_features=1024),
            nn.ReLU(),
            # 1024-> 512
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            # 512 -> 256
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            # 256 -> 128
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            # 128 -> num_classes
            nn.Linear(in_features=128, out_features=self.num_classes),
        )

    def forward(self, inputs):
        # 展平,[注意1]第0维是batch_size,默认从第1维开始
        x = torch.flatten(inputs, 1)
        # 分类
        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

if __name__ == "__main__":
    model = ANN(input_size=28*28, num_classes=10)
    model.initialize_weights()
    model.to(torch.device("cuda:0"), dtype=torch.float32)
    print(model)
    summary(model, (1, 28, 28))  # 输入的形状应为1*28*28，但会在forward函数中展平为784
