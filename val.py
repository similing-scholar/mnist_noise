"""
使用测试数据集来做验证
"""
import torch
import matplotlib.pyplot as plt

from model.load_dataset_interface_v3 import Basic2DDataset, load_dataset
from model.MSRNet import MSRNet_2D, MSRNet2
from model.LeNet5 import LeNet5
from model.ResNet50 import ResNet50
from model.ResNet18 import ResNet18
from model.VGG19 import VGG19
from model.Unet import UNet
from model.SmallUnet import UNet2
from model.ResUNet import ResUNet
from model.ResUnet2 import Resnet34_Unet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ------------------------------------ step 0/4 : 配置设备------------------------------------
USE_GPU = True  # 设置设备（CPU或GPU）
device = torch.device("cuda:0" if USE_GPU else "cpu")
dtype = torch.float32  # 将模型的权重参数转换为 float32 类型
print(f'using device:{device}, dtype:{dtype}')

# ------------------------------------ step 1/4 : 加载数据------------------------------------
# 定义MNIST数据集的预处理步骤
transform = transforms.Compose([
    transforms.ToTensor(),
])
# 加载MNIST训练数据集
mnist_train_dataset = datasets.MNIST(root='../dataset', train=True, download=False, transform=transform)
# 加载MNIST测试数据集
mnist_test_dataset = datasets.MNIST(root='../dataset', train=False, download=False, transform=transform)
# 第2类：实例化数据集（包含输入和输出）
# 实例化测试数据集
noise_type = 'percent gaussian'  # gaussian, percent gaussian, striped, mosaic
noise_factor = 0.362
if noise_factor == 'mix':
    noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
else:
    noise_levels = [noise_factor]
test_dataset = Basic2DDataset(mnist_test_dataset, noise_type=noise_type, noise_levels=noise_levels, seed=42)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)


# ------------------------------------ step 2/4 : 加载网络------------------------------------
net_name = 'UNet2'

if net_name == 'MSRNet':
    num_blocks = 8
    block_channels = 4
    model = MSRNet_2D(num_blocks=num_blocks, block_channels=block_channels, input_size=(28, 28))  # MSRNET
    model_parameter = f'({num_blocks}block{block_channels}channel)'
if net_name == 'MSRNet2':
    num_blocks = 4
    block_channels = 8
    model = MSRNet2(num_blocks=num_blocks, block_channels=block_channels, input_size=(28, 28))  # MSRNET
    model_parameter = f'({num_blocks}block{block_channels}channel)'
elif net_name == 'ResNet50':
    model = ResNet50(num_classes=10, grayscale=True)
    model_parameter = ''
elif net_name == 'ResNet18':
    model = ResNet18(num_classes=10, grayscale=True)
    model_parameter = ''
elif net_name == 'VGG19':
    model = VGG19(num_classes=10)
    model_parameter = ''
elif net_name == 'LeNet5':
    model = LeNet5(grayscale=True)
    model_parameter = ''
elif net_name == 'UNet':
    model = UNet(n_channels=1, n_classes=10)
    model_parameter = ''
elif net_name == 'UNet2':
    model = UNet2(n_channels=1, n_classes=10)
    model_parameter = ''
elif net_name == 'ResUNet':
    model = ResUNet(n_channels=1, n_classes=10)
    model_parameter = ''
elif net_name == 'ResUNet2':
    model = Resnet34_Unet(in_channel=1, out_channel=8, pretrained=False)
    model_parameter = ''

# 选择网络
net = model
net.initialize_weights()  # 初始化权值
net.to(device, dtype)  # 将网络移动到gpu/cpu

# ------------------------------------ step 3/4 : 加载模型权重 ------------------------------------
ckpt_path = 'logs/202407120005_UNet2_gaussian1.0noise_checkpoint.pt'
# 方法2：因为预测只需要网络和权重，可以在net中使用torch的load_state_dict
model = net
model.load_state_dict(torch.load(ckpt_path))

# ------------------------------------ step 4/4 : 模型验证结果展示 ------------------------------------
model.eval()  # 切换模型到评估模式
with torch.no_grad():
    for batch in test_loader:
        input, label = batch  # 对于train_dataset，batch有两个元素
        input = input.to(device, dtype)
        label = label.to(device, torch.long)

        result = model(input)
        _, predicted = torch.max(result, 1)

        # 展示前10个样本
        for i in range(min(10, input.size(0))):
            input_img = input[i+9].cpu().numpy().squeeze()
            label_true = label[i+9].item()
            label_pred = predicted[i+9].item()

            plt.figure(figsize=(5, 5))
            plt.imshow(input_img, cmap='gray')
            # plt.title(f'True Label: {label_true}, Predicted Label: {label_pred}, noise_ratio:{noise_factor} ')
            plt.title(f'noise_ratio:{noise_factor} ')
            plt.axis('off')
            plt.show()
