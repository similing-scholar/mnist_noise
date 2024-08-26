import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import time
import pandas as pd
from matplotlib import pyplot as plt

import wandb

from model.load_dataset_interface_v3 import Basic2DDataset, load_dataset
from utils.utils import load_h5, normalize_data
from model.MSRNet import MSRNet_2D, MSRNet2
from model.LeNet5 import LeNet5
from model.ResNet50 import ResNet50
from model.ResNet18 import ResNet18
from model.VGG19 import VGG19
from model.Unet import UNet
from model.SmallUnet import UNet2
from model.ResUNet import ResUNet
from model.ResUnet2 import Resnet34_Unet
from model.ANN import ANN
from torchvision import datasets, transforms
from model.train_model_interface import KerasModel
from utils.train_plot_interface import plot_metric

# ------------------------------------ step 0/5 : 配置设备------------------------------------
USE_GPU = True  # 设置设备（CPU或GPU）
device = torch.device("cuda:0" if USE_GPU else "cpu")
dtype = torch.float32  # 将模型的权重参数转换为 float32 类型
print(f'using device:{device}, dtype:{dtype}')

# ------------------------------------ step 1/5 : 加载数据------------------------------------
# 定义MNIST数据集的预处理步骤
transform = transforms.Compose([
    transforms.ToTensor(),
])
# 加载MNIST训练数据集
mnist_train_dataset = datasets.MNIST(root='../dataset', train=True, download=False, transform=transform)
# 创建数据集
noise_type = 'gaussian'  # gaussian, striped, mosaic
noise_factor = 'mix'
if noise_factor == 'mix':
    noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
else:
    noise_levels = [noise_factor]

train_dataset = Basic2DDataset(mnist_train_dataset, noise_levels=noise_levels, noise_type=noise_type, seed=42)

# 定义划分比例和batch大小
train_ratio = 0.8
val_ratio = 0.2
batch_size = 64
# 加载数据集(观测值y和label)
train_loader, val_loader, test_loader = load_dataset(train_dataset, train_ratio, val_ratio, batch_size)

# ------------------------------------ step 2/5 : 创建网络------------------------------------
net_name = 'ANN'

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
elif net_name == 'ANN':
    model = ANN()
    model_parameter = ''

# 选择网络
net = model
net.initialize_weights()  # 初始化权值
net.to(device, dtype)  # 将网络移动到gpu/cpu
# 是否使用预训练权重
pre_training = False
pre_training_data = f'logs/202406302055_MSRNet(3block8channel)_1.0noise_checkpoint.pt'
if pre_training:
    net.load_state_dict(torch.load(pre_training_data))  # 加载预训练权重

# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------
model = KerasModel(device=device, dtype=dtype, net=net,
                   metrics_dict={"acc": torchmetrics.Accuracy(task='multiclass', num_classes=10).to(device)},
                   loss_fn=nn.CrossEntropyLoss(),
                   optimizer=optim.Adam(net.parameters(), lr=1e-3, weight_decay=0)
                   )

# ------------------------------------ step 4/5 : 训练模型并保存 --------------------------------------------------
time = time.strftime("%Y%m%d%H%M", time.localtime())
ckpt_path = f'logs/{time}_{net_name}{model_parameter}_{noise_type}{noise_factor}noise_checkpoint.pt'
model.fit(train_data=train_loader,
          val_data=val_loader,
          epochs=500,
          ckpt_path=ckpt_path,
          patience=15,
          monitor='val_acc',
          mode='max')  # 监控的指标为验证集上的损失函数，模式为最小化

# ------------------------------------ step 5/5 : 绘制训练曲线评估模型 --------------------------------------------------
df_history = pd.DataFrame(model.history)  # 将训练过程中的损失和指标数据保存为DataFrame格式
csv_path = f'logs/{time}_{net_name}{model_parameter}_{noise_type}{noise_factor}noise_history.csv'
df_history.to_csv(csv_path, index=False)  # 保存为csv文件

# 绘制损失曲线
fig_loss = plot_metric(df_history, "loss")
fig_loss_path = f'logs/{time}_{net_name}{model_parameter}_{noise_type}{noise_factor}noise_figloss.png'
fig_loss.savefig(fig_loss_path, dpi=300)
plt.close(fig_loss)

# 绘制准确率曲线
fig_accuracy = plot_metric(df_history, "acc")
fig_accuracy_path = f'logs/{time}_{net_name}{model_parameter}_{noise_type}{noise_factor}noise_figacc.png'
fig_accuracy.savefig(fig_accuracy_path, dpi=300)
plt.close(fig_accuracy)
