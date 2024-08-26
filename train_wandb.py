import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import time
import pandas as pd
import wandb

from model.load_dataset_interface_v1 import Basic2DDataset, load_dataset
from utils.utils import load_h5, normalize_data
from model.MSRNet import MSRNet_2D
from model.LeNet5 import LeNet5
from model.ResNet50 import ResNet50
from model.VGG19 import VGG19
from model.SqueezeNet import SqueezeNet
from torchvision import datasets, transforms
from model.train_model_interface import KerasModel
from utils.train_plot_interface import plot_metric

# ------------------------------------ step 0/5 : 配置设备------------------------------------
USE_GPU = True  # 设置设备（CPU或GPU）
device = torch.device("cuda:0" if USE_GPU else "cpu")
dtype = torch.float32  # 将模型的权重参数转换为 float32 类型
print(f'using device:{device}, dtype:{dtype}')


def train_model_with_noise_factor(noise_factor, net_name='LeNet5'):
    # Initialize wandb run
    wandb.init(project="MNIST_Classification", config={
        "noise_factor": noise_factor,
        "architecture": net_name,
        "dataset": "MNIST",
        "epochs": 1000,
        "batch_size": 64,
        "learning_rate": 1e-3
    })

    config = wandb.config

    # ------------------------------------ step 1/5 : 加载数据------------------------------------
    # 定义MNIST数据集的预处理步骤
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # 加载MNIST训练数据集
    mnist_train_dataset = datasets.MNIST(root='../dataset', train=True, download=False, transform=transform)
    # 创建数据集
    train_dataset = Basic2DDataset(mnist_train_dataset, noise_factor=noise_factor, seed=42)

    # 定义划分比例和batch大小
    train_ratio = 0.8
    val_ratio = 0.2
    batch_size = config.batch_size
    # 加载数据集(观测值y和label)
    train_loader, val_loader, test_loader = load_dataset(train_dataset, train_ratio, val_ratio, batch_size)

    # ------------------------------------ step 2/5 : 创建网络------------------------------------
    if net_name == 'MSRNet':
        num_blocks = 3
        block_channels = 8
        model = MSRNet_2D(num_blocks=num_blocks, block_channels=block_channels, input_size=(28, 28))  # MSRNET
        model_parameter = f'({num_blocks}block{block_channels}channel)'
    elif net_name == 'ResNet50':
        model = ResNet50(num_classes=10, grayscale=True)
        model_parameter = ''
    elif net_name == 'VGG19':
        model = VGG19(num_classes=10)
        model_parameter = ''
    elif net_name == 'LeNet5':
        model = LeNet5(grayscale=True)
        model_parameter = ''

    # 选择网络
    net = model
    net.initialize_weights()  # 初始化权值
    net.to(device, dtype)  # 将网络移动到gpu/cpu
    # 是否使用预训练权重
    pre_training = False
    pre_training_data = f'logs/202406262318_MSRNet(2block4channel)_1noise_checkpoint.pt'
    if pre_training:
        net.load_state_dict(torch.load(pre_training_data))  # 加载预训练权重

    # ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------
    model = KerasModel(device=device, dtype=dtype, net=net,
                       metrics_dict={"acc": torchmetrics.Accuracy(task='multiclass', num_classes=10).to(device)},
                       loss_fn=nn.CrossEntropyLoss(),
                       optimizer=optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=0)
                       )

    # ------------------------------------ step 4/5 : 训练模型并保存 --------------------------------------------------
    current_time = time.strftime("%Y%m%d%H%M", time.localtime())
    ckpt_path = f'logs/{current_time}_{net_name}{model_parameter}_{noise_factor}noise_checkpoint.pt'
    model.fit(train_data=train_loader,
              val_data=val_loader,
              epochs=config.epochs,
              ckpt_path=ckpt_path,
              patience=15,
              monitor='val_acc',
              mode='max')  # 监控的指标为验证集上的准确率，模式为最大化

    # Log metrics to wandb
    wandb.log({"val_acc": max(model.history["val_acc"]), "val_loss": min(model.history["val_loss"])})

    # ------------------------------------ step 5/5 : 绘制训练曲线评估模型 --------------------------------------------------
    df_history = pd.DataFrame(model.history)  # 将训练过程中的损失和指标数据保存为DataFrame格式
    csv_path = f'logs/{current_time}_{net_name}{model_parameter}_{noise_factor}noise_history.csv'
    df_history.to_csv(csv_path, index=False)  # 保存为csv文件

    # 绘制损失曲线
    fig_loss = plot_metric(df_history, "loss")
    fig_loss_path = f'logs/{current_time}_{net_name}{model_parameter}_{noise_factor}noise_figloss.png'
    fig_loss.savefig(fig_loss_path, dpi=300)
    wandb.log({"loss_curve": wandb.Image(fig_loss)})

    # 绘制准确率曲线
    fig_accuracy = plot_metric(df_history, "acc")
    fig_accuracy_path = f'logs/{current_time}_{net_name}{model_parameter}_{noise_factor}noise_figacc.png'
    fig_accuracy.savefig(fig_accuracy_path, dpi=300)
    wandb.log({"accuracy_curve": wandb.Image(fig_accuracy)})

    wandb.finish()


if __name__ == "__main__":
    noise_factors = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    net_names = ['ResNet50', 'VGG19', 'MSRNet']
    for net_name in net_names:
        for noise_factor in noise_factors:
            train_model_with_noise_factor(noise_factor, net_name)
