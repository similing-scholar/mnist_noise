import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms
import numpy as np


class Basic2DDataset(Dataset):
    def __init__(self, mnist_dataset, encoder='train', noise_factor=0.1, seed=0):
        """
        对于训练和验证模型，需要传入图像和标签。
        对于预测模型，只有图像。
        """
        self.encoder = encoder  # encoder的值存储为实例变量
        self.data = mnist_dataset
        self.noise_factor = noise_factor
        self.seed = seed

    def __len__(self):
        return len(self.data)  # 数据集的长度是输入数据的数量

    def __getitem__(self, index):
        # 根据索引获取对应的图像和标签
        image, label = self.data[index]

        # 设置随机数种子
        np.random.seed(self.seed + index)

        # 生成热噪声并添加到图像
        noise = torch.tensor(np.random.normal(0, 1, image.size()), dtype=torch.float32)
        image = image + self.noise_factor * noise
        image = torch.clamp(image, 0, 1)  # 将图像像素值限制在 [0, 1] 范围内

        if self.encoder == 'train':  # 如果是训练模式，返回图像和标签对
            return image, torch.tensor(label, dtype=torch.long)  # 确保标签是long类型
        elif self.encoder == 'predict':  # 如果是预测模式，只返回图像
            return image


def load_dataset(mnist_dataset, train_ratio, val_ratio, batch_size):
    """加载数据集，划分数据集，返回数据加载器。
    train_ratio + val_ratio ≤ 1，否则给出报错提示
    """
    if train_ratio + val_ratio > 1.0:  # 检查train_ratio和val_ratio的合法性
        raise ValueError("train_ratio + val_ratio 必须小于等于 1.0")

    # 计算划分的样本数量
    total_samples = len(mnist_dataset)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    # 使用random_split划分数据集
    train_dataset, val_dataset, test_dataset = random_split(mnist_dataset,
                                                            [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(0)
                                                            )

    # 创建数据加载器
    loader_args = dict(batch_size=batch_size, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 定义MNIST数据集的预处理步骤
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))  #  标准化会影响噪声的添加
    ])
    # 加载MNIST训练数据集
    mnist_train_dataset = datasets.MNIST(root='../dataset', train=True, download=False, transform=transform)
    # 加载MNIST测试数据集
    mnist_test_dataset = datasets.MNIST(root='../dataset', train=False, download=False, transform=transform)

    # ----------实例化训练数据集----------
    train_dataset = Basic2DDataset(mnist_train_dataset, noise_factor=0.1, seed=42)
    train_loader, val_loader, test_loader = load_dataset(train_dataset, 0.8, 0.2, 64)

    # 遍历数据加载器以进行训练
    for input_batch, label_batch in train_loader:
        # 在这里进行模型的训练
        print(input_batch.shape, label_batch.shape)

    first_input_batch, first_label_batch = next(iter(train_loader))
    print("input:", first_input_batch[0])
    print("label:", first_label_batch[0])

    # ----------实例化测试数据集----------
    test_dataset = Basic2DDataset(mnist_test_dataset, noise_factor=0.1, seed=42)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

    # 遍历数据集进行测试
    for input_batch, label_batch in test_loader:
        print(input_batch.shape, label_batch.shape)