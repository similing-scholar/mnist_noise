import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms
import numpy as np


class Basic2DDataset(Dataset):
    def __init__(self, mnist_dataset, encoder='train', noise_levels=None, seed=0):
        """
        对于训练和验证模型，需要传入图像和标签。
        对于预测模型，只有图像。
        noise_levels: 列表，包含不同噪声水平。默认值为 [0.1]。
        """
        self.encoder = encoder
        self.data = mnist_dataset
        self.noise_levels = noise_levels if noise_levels else [0.1]
        self.seed = seed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]

        # 设置随机数种子
        np.random.seed(self.seed + index)

        # 确定噪声水平
        noise_level = self.noise_levels[index % len(self.noise_levels)]

        # 生成热噪声并添加到图像
        noise = torch.tensor(np.random.normal(0, 1, image.size()), dtype=torch.float32)  # 好像加错了
        image = image + noise_level * noise
        image = torch.clamp(image, 0, 1)

        if self.encoder == 'train':
            return image, torch.tensor(label, dtype=torch.long)
        elif self.encoder == 'predict':
            return image


def load_dataset(mnist_dataset, train_ratio, val_ratio, batch_size):
    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio 必须小于等于 1.0")

    total_samples = len(mnist_dataset)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(mnist_dataset,
                                                            [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=batch_size, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    mnist_train_dataset = datasets.MNIST(root='../dataset', train=True, download=False, transform=transform)
    mnist_test_dataset = datasets.MNIST(root='../dataset', train=False, download=False, transform=transform)

    noise_factor = 'mix'
    if noise_factor == 'mix':
        noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    else:
        noise_levels = [noise_factor]

    train_dataset = Basic2DDataset(mnist_train_dataset, noise_levels=noise_levels, seed=42)
    train_loader, val_loader, test_loader = load_dataset(train_dataset, 0.8, 0.2, 64)

    for input_batch, label_batch in train_loader:
        print(input_batch.shape, label_batch.shape)

    first_input_batch, first_label_batch = next(iter(train_loader))
    print("input:", first_input_batch[0])
    print("label:", first_label_batch[0])

    test_dataset = Basic2DDataset(mnist_test_dataset, noise_levels=noise_levels, seed=42)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

    for input_batch, label_batch in test_loader:
        print(input_batch.shape, label_batch.shape)
