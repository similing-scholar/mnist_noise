import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# 定义MNIST数据集的预处理步骤
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    # transforms.Normalize((0.1307,), (0.3081,))  # 标准化图像
])
# 加载MNIST训练数据集
mnist_train_dataset = datasets.MNIST(root='../dataset', train=True, download=False, transform=transform)
# 加载MNIST测试数据集
mnist_test_dataset = datasets.MNIST(root='../dataset', train=False, download=False, transform=transform)

# 定义标准化和反标准化的transform
normalize = transforms.Normalize((0.1307,), (0.3081,))
unnormalize = transforms.Normalize((-0.1307/0.3081,), (1/0.3081,))

class Basic2DDataset(Dataset):
    def __init__(self, mnist_dataset, encoder='train', noise_factor=0.1, noise_type='gaussian', seed=0, return_original=False):
        """
        对于训练和验证模型，需要传入图像和标签。
        对于预测模型，只有图像。
        """
        self.encoder = encoder  # encoder的值存储为实例变量
        self.data = mnist_dataset
        self.noise_factor = noise_factor
        self.noise_type = noise_type
        self.seed = seed
        self.return_original = return_original  # 是否返回原始图像

    def __len__(self):
        return len(self.data)  # 数据集的长度是输入数据的数量

    def __getitem__(self, index):
        # 根据索引获取对应的图像和标签
        image, label = self.data[index]

        # 设置随机数种子
        np.random.seed(self.seed + index)

        # 添加噪声
        if self.noise_type == 'gaussian':
            noise = torch.tensor(np.random.normal(0, self.noise_factor, image.size()), dtype=torch.float32)
        elif self.noise_type == 'striped':
            noise = torch.tensor(self._generate_striped_noise(image.size(), self.noise_factor), dtype=torch.float32)
        elif self.noise_type == 'mosaic':
            noise = torch.tensor(self._generate_mosaic_noise(image.size(), self.noise_factor), dtype=torch.float32)
        else:
            raise ValueError("Unsupported noise type")

        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, 0, 1)  # 将图像像素值限制在 [0, 1] 范围内

        if self.return_original:
            return noisy_image, label, image  # 返回原始图像用于可视化
        else:
            if self.encoder == 'train':  # 如果是训练模式，返回图像和标签对
                return noisy_image, label
            elif self.encoder == 'predict':  # 如果是预测模式，只返回图像
                return noisy_image

    def _generate_striped_noise(self, size, noise_factor):
        noise = np.zeros(size)
        stripe_width = 2
        for i in range(0, size[1], stripe_width * 2):
            noise[:, i:i+stripe_width] = noise_factor
        return noise

    def _generate_mosaic_noise(self, size, noise_factor):
        noise = np.random.rand(*size)
        mosaic_size = 4
        for i in range(0, size[1], mosaic_size):
            for j in range(0, size[2], mosaic_size):
                block_value = np.random.normal(0, noise_factor)
                noise[:, i:i+mosaic_size, j:j+mosaic_size] = block_value
        return noise

def load_dataset(mnist_dataset, train_ratio, val_ratio, batch_size, return_original=False):
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
                                                            generator=torch.Generator().manual_seed(0))

    # 实例化数据集，设置return_original参数
    train_dataset.dataset.return_original = return_original
    val_dataset.dataset.return_original = return_original

    # 创建数据加载器
    loader_args = dict(batch_size=batch_size, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader

def visualize_noisy_images(data_loader, num_images=5):
    data_iter = iter(data_loader)
    fig, axs = plt.subplots(num_images, 2, figsize=(10, 2 * num_images))

    for i in range(num_images):
        noisy_image, label, original_image = next(data_iter)
        noisy_image = noisy_image[0].squeeze().numpy()
        original_image = original_image[0].squeeze().numpy()

        axs[i, 0].imshow(original_image, cmap='gray')
        axs[i, 0].set_title(f'Original Image - Label: {label[0].item()}')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(noisy_image, cmap='gray')
        axs[i, 1].set_title('Noisy Image')
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # ----------实例化训练数据集----------
    train_dataset = Basic2DDataset(mnist_train_dataset, noise_factor=0.5, noise_type='gaussian', seed=42, return_original=True)
    train_loader, val_loader, test_loader = load_dataset(train_dataset, 0.8, 0.2, 64, return_original=True)

    # 遍历数据加载器以进行训练
    for input_batch, label_batch, original_batch in train_loader:
        # 在这里进行模型的训练
        print(input_batch.shape, label_batch.shape, original_batch.shape)
        break  # 只打印一个batch的shape

    first_input_batch, first_label_batch, first_original_batch = next(iter(train_loader))
    print("noisy input:", first_input_batch[0])
    print("original input:", first_original_batch[0])
    print("label:", first_label_batch[0])

    # 可视化原始和噪声图像
    visualize_noisy_images(train_loader)
