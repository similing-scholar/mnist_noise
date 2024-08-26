import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


class Basic2DDataset(Dataset):
    def __init__(self, mnist_dataset, encoder='train', noise_levels=None, noise_type='gaussian', seed=0, return_original=False):
        self.encoder = encoder
        self.data = mnist_dataset
        self.noise_levels = noise_levels if noise_levels else [0.1]
        self.noise_type = noise_type
        self.seed = seed
        self.return_original = return_original

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        np.random.seed(self.seed + index)
        noise_level = self.noise_levels[index % len(self.noise_levels)]

        if self.noise_type == 'gaussian':
            noise = torch.tensor(np.random.normal(0, noise_level, image.size()), dtype=torch.float32)
            noisy_image = image + noise
        elif self.noise_type == 'percent gaussian':
            noise = torch.tensor(np.random.normal(0, 1, image.size()), dtype=torch.float32)
            noisy_image = image + noise_level * noise
        elif self.noise_type == 'striped':
            noise = torch.tensor(self._generate_striped_noise(image.size(), noise_level), dtype=torch.float32)
            noisy_image = image + noise
        elif self.noise_type == 'mosaic':
            noise = torch.tensor(self._generate_mosaic_noise(image.size(), noise_level), dtype=torch.float32)
            noisy_image = image + noise
        else:
            raise ValueError("Unsupported noise type")

        noisy_image = torch.clamp(noisy_image, 0, 1)

        if self.return_original:
            return noisy_image, label, image
        else:
            if self.encoder == 'train':
                return noisy_image, label
            elif self.encoder == 'predict':
                return noisy_image

    def _generate_striped_noise(self, size, noise_level):
        noise = np.zeros(size)
        stripe_width = 2
        for i in range(0, size[1], stripe_width * 2):
            noise[:, i:i+stripe_width] = noise_level
        return noise

    def _generate_mosaic_noise(self, size, noise_level):
        noise = np.random.rand(*size)
        mosaic_size = 4
        for i in range(0, size[0], mosaic_size):
            for j in range(0, size[1], mosaic_size):
                block_value = np.random.normal(0, noise_level)
                noise[i:i+mosaic_size, j:j+mosaic_size] = block_value
        return noise

def load_dataset(mnist_dataset, train_ratio, val_ratio, batch_size, return_original=False, seed=0):
    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio 必须小于等于 1.0")

    total_samples = len(mnist_dataset)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(mnist_dataset,
                                                            [train_size, val_size, test_size],
                                                            generator=generator)

    train_dataset.dataset.return_original = return_original
    val_dataset.dataset.return_original = return_original

    loader_args = dict(batch_size=batch_size, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader

def visualize_noisy_images(data_loader, digit, num_images=5):
    filtered_images = []
    filtered_labels = []
    filtered_originals = []

    for noisy_image, label, original_image in data_loader:
        if label[0].item() == digit:
            filtered_images.append(noisy_image[0])
            filtered_labels.append(label[0])
            filtered_originals.append(original_image[0])
        if len(filtered_images) >= num_images:
            break

    fig, axs = plt.subplots(num_images, 2, figsize=(10, 2 * num_images))

    for i in range(num_images):
        noisy_image = filtered_images[i].squeeze().numpy()
        original_image = filtered_originals[i].squeeze().numpy()
        label = filtered_labels[i].item()

        axs[i, 0].imshow(original_image, cmap='gray')
        axs[i, 0].set_title(f'Original Image - Label: {label}')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(noisy_image, cmap='gray')
        axs[i, 1].set_title('Noisy Image')
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    mnist_train_dataset = datasets.MNIST(root='../dataset', train=True, download=False, transform=transform)
    mnist_test_dataset = datasets.MNIST(root='../dataset', train=False, download=False, transform=transform)

    noise_factor = 1
    if noise_factor == 'mix':
        noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    else:
        noise_levels = [noise_factor]

    train_dataset = Basic2DDataset(mnist_train_dataset, noise_levels=noise_levels, noise_type='gaussian', seed=42, return_original=True)
    train_loader, val_loader, test_loader = load_dataset(train_dataset, 0.8, 0.2, 64, return_original=True, seed=42)

    for input_batch, label_batch, original_batch in train_loader:
        print(input_batch.shape, label_batch.shape, original_batch.shape)
        break

    first_input_batch, first_label_batch, first_original_batch = next(iter(train_loader))
    print("noisy input:", first_input_batch[0])
    print("original input:", first_original_batch[0])
    print("label:", first_label_batch[0])

    digit_to_visualize = 9  # 选择要显示的数字
    visualize_noisy_images(train_loader, digit_to_visualize)
