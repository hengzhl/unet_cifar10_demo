import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cfg import train_cfg

def get_dataloader(train=True):
    # 1. 定义数据预处理
    # ToTensor: [0, 255] -> [0, 1]
    # Normalize: [0, 1] -> [-1, 1] 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 2. 加载数据集
    dataset = datasets.CIFAR10(
        root='./data', 
        train=train, 
        download=True, 
        transform=transform
    )

    # 3. 构建 DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=train_cfg.batch_size, 
        shuffle=True, 
        num_workers=train_cfg.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return dataloader

def denormalize(tensor):   # 将 [-1, 1] 的 Tensor 转回 [0, 1]，用于可视化
    return (tensor * 0.5 + 0.5).clamp(0, 1)


train_loader = get_dataloader(train=True)
test_loader = get_dataloader(train=False)
