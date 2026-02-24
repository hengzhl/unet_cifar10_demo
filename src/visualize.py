import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

# 导入您项目中的组件
from cfg import train_cfg
from unet import UNetDiffusionCIFAR as net

def visualize_results(model_path, num_samples=5):
    # --- 1. 环境准备 ---
    res_dir = Path(train_cfg.save_path) / "res"
    res_dir.mkdir(parents=True, exist_ok=True)
    device = train_cfg.device

    # --- 2. 加载模型 ---
    model = net(t_dim=train_cfg.t_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"成功加载模型: {model_path}")

    # --- 3. 准备测试数据 ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=num_samples, shuffle=True)
    
    # 获取一批随机样本
    images, _ = next(iter(test_loader))
    images = images.to(device)

    # --- 4. 模拟加噪与模型去噪 ---
    # 我们测试不同的时间步 t，观察去噪效果
    t_val = train_cfg.noise_steps - 1 # 使用最大噪声步进行测试
    t = torch.full((num_samples,), t_val, device=device)
    
    # 生成噪声并加噪 (复现 train.py 中的逻辑)
    noise = torch.randn_like(images)
    t_weighted = (t / train_cfg.noise_steps).view(-1, 1, 1, 1).float()
    noisy_images = images + noise * t_weighted
    
    # 模型预测噪声并减去噪声实现去噪
    with torch.no_grad():
        predicted_noise = model(noisy_images, t)
        # 简单去噪公式：还原图 = 噪声图 - 预测的噪声
        denoised_images = noisy_images - predicted_noise

    # --- 5. 静态对比图保存 ---
    def denormalize(img):
        return img * 0.5 + 0.5 # 还原到 [0, 1] 范围

    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))
    titles = ["Original", f"Noisy (t={t_val})", "Denoised"]
    
    for i in range(num_samples):
        # 原图
        axes[0, i].imshow(denormalize(images[i]).cpu().permute(1, 2, 0).numpy().clip(0, 1))
        # 加噪图
        axes[1, i].imshow(denormalize(noisy_images[i]).cpu().permute(1, 2, 0).numpy().clip(0, 1))
        # 去噪图
        axes[2, i].imshow(denormalize(denoised_images[i]).cpu().permute(1, 2, 0).numpy().clip(0, 1))
        
        if i == 0:
            for r in range(3): axes[r, 0].set_ylabel(titles[r], fontsize=12)

    plt.tight_layout()
    plt.savefig(res_dir / f"comparison_result_{Path(model_path).stem}.png")
    print(f"静态对比图已保存至: {res_dir / f'comparison_result_{Path(model_path).stem}.png'}")




if __name__ == "__main__":
    # 请根据您的实际文件名修改此处的模型路径
    MODEL_FILE = "./model/unet32_e19.pth" 
    if Path(MODEL_FILE).exists():
        visualize_results(MODEL_FILE)
    else:
        print(f"错误：未找到模型文件 {MODEL_FILE}，请先运行 train.py")