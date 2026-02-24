import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import torch.optim.lr_scheduler as scheduler
from tqdm import tqdm

from cfg import train_cfg  
from unet import UNetDiffusionCIFAR as net

def train():
    save_dir = Path(train_cfg.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. 数据准备 ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 归一化到 [-1, 1]
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers)

    # --- 2. 初始化 ---
    model = net(t_dim = train_cfg.t_dim).to(train_cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr)
    # 引入余弦退火调度器：从 cfg.lr 开始，在指定 Epoch 内降至接近 0
    lr_scheduler = scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.epochs)
    criterion = nn.MSELoss()

    # --- 断点续训加载逻辑 ---
    start_epoch = 0
    resume_checkpoint = Path(train_cfg.resume_path) if train_cfg.resume_path else None
    
    if resume_checkpoint and resume_checkpoint.exists():
        print(f"正在从断点恢复: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=train_cfg.device)
        
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # 恢复学习率调度器的进度
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        start_epoch = checkpoint['epoch'] + 1  
        print(f"恢复成功！将从第 {start_epoch + 1} 个 Epoch 继续训练。")

    print(f"训练开始，使用设备: {train_cfg.device}")

    # --- 3. 训练循环 ---
    for epoch in range(start_epoch,train_cfg.epochs):
        model.train()
        pbar = tqdm(dataloader)
        pbar.set_description(f"Epoch {epoch+1}/{train_cfg.epochs}")

        total_loss = 0
        
        for images, _ in pbar:
            images = images.to(train_cfg.device)
            
            # A. 采样时间步 t: 每个 batch 随机生成不同的 t
            t = torch.randint(low=1, high=train_cfg.noise_steps, size=(images.shape[0],)).to(train_cfg.device)
            
            # B. 注入高斯噪声: 模拟 Diffusion 过程
            # 这里的 noise_level 简化为随 t 线性增加
            noise = torch.randn_like(images)
            # 简单的加噪公式: noisy_img = images + noise * (t/T)
            # 注意：实际 Diffusion 使用 alpha/beta 调度，这里为简化演示精髓
            t_weighted = (t / train_cfg.noise_steps).view(-1, 1, 1, 1).float()
            noisy_images = images + noise * t_weighted
            
            # C. 前向预测噪声
            predicted_noise = model(noisy_images, t)
            
            # D. 计算损失
            loss = criterion(predicted_noise, noise)
            
            # E. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 在进度条实时显示 Loss 和 当前学习率
            loss = criterion(predicted_noise, noise)
            total_loss += loss.item() # 累加 Loss
            avg_loss = total_loss / (pbar.n + 1)# 实时显示平均 Loss (当前已处理 Batch 的平均值)
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(Avg_Loss=f"{avg_loss:.4f}", LR=f"{current_lr:.2e}")

        lr_scheduler.step() # 更新学习率

        # 保存模型
        if (epoch + 1) % train_cfg.save_model_epochs == 0 or epoch == train_cfg.epochs - 1:
            save_name = f"unet{train_cfg.img_size}_e{epoch}.pth"
            save_full_path = save_dir / save_name

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, save_full_path)
            print(f"模型已保存: {save_full_path}")


if __name__ == "__main__":
    train()