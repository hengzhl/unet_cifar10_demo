import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

from cfg import train_cfg
from unet import UNetDiffusionCIFAR as net
from loader import train_loader, denormalize

def overfit_test(model, train_loader):
    """
    针对单张图片进行噪声预测的过拟合测试
    """
    model.train()

    res_dir = Path(train_cfg.save_path) / "res"
    res_dir.mkdir(parents=True, exist_ok=True) 
    device = train_cfg.device
    
    # 取一个 Batch 中的第一张图片
    inputs, _ = next(iter(train_loader))
    inputs = inputs[0:1].to(device)      # 形状 [1, 3, 32, 32]
    
    t = torch.tensor([5]).to(device)     # 不妨选取一个中间的时间步进行测试，t=5（假设 noise_steps=10）
    
    noise = torch.randn_like(inputs).to(device)
    t_weighted = (t / train_cfg.noise_steps).view(-1, 1, 1, 1).float()
    noisy_inputs = inputs + noise * t_weighted
    
    target = noise 

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"开始过拟合测试 (针对单张图片预测 t={t.item()} 时的噪声)...")
    
    for epoch in range(1000): 
        optimizer.zero_grad()
    
        predicted_noise = model(noisy_inputs, t)
        
        loss = criterion(predicted_noise, target)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/1000], Loss: {loss.item():.8f}")

    # 可视化对比
    model.eval()
    with torch.no_grad():
        pred_noise = model(noisy_inputs, t)
        reconstructed = noisy_inputs - pred_noise * t_weighted
    
    def to_img(tensor):
        img = denormalize(tensor)
        img = img.cpu().squeeze().permute(1, 2, 0) # [H,W,C]
        return img.numpy()

    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Original (Ground Truth)", f"Noisy Input (t={t.item()})", "Reconstructed"]

    axes[0].imshow(to_img(inputs))
    axes[0].set_title(titles[0])

    axes[1].imshow(to_img(noisy_inputs))
    axes[1].set_title(titles[1])

    axes[2].imshow(to_img(reconstructed))
    axes[2].set_title(titles[2])

    plt.tight_layout()
    plt.savefig(res_dir / "comparison_result_overfit.png")
    print(f"静态对比图已保存至: {res_dir / 'comparison_result_overfit.png'}")

    plt.show()


if __name__ == "__main__":
    model = net(t_dim=train_cfg.t_dim).to(train_cfg.device)
    overfit_test(model, train_loader)