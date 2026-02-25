import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 导入您项目中的组件
from cfg import train_cfg
from unet import UNetDiffusionCIFAR as net
from loader import test_loader, denormalize


def visualize_results(model_path, num_samples=None):

    if num_samples is None:
        num_samples = train_cfg.test_num_samples

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
    images, _ = next(iter(test_loader))  # 抓取一个batch的图像
    images = images[:num_samples]        # 只取前 num_samples 张图
    images = images.to(device)

    # --- 4. 模拟加噪与模型去噪 ---
    t_val = train_cfg.noise_steps - 1                        # 使用最大噪声步进行测试
    t = torch.full((num_samples,), t_val, device=device)     # 创建一个时间步张量，所有元素都是 t_val，长度为 num_samples
    
    noise = torch.randn_like(images)
    t_weighted = (t / train_cfg.noise_steps).view(-1, 1, 1, 1).float()
    noisy_images = images + noise * t_weighted
    
    with torch.no_grad():
        predicted_noise = model(noisy_images, t)
        denoised_images = noisy_images - predicted_noise     # 简单的一步去噪

    # --- 5. 静态对比图保存 ---
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

    plt.tight_layout()  # 自动调整子图间距，防止重叠
    plt.savefig(res_dir / f"comparison_result_{Path(model_path).stem}.png")
    print(f"静态对比图已保存至: {res_dir / f'comparison_result_{Path(model_path).stem}.png'}")


if __name__ == "__main__":
    MODEL_FILE = train_cfg.test_model_path
    if Path(MODEL_FILE).exists():
        visualize_results(MODEL_FILE)
    else:
        print(f"错误：未找到模型文件 {MODEL_FILE}，请先运行 train.py")