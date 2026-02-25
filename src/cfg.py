from dataclasses import dataclass
import torch

@dataclass
class Config:
    img_size = 32
    t_dim=256
    noise_steps = 10       
    noise_level = 0.2
    num_workers = 0
    batch_size = 8
    epochs = 20
    lr = 2e-4     # loss收敛慢，调高；loss爆炸，调低
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_model_epochs = 1
    save_path = "./model"
    resume_path = "./model/unet32_e99.pth"  # 设置为你想要恢复的检查点路径，设为 None 则从头开始训练

    test_model_path = "./model/unet32_e99.pth"  # 待检测的模型路径
    test_num_samples = 5  # 可视化的样本数量

train_cfg = Config()