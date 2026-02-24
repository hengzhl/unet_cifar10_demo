import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDiffusionCIFAR(nn.Module):
    def __init__(self, t_dim=256):
        super().__init__()
        self.t_dim = t_dim
        
        # 1. 时间嵌入 MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, t_dim),
            nn.ReLU(),
            nn.Linear(t_dim, t_dim)
        )

        # 2. Encoder
        self.enc1 = ResidualBlock(3, 64, t_dim)
        self.pool1 = nn.MaxPool2d(2)  # 16x16
        self.enc2 = ResidualBlock(64, 128, t_dim)
        self.pool2 = nn.MaxPool2d(2)  # 8x8

        # 3. Bottleneck
        self.bottleneck = ResidualBlock(128, 256, t_dim)

        # 4. Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = ResidualBlock(128 + 128, 128, t_dim)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ResidualBlock(64 + 64, 64, t_dim)

        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, x, t):
        # 转换 t 为向量 [Batch, 1] -> [Batch, t_dim]
        t_emb = self.time_mlp(t.view(-1, 1).float())

        # Encoder
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool1(e1), t_emb)

        # Bottleneck
        b = self.bottleneck(self.pool2(e2), t_emb)

        # Decoder + Skip Connections
        d1 = self.up1(b)
        d1 = self.dec1(torch.cat([d1, e2], dim=1), t_emb)
        
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1), t_emb)

        return self.final(d2)

# --- 辅助残差块定义 ---
# 使用ResNet思想，将输入x经过卷积块拟合函数f(x)后，再与输入x相加，得到H(x)=input+f(x)
# 卷积块拟合函数输出H(x)=input+f(x)，残差块拟合函数过程的变化f(x)
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, t_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_c) # 将时间维度映射到通道维度
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity() # 把 3 通道的输入“变宽”成 64 通道，但像素内容基本不变，方便后续相加
        
    def forward(self, x, t_emb):
        h = F.relu(self.conv1(x))
        h = h + self.t_proj(t_emb)[:, :, None, None] # 广播，注入时间信息
        h = F.relu(self.conv2(h))
        return h+self.shortcut(x)