U-Net论文：[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

项目代码仓：[hengzhl/diffusion_demos/unet_cifar10_demo](https://github.com/hengzhl/diffusion_demos/tree/main/unet_cifar10_demo)

```
git clone https://github.com/hengzhl/unet_cifar10_demo.git
```

>U-Net 是一个卷积神经网络，通常用于**图 $\rightarrow$ 图（同尺寸）的任务**，具有语义分割和**空间定位**能力。它通过**对称的 U 型结构和跳跃连接**，融合了图片的抽象语义和几何细节。

### 一、Project Overview

本仓库实现了一个基于 **U-Net** 架构的卷积神经网络，用于对 **CIFAR-10** 数据集进行去噪处理。本项目模拟了扩散模型（Diffusion Model）中的核心去噪步骤：即通过神经网络预测并消除图像中的高斯噪声。

### 二、Model Architecture

本项目实现了一个**时间感知型 U-Net (Time-conditioned U-Net)**，其核心在于将扩散模型的时间步 $t$ 注入到每一个残差层中。
        
- **编码器与解码器 (Encoder & Decoder):**
    
    - **Encoder:** 包含两层残差块，配合 `MaxPool2d` 进行下采样，空间维度从 $32 \times 32$ 降至 $8 \times 8$
        
    - **Bottleneck:** 中间层使用 $256$ 通道的残差块提取高维特征。
        
    - **Decoder:** 使用 `ConvTranspose2d` 进行转置卷积上采样，并通过 `torch.cat` 实现**跳跃连接 (Skip Connections)**，将编码器的低级特征与解码器的高级特征融合。

- **时间嵌入 MLP (Time Embedding):** * 将标量时间步 $t$ 通过两层全连接网络（`nn.Linear`）映射为 $256$ 维的向量 `t_emb`。这使得模型能够根据噪声强度的不同调整其去噪行为。

- **残差块 (Residual Block):** * **时间注入：** 通过 `t_proj` 将时间向量映射到特征图的通道维度，并利用广播机制（Broadcasting）进行元素级相加。
    - **跳跃连接：** 内部使用 `shortcut`（`nn.Identity` 或 $1\times 1$ 卷积）确保输入与输出形状匹配，实现残差学习。

### 三、IPO Pipeline

该流程模拟了扩散模型的前向加噪与反向预测逻辑：

| **阶段**           | **详细内容**                                                                                                                |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Input (输入)**   | **图像:** $3 \times 32 \times 32$ 的 CIFAR-10 图像（归一化至 $[-1, 1]$）。<br><br>  <br><br>**时间步:** 随机生成的整数 $t \in [1, 10)$。       |
| **Process (处理)** | <br>**简单加噪:** 根据公式 $x_t = x_0 + \epsilon \cdot \frac{t}{T}$ 生成噪声图像。<br>  <br>**特征提取:** U-Net 接收噪声图像和时间嵌入，进行多尺度卷积运算。<br> |
| **Output (输出)**  | **预测噪声:** 模型输出一个 $3 \times 32 \times 32$ 的张量，即模型认为图像中被添加的噪声 $\hat{\epsilon}$                                            |

### 四、Training Strategy

训练过程遵循回归任务的逻辑，目标是使预测噪声尽可能接近真实添加的噪声。

- **损失函数:** 使用 **均方误差 (MSE Loss)**，计算 `predicted_noise` 与真实 `noise` 之间的差异。
    
- **优化算法:** 选用 `AdamW` 优化器，学习率为 $2 \times 10^{-4}$。
    
- **采样逻辑:**  在每个 batch 中，为每张图片随机采样不同的时间步 $t$。

    - 这种策略确保模型在一次迭代中能学习到不同受损程度下的去噪能力。
        
- **持久化存储:**  每隔 `save_model_epochs` 保存一次模型权重。
    
    - 保存内容包括模型状态字典（`state_dict`）、优化器状态以及当前 Epoch 数，支持断点续训。
        
### 七、Quick Start

```bash
# 克隆仓库
git clone https://github.com/your-username/unet-denoising.git
# 运行训练脚本
python train.py 
```

### 八、Results

先进行过拟合测试（`overfit.py`），来证明模型框架具备去噪能力：

![overfit](https://raw.githubusercontent.com/hengzhl/img-bed/main/img/comparison_result_overfit.png)

然后使用训练集训练模型（`train.py`），

```python
......
Epoch 18/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 6250/6250 [17:53<00:00,  5.82it/s, Avg_Loss=0.0825, LR=5.46e-05] 
模型已保存: model\unet32_e17.pth

Epoch 19/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 6250/6250 [17:51<00:00,  5.83it/s, Avg_Loss=0.0825, LR=4.12e-05] 
模型已保存: model\unet32_e18.pth

Epoch 20/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 6250/6250 [18:04<00:00,  5.76it/s, Avg_Loss=0.0820, LR=2.93e-05]
模型已保存: model\unet32_e19.pth
```

![对比图](https://raw.githubusercontent.com/hengzhl/img-bed/main/img/comparison_result.png)

训练效果不理想，一方面是训练轮数太少，另一方面对于多步加噪数据，仅采用简单的一步去噪；故该项目仅作为熟悉`unet`的学习性项目。

### 附录：Analysis and Design

`CIFAR-10` 的图片尺寸是 $32 \times 32 \times 3$（RGB），属于微缩版的图像处理。要设计这个网络，我们需要遵循“对称、压缩、连接、扩张”的原则。由于图片很小（$32 \times 32$），我们不需要像处理 $512 \times 512$医疗影像那样设计太深，通常 **3 层下采样** 就足够了。

IPO 流程：

- **Input**:  $32 \times 32 \times 3$ 的加噪图像。像素值已被高斯噪声污染。
    
- **Process**:
    
    - **Level 1**: $32 \times 32 \rightarrow$ 卷积提取特征 ($64$ 通道) $\rightarrow$ 池化降维至 $16 \times 16$。
        
    - **Level 2**: $16 \times 16 \rightarrow$ 卷积提取特征 ($128$ 通道) $\rightarrow$ 池化降维至 $8 \times 8$。
        
    - **Bottleneck (瓶颈层)**: 此时图片只有 $8 \times 8$，这是**语义最浓缩**的地方。
        
    - **Level 2 Up**: 上采样回 $16 \times 16 +$ **融合 Level 2 的原始特征**。
        
    - **Level 1 Up**: 上采样回 $32 \times 32 +$ **融合 Level 1 的原始特征**。
        
- **Output**:  $32 \times 32 \times 3$ 的噪声预测图。

| **      阶段        ** | **模块名称**   | **输入尺寸**      | **输出尺寸**      | **      核心参数设计 (PyTorch 风格)            ** |
| -------------------- | ---------- | ------------- | ------------- | ----------------------------------------- |
| **Input**            | 原始输入       | $32, 32, 3$   | $32, 32, 3$   | -                                         |
| **Encoder 1**        | Level 1    | $32, 32, 3$   | $16, 16, 64$  | `Conv2d(3, 64)`, `MaxPool2d(2)`           |
| **Encoder 2**        | Level 2    | $16, 16, 64$  | $8, 8, 128$   | `Conv2d(64, 128)`, `MaxPool2d(2)`         |
| **Middle**           | Bottleneck | $8, 8, 128$   | $8, 8, 256$   | `Conv2d(128, 256)` (语义最浓缩层)               |
| **Decoder 1**        | Level 2 Up | $8, 8, 256$   | $16, 16, 128$ | `Upsample`, `Cat` (与 Down 2 拼接), `Conv2d` |
| **Decoder 2**        | Level 1 Up | $16, 16, 128$ | $32, 32, 64$  | `Upsample`, `Cat` (与 Down 1 拼接), `Conv2d` |
| **Output**           | Final      | $32, 32, 64$  | $32, 32, 3$   | `Conv2d(64, 3, 1)` (还原通道)                 |

综上所述，unet.py设计如下：

```python
import torch
import torch.nn as nn

class UNetCIFAR(nn.Module):
    def __init__(self):
        super(UNetCIFAR, self).__init__()

        # --- Encoder 部分 ---
        # 第一层：32x32 -> 16x16
        self.enc1 = self._block(3, 64)             # (N, 64, 32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)   # (N, 64, 16, 16)
        
        # 第二层：16x16 -> 8x8
        self.enc2 = self._block(64, 128)           # (N, 128, 16, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=2)   # (N, 128, 8, 8)

        # --- Bottleneck ---
        self.bottleneck = self._block(128, 256)    # (N, 256, 8, 8)

        # --- Decoder 部分 ---
        # 第一层上采样：8x8 -> 16x16 
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 注意：拼接后输入通道是 256(来自Up) + 128(来自Encoder2) = 384
        self.dec1 = self._block(256 + 128, 128)
        
        # 第二层上采样：16x16 -> 32x32
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 拼接后输入通道是 128 + 64 = 192
        self.dec2 = self._block(128 + 64, 64)

        # 最终映射层：保持尺寸不变，通道变回 3 (RGB)
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def _block(self, in_channels, out_channels):
        """基础卷积块：卷积 + BN + 激活"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):                   # (N, 3, 32, 32)
        # Encoder
        e1 = self.enc1(x)                   # (N, 64, 32, 32)
        e2 = self.enc2(self.pool1(e1))      # (N, 128, 16, 16)
        
        # Middle
        b = self.bottleneck(self.pool2(e2)) # (N, 256, 8, 8)
        
        # Decoder + Skip Connections
        d1 = self.up1(b)                    # (N, 256, 16, 16)
        d1 = torch.cat([d1, e2], dim=1)     # (N, 384, 16, 16)
        d1 = self.dec1(d1)                  # (N, 128, 16, 16)
        
        d2 = self.up2(d1)                   # (N, 128, 32, 32)
        d2 = torch.cat([d2, e1], dim=1)     # (N, 192, 32, 32)
        d2 = self.dec2(d2)                  # (N, 64, 32, 32)
        
        return self.final(d2)               # (N, 3, 32, 32)

# 测试模型
if __name__ == "__main__":
    model = UNetCIFAR()
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(f"输入形状: {x.shape} -> 输出形状: {output.shape}")
```


**如果加入时间步 t 表示噪声的强度**（比如为CIRFAR10加入 t 次高斯噪声得到含噪图），输入（Input）多了一个 t ，过程处理中将标量 $t$ 通过正弦编码或 MLP 转化为一个高维向量，在 U-Net 的每一层或 Bottleneck 处，将这个时间向量“注入”到卷积提取的特征图中。

在 PyTorch 中，我们通常使用 `Linear` 层将时间向量映射到与当前特征图相同的通道数，然后通过**相加 (Addition)** 的方式融合。

我们需要一个模块把数字 $t$ 变成向量：

```python
class TimeEmbedding(nn.Module):
    def __init__(self, t_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(t_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, t):
        # t 形状: (Batch, t_dim) -> (Batch, out_dim)
        return self.mlp(t)
```

修改卷积块，现在不仅接收图像特征 `x`，还接收时间特征 `t_emb`：

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        # 时间映射层：将时间向量映射到当前通道数
        self.time_mlp = nn.Linear(t_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t_emb):
        h = self.relu(self.conv1(x))
        # 核心步骤：将时间嵌入转换为 (Batch, Channels, 1, 1) 并加到特征图上
        time_gate = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_gate 
        h = self.relu(self.conv2(h))
        return h
```

最后给出`unet.py`代码：

```python
import torch
import torch.nn as nn

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
```

