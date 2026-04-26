"""
混合图像模型模块.

本模块提供 PyTorch 模型类, 用于创建混合图像. 混合图像通过将一张图像的低频分量
与另一张图像的高频分量合并而成.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
import numpy as np


class HybridImageModel(nn.Module):
    """
    混合图像模型类.

    通过高斯低通滤波器从第一张图像提取低频分量, 从第二张图像提取高频分量,
    然后合并两者生成混合图像.

    Attributes:
        cutoff_frequency (float): 高斯滤波器的截止频率, 决定低频/高频分量的分割点.
        kernel_size (int): 高斯核的大小, 自动根据截止频率计算.

    Examples:
        >>> model = HybridImageModel(cutoff_frequency=10.0)
        >>> low_freq, high_freq, hybrid = model(image_a, image_b)
    """

    def __init__(self, cutoff_frequency: float) -> None:
        """
        初始化混合图像模型.

        Args:
            cutoff_frequency (float): 高斯滤波器的截止频率, 值越大保留的低频越多.
        """
        super().__init__()
        self.cutoff_frequency = cutoff_frequency
        self.kernel_size = self._compute_kernel_size(cutoff_frequency)

        # 注册缓冲区, 使高斯核成为模型状态的一部分但不是可训练参数
        self.register_buffer(
            "gaussian_kernel",
            self._create_gaussian_kernel(cutoff_frequency, self.kernel_size),
        )

    def _compute_kernel_size(self, cutoff_frequency: float) -> int:
        """
        根据截止频率计算高斯核大小.

        Args:
            cutoff_frequency (float): 截止频率.

        Returns:
            int: 高斯核大小, 确保为奇数.
        """
        return int(cutoff_frequency * 4 + 1)

    def _create_gaussian_kernel(
        self, cutoff_frequency: float, kernel_size: int
    ) -> torch.Tensor:
        """
        创建 2D 高斯核.

        Args:
            cutoff_frequency (float): 标准差 (sigma).
            kernel_size (int): 核大小.

        Returns:
            torch.Tensor: 形状为 (1, 1, kernel_size, kernel_size) 的高斯核.
        """
        # 创建坐标向量
        coords = torch.arange(kernel_size, dtype=torch.float32)
        center = kernel_size // 2

        # 计算 1D 高斯函数
        gaussian_1d = torch.exp(
            -((coords - center) ** 2) / (2 * cutoff_frequency ** 2)
        )
        gaussian_1d = gaussian_1d / (cutoff_frequency * np.sqrt(2 * np.pi))

        # 通过外积创建 2D 高斯核
        kernel_2d = torch.outer(gaussian_1d, gaussian_1d)

        # 归一化使和为 1
        kernel_2d = kernel_2d / kernel_2d.sum()

        # 调整为卷积所需的形状: (out_channels, in_channels, H, W)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

        return kernel_2d

    def _apply_filter(
        self, image: torch.Tensor, kernel: torch.Tensor, padding: int
    ) -> torch.Tensor:
        """
        对图像应用高斯滤波器.

        使用深度可分离卷积, 对每个通道独立应用相同的滤波器.

        Args:
            image (torch.Tensor): 输入图像, 形状为 (B, C, H, W) 或 (C, H, W).
            kernel (torch.Tensor): 高斯核, 形状为 (1, 1, k, k).
            padding (int): 填充大小.

        Returns:
            torch.Tensor: 滤波后的图像.
        """
        # 确保输入为 4D: (B, C, H, W)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        channels = image.shape[1]

        # 扩展卷积核以匹配通道数: (channels, 1, k, k)
        kernel_expanded = kernel.expand(channels, -1, -1, -1)

        # 使用深度卷积, 每个通道独立应用相同的滤波器
        filtered = F.conv2d(
            image,
            kernel_expanded,
            padding=padding,
            groups=channels,
        )

        return filtered

    def forward(
        self,
        image_a: torch.Tensor | NDArray[np.floating],
        image_b: torch.Tensor | NDArray[np.floating],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播: 创建混合图像.

        Args:
            image_a (torch.Tensor | NDArray[np.floating]): 第一张图像, 用于提取低频分量.
                                                          形状为 (C, H, W) 或 (B, C, H, W).
            image_b (torch.Tensor | NDArray[np.floating]): 第二张图像, 用于提取高频分量.
                                                          形状为 (C, H, W) 或 (B, C, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 包含三个元素的元组:
                - 低频分量 (low_frequencies): 来自 image_a, 形状与输入相同.
                - 高频分量 (high_frequencies): 来自 image_b, 形状与输入相同.
                - 混合图像 (hybrid_image): 低频 + 高频, 形状与输入相同.

        Raises:
            ValueError: 当输入图像形状不匹配时抛出异常.
        """
        # 转换 NumPy 数组为 PyTorch 张量
        if isinstance(image_a, np.ndarray):
            image_a = torch.from_numpy(image_a).permute(2, 0, 1).unsqueeze(0)
        if isinstance(image_b, np.ndarray):
            image_b = torch.from_numpy(image_b).permute(2, 0, 1).unsqueeze(0)

        # 确保输入为 4D: (B, C, H, W)
        if image_a.dim() == 3:
            image_a = image_a.unsqueeze(0)
        if image_b.dim() == 3:
            image_b = image_b.unsqueeze(0)

        # 验证形状匹配
        if image_a.shape != image_b.shape:
            raise ValueError(
                f"\033[1;91m输入图像形状不匹配: {image_a.shape} vs {image_b.shape}\033[0m"
            )

        # 计算填充量
        kernel_size = self.gaussian_kernel.shape[-1]
        padding = kernel_size // 2

        # 提取低频分量
        low_frequencies = self._apply_filter(image_a, self.gaussian_kernel, padding)

        # 提取高频分量: image_b 的低频被移除
        image_b_low = self._apply_filter(image_b, self.gaussian_kernel, padding)
        high_frequencies = image_b - image_b_low

        # 合成混合图像
        hybrid_image = low_frequencies + high_frequencies

        # 裁剪到 [0, 1] 范围
        hybrid_image = torch.clamp(hybrid_image, 0.0, 1.0)

        return low_frequencies, high_frequencies, hybrid_image

    def update_cutoff_frequency(self, cutoff_frequency: float) -> None:
        """
        更新截止频率并重新计算高斯核.

        Args:
            cutoff_frequency (float): 新的截止频率.
        """
        self.cutoff_frequency = cutoff_frequency
        self.kernel_size = self._compute_kernel_size(cutoff_frequency)
        self.gaussian_kernel = self._create_gaussian_kernel(
            cutoff_frequency, self.kernel_size
        )
