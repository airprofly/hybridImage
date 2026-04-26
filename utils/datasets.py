"""
混合图像数据集模块.

本模块提供 PyTorch Dataset 类, 用于加载和管理混合图像实验所需的图像对数据.
"""

from pathlib import Path
from typing import cast

import numpy as np
from torch.utils import data
from numpy.typing import NDArray

from configs.config import AppConfig
from utils.utils import load_image


class HybridImageDataset(data.Dataset):
    """
    混合图像数据集类.

    用于加载和管理混合图像实验所需的图像对数据. 每个样本包含一对图像及其对应的截止频率.

    Attributes:
        image_pairs (list[ImagePairConfig]): 图像对配置列表, 从配置文件加载.
        best_cutoffs (list[float]): 每个图像对对应的最佳截止频率列表.

    Examples:
        >>> from configs.config import APP_CONFIG
        >>> dataset = HybridImageDataset(APP_CONFIG)
        >>> image_a, image_b, cutoff = dataset[0]
    """

    def __init__(self, config: AppConfig) -> None:
        """
        初始化混合图像数据集.

        Args:
            config (AppConfig): 应用程序配置对象, 包含路径和混合参数配置.

        Raises:
            ValueError: 当图像对数量与截止频率数量不匹配时抛出异常.
        """
        print(f"\n\033[1;96m[START] 初始化混合图像数据集...\033[0m\n")

        self.image_pairs = config.paths.image_pairs
        self.best_cutoffs = config.hybrid.best_cutoffs

        # 验证图像对数量与截止频率数量是否匹配
        if len(self.image_pairs) != len(self.best_cutoffs):
            raise ValueError(
                f"\033[1;91m图像对数量 ({len(self.image_pairs)}) 与截止频率数量 "
                f"({len(self.best_cutoffs)}) 不匹配\033[0m"
            )

        print(f"图像对数量: {len(self.image_pairs)}")
        print(f"截止频率数量: {len(self.best_cutoffs)}\n")
        print(f"\033[1;92m[SUCCESS] 数据集初始化完成!\033[0m\n")

    def __len__(self) -> int:
        """
        获取数据集样本数量.

        Returns:
            int: 数据集中图像对的总数量.
        """
        return len(self.image_pairs)

    def __getitem__(
        self, index: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
        """
        获取指定索引的图像对和对应的截止频率.

        Args:
            index (int): 样本索引, 范围为 [0, len(self)).

        Returns:
            tuple[NDArray[np.float32], NDArray[np.float32], float]: 包含三个元素的元组:
                - 第一张图像 (image_a), 形状为 (H, W, C), 值域 [0, 1].
                - 第二张图像 (image_b), 形状为 (H, W, C), 值域 [0, 1].
                - 截止频率 (cutoff_frequency), 浮点数.

        Raises:
            IndexError: 当索引超出范围时抛出异常.
            FileNotFoundError: 当图像文件不存在时抛出异常.
        """
        if index >= len(self):
            raise IndexError(
                f"\033[1;91m索引 {index} 超出范围, 数据集大小为 {len(self)}\033[0m"
            )

        pair = self.image_pairs[index]
        cutoff_frequency = self.best_cutoffs[index]

        # 类型断言为 Path
        image_a_path = cast(Path, pair.image_a)
        image_b_path = cast(Path, pair.image_b)

        # 验证文件是否存在
        if not image_a_path.exists():
            raise FileNotFoundError(
                f"\033[1;91m图像文件不存在: {image_a_path}\033[0m"
            )
        if not image_b_path.exists():
            raise FileNotFoundError(
                f"\033[1;91m图像文件不存在: {image_b_path}\033[0m"
            )

        # 加载图像
        image_a = load_image(image_a_path)
        image_b = load_image(image_b_path)

        return image_a, image_b, cutoff_frequency

    def get_pair_info(self, index: int) -> dict[str, str | float]:
        """
        获取指定索引图像对的元信息.

        Args:
            index (int): 样本索引.

        Returns:
            dict[str, str | float]: 包含图像对元信息的字典:
                - "image_a_path": 第一张图像的路径.
                - "image_b_path": 第二张图像的路径.
                - "cutoff_frequency": 截止频率.

        Raises:
            IndexError: 当索引超出范围时抛出异常.
        """
        if index >= len(self):
            raise IndexError(
                f"\033[1;91m索引 {index} 超出范围, 数据集大小为 {len(self)}\033[0m"
            )

        pair = self.image_pairs[index]

        # 类型断言为 Path
        image_a_path = cast(Path, pair.image_a)
        image_b_path = cast(Path, pair.image_b)

        return {
            "image_a_path": str(image_a_path),
            "image_b_path": str(image_b_path),
            "cutoff_frequency": self.best_cutoffs[index],
        }
