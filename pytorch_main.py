"""
PyTorch 混合图像处理主程序.

本程序使用 HybridImageDataset 和 HybridImageModel 处理图像对, 生成混合图像并保存结果.
"""

from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import torch
from torch.utils import data

from configs.config import APP_CONFIG
from utils.datasets import HybridImageDataset
from utils.model import HybridImageModel
from utils.utils import save_image, show_image


def main() -> None:
    """
    执行混合图像处理流程.

    流程:
        1. 加载数据集和创建 DataLoader.
        2. 遍历每对图像, 更新模型截止频率.
        3. 生成混合图像并保存到 outputs/hybrid_pytorch/pair_N/.
        4. 为每对图像生成可视化结果.
    """
    # 初始化数据集
    dataset = HybridImageDataset(APP_CONFIG)

    # 创建 DataLoader
    dataloader = data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    # 输出目录 - PyTorch 版本
    output_dir = Path(APP_CONFIG.paths.output_dir).parent / "hybrid_pytorch"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建模型（使用第一个截止频率初始化）
    model = HybridImageModel(cutoff_frequency=APP_CONFIG.hybrid.best_cutoffs[0])
    model.eval()

    # 处理每对图像
    for batch_idx, (image_a, image_b, cutoff_frequency) in enumerate(dataloader):
        # DataLoader 自动将 NumPy 转为 Tensor，形状为 (B, H, W, C)，需转为 (B, C, H, W)
        image_a_tensor: torch.Tensor = image_a.permute(0, 3, 1, 2).float()
        image_b_tensor: torch.Tensor = image_b.permute(0, 3, 1, 2).float()

        # 获取截止频率值并更新模型
        cutoff = cutoff_frequency.item()
        model.update_cutoff_frequency(cutoff)

        with torch.no_grad():
            low_freq, high_freq, hybrid_image = model(image_a_tensor, image_b_tensor)

        # 转换回 NumPy: (H, W, C)
        low_freq_np: np.ndarray = low_freq.squeeze(0).permute(1, 2, 0).numpy()
        high_freq_np: np.ndarray = high_freq.squeeze(0).permute(1, 2, 0).numpy()
        hybrid_np: np.ndarray = hybrid_image.squeeze(0).permute(1, 2, 0).numpy()

        # 创建每对图像的子目录
        pair_dir = output_dir / f"pair_{batch_idx + 1}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        # 保存结果
        save_image(low_freq_np, pair_dir / "low_frequencies.jpg")
        save_image(high_freq_np + 0.5, pair_dir / "high_frequencies.jpg")
        save_image(hybrid_np, pair_dir / "hybrid_image.jpg")

        # 获取图像路径信息
        pair_info = dataset.get_pair_info(batch_idx)
        image_a_name = str(pair_info["image_a_path"]).split("/")[-1]
        image_b_name = str(pair_info["image_b_path"]).split("/")[-1]

        print(f"图像对 {batch_idx + 1} 已保存至 {pair_dir}")

        # 可视化
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        show_image(ax[0], low_freq_np, "Low Frequencies")
        show_image(ax[1], high_freq_np + 0.5, "High Frequencies (shifted)")
        show_image(ax[2], hybrid_np, "Hybrid Image")
        fig.suptitle(
            f"Image Pair {batch_idx + 1}: {image_a_name} & {image_b_name} (cutoff={cutoff})"
        )
        fig.savefig(pair_dir / "results.png")
        plt.close(fig)


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
