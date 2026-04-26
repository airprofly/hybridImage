"""
时间比较脚本 - 比较 NumPy 和 PyTorch 两种实现的性能
"""

import time
from dotenv import load_dotenv
import numpy as np
import torch
from torch.utils import data

from configs.config import APP_CONFIG
from utils.datasets import HybridImageDataset
from utils.model import HybridImageModel
from utils.utils import create_Gaussian_kernel, create_hybrid_image, load_image


def time_numpy_implementation(rounds: int = 3) -> dict:
    """
    测试 NumPy 实现的执行时间.

    Args:
        rounds: 测试轮数

    Returns:
        包含平均时间、各轮时间的字典
    """
    image_pairs = APP_CONFIG.paths.image_pairs
    times_per_round: list[float] = []

    for round_idx in range(rounds):
        start_time = time.time()

        for i, pair in enumerate(image_pairs):
            image_a = load_image(pair.image_a)
            image_b = load_image(pair.image_b)
            cutoff_frequency = APP_CONFIG.hybrid.best_cutoffs[i]

            filter_kernel = create_Gaussian_kernel(cutoff_frequency)
            _, _, _ = create_hybrid_image(image_a, image_b, filter_kernel)

        end_time = time.time()
        elapsed = end_time - start_time
        times_per_round.append(elapsed)
        print(f"NumPy - Round {round_idx + 1}: {elapsed:.4f} 秒")

    return {
        "mean": np.mean(times_per_round),
        "std": np.std(times_per_round),
        "times": times_per_round,
    }


def time_pytorch_implementation(rounds: int = 3) -> dict:
    """
    测试 PyTorch 实现的执行时间.

    Args:
        rounds: 测试轮数

    Returns:
        包含平均时间、各轮时间的字典
    """
    dataset = HybridImageDataset(APP_CONFIG)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    model = HybridImageModel(cutoff_frequency=APP_CONFIG.hybrid.best_cutoffs[0])
    model.eval()

    times_per_round: list[float] = []

    for round_idx in range(rounds):
        start_time = time.time()

        with torch.no_grad():
            for image_a, image_b, cutoff_frequency in dataloader:
                image_a_tensor: torch.Tensor = image_a.permute(0, 3, 1, 2).float()
                image_b_tensor: torch.Tensor = image_b.permute(0, 3, 1, 2).float()

                cutoff = cutoff_frequency.item()
                model.update_cutoff_frequency(cutoff)

                _, _, _ = model(image_a_tensor, image_b_tensor)

        end_time = time.time()
        elapsed = end_time - start_time
        times_per_round.append(elapsed)
        print(f"PyTorch - Round {round_idx + 1}: {elapsed:.4f} 秒")

    return {
        "mean": np.mean(times_per_round),
        "std": np.std(times_per_round),
        "times": times_per_round,
    }


def print_comparison(numpy_results: dict, pytorch_results: dict) -> None:
    """
    打印性能比较结果.

    Args:
        numpy_results: NumPy 实现的结果
        pytorch_results: PyTorch 实现的结果
    """
    print("\n" + "=" * 60)
    print("性能比较结果")
    print("=" * 60)

    print(f"\nNumPy 实现:")
    print(f"  平均时间: {numpy_results['mean']:.4f} 秒")
    print(f"  标准差:   {numpy_results['std']:.4f} 秒")

    print(f"\nPyTorch 实现:")
    print(f"  平均时间: {pytorch_results['mean']:.4f} 秒")
    print(f"  标准差:   {pytorch_results['std']:.4f} 秒")

    speedup = numpy_results['mean'] / pytorch_results['mean']
    print(f"\n速度提升: {speedup:.2f}x")

    if speedup > 1:
        print(f"  => PyTorch 比 NumPy 快 {speedup:.2f} 倍")
    else:
        print(f"  => NumPy 比 PyTorch 快 {1/speedup:.2f} 倍")

    print("=" * 60 + "\n")


def main() -> None:
    """
    执行时间比较测试.
    """
    print("\n开始性能比较测试...\n")
    print(f"数据集: {len(APP_CONFIG.paths.image_pairs)} 对图像")
    print(f"测试轮数: 3\n")

    print("-" * 60)
    print("测试 NumPy 实现...")
    print("-" * 60)
    numpy_results = time_numpy_implementation(rounds=3)

    print("\n" + "-" * 60)
    print("测试 PyTorch 实现...")
    print("-" * 60)
    pytorch_results = time_pytorch_implementation(rounds=3)

    print_comparison(numpy_results, pytorch_results)


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
