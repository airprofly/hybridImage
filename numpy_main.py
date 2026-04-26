from typing import cast

from dotenv import load_dotenv
from pathlib import Path

from matplotlib import pyplot as plt

from utils.utils import (
    create_Gaussian_kernel,
    create_hybrid_image,
    load_image,
    save_image,
    show_image,
)


def main() -> None:
    from configs.config import APP_CONFIG

    # 输出目录 - NumPy 版本
    output_dir = Path(APP_CONFIG.paths.output_dir).parent / "hybrid_numpy"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_pairs = APP_CONFIG.paths.image_pairs
    for i, pair in enumerate(image_pairs):
        print(
            f"\nProcessing image pair {i + 1}/{len(image_pairs)}: {pair.image_a} & {pair.image_b}"
        )
        image_a_path = cast(Path, pair.image_a)
        image_b_path = cast(Path, pair.image_b)
        image_a = load_image(image_a_path)
        image_b = load_image(image_b_path)

        cutoff_frequency = APP_CONFIG.hybrid.best_cutoffs[i]
        print(f"Using cutoff frequency: {cutoff_frequency}")

        filter_kernel = create_Gaussian_kernel(cutoff_frequency)
        low_frequencies, high_frequencies, hybrid_image = create_hybrid_image(
            image_a, image_b, filter_kernel
        )

        # 为每对图像创建单独的子目录
        pair_dir = output_dir / f"pair_{i + 1}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        save_image(low_frequencies, pair_dir / "low_frequencies.jpg")
        save_image(
            high_frequencies + 0.5, pair_dir / "high_frequencies.jpg"
        )  # Shift for visualization
        save_image(hybrid_image, pair_dir / "hybrid_image.jpg")

        print(f"Saved results for image pair {i + 1} to {output_dir}")

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        show_image(ax[0], low_frequencies, "Low Frequencies")
        show_image(ax[1], high_frequencies + 0.5, "High Frequencies (shifted)")
        show_image(ax[2], hybrid_image, "Hybrid Image")
        plt.suptitle(
            f"Image Pair {i + 1}: {image_a_path.name} & {image_b_path.name} (cutoff={cutoff_frequency})"
        )
        plt.savefig(pair_dir / "results.png")
        plt.close()  # 关闭图形，避免内存泄漏


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
