from pathlib import Path
from typing import cast

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray
from numpy.lib.stride_tricks import as_strided
from PIL import Image
import os


def load_image(image_path: str | Path) -> NDArray[np.float32]:
    """
    Load an image from the specified path and convert it to a NumPy array. The image is normalized to the range [0, 1].
    Args:
        path (str): The file path to the image.
    Returns:
        np.ndarray: A NumPy array representing the image, with pixel values normalized to the range [0, 1].
    """
    image = Image.open(image_path).convert("RGB")
    image = np.asarray(image)
    float_image = image.astype(np.float32) / 255.0
    return float_image


def save_image(image: NDArray[np.floating], path: str | Path) -> None:
    """
    Save a NumPy array as an image to the specified path. The pixel values are expected to be in the range [0, 1] and will be scaled to [0, 255] before saving.
    Args:
        image (np.ndarray): A NumPy array representing the image, with pixel values in the range [0, 1].
        path (str): The file path where the image will be saved.
    """
    scaled_image = (image * 255).astype(np.uint8)
    pil_image = Image.fromarray(scaled_image)
    pil_image.save(path)


def create_Gaussian_kernel(cutoff_frequency: int | float) -> NDArray[np.float64]:
    """
    Returns a 2D Gaussian kernel using the specified filter size standard
    deviation and cutoff frequency.

    The kernel should have:
    - shape (k, k) where k = cutoff_frequency * 4 + 1
    - mean = floor(k / 2)
    - standard deviation = cutoff_frequency
    - values that sum to 1

    Args:
    - cutoff_frequency: an int controlling how much low frequency to leave in
      the image.
    Returns:
    - kernel: numpy nd-array of shape (k, k)

    HINT:
    - The 2D Gaussian kernel here can be calculated as the outer product of two
      vectors with values populated from evaluating the 1D Gaussian PDF at each
      corrdinate.
    """
    # 计算核的大小
    k = cutoff_frequency * 4 + 1

    # 计算均值（中心位置）
    mean = k // 2

    # 标准差
    sigma = cutoff_frequency

    # 创建坐标向量
    x = np.arange(k)

    # 计算 1D 高斯 PDF
    # 高斯公式: (1 / (sigma * sqrt(2*pi))) * exp(-((x - mean)^2) / (2 * sigma^2))
    gaussian_1d = np.exp(-((x - mean) ** 2) / (2 * sigma**2))
    gaussian_1d: np.ndarray = gaussian_1d / (sigma * np.sqrt(2 * np.pi))

    # 通过外积计算 2D 高斯核
    kernel = np.outer(gaussian_1d, gaussian_1d)

    # 归一化，确保总和为 1
    kernel: np.ndarray = kernel / np.sum(kernel)

    return kernel


def apply_filter(
    image: NDArray[np.floating], kernel: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Apply a filter to an image. Return the filtered image.

    Args
    - image: numpy nd-array of shape (img_height, img_width, num_channels)
    - kernel: numpy nd-array of shape (kernel_height, kernel_width)
    Returns
    - output_image: numpy nd-array of shape (img_height, img_width, num_channels)

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to work
    with matrices is fine and encouraged. Using OpenCV or similar to do the
    filtering for you is not allowed.
    - I encourage you to try implementing this naively first, just be aware that
    it may take an absurdly long time to run. You will need to get a function
    that takes a reasonable amount of time to run so that the TAs can verify
    your code works.
    """
    img_height: int = image.shape[0]
    img_width: int = image.shape[1]
    num_channels: int = image.shape[2]
    kernel_height: int = kernel.shape[0]
    kernel_width: int = kernel.shape[1]

    # 计算填充量
    pad_h: int = kernel_height // 2
    pad_w: int = kernel_width // 2

    # 使用边缘像素值填充图像
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="edge")

    # 使用 as_strided 创建滑动窗口视图
    # 形状: (img_height, img_width, kernel_height, kernel_width, num_channels)
    strides = padded_image.strides  # (行步幅, 列步幅, 通道步幅)
    window_shape = (img_height, img_width, kernel_height, kernel_width, num_channels)
    window_strides = (strides[0], strides[1], strides[0], strides[1], strides[2])
    windows = as_strided(padded_image, shape=window_shape, strides=window_strides)

    # 使用 einsum 进行批量卷积: 对 kernel_height 和 kernel_width 维度求和
    output_image = np.einsum("ijklm,kl->ijm", windows, kernel)

    return output_image


def show_image(axis: Axes, image: NDArray[np.floating], title: str) -> None:
    """
    Display an image on the specified matplotlib axis.

    Args:
        axis (matplotlib.axes.Axes): The matplotlib axis object to display the image on.
        image (NDArray[np.floating]): The image array to display.
        title (str): The title for the image.
    """
    axis.imshow(np.clip(image, 0, 1))
    axis.set_title(title)
    axis.axis("off")


def create_hybrid_image(
    image1: NDArray[np.floating],
    image2: NDArray[np.floating],
    filter_kernel: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Takes two images and a low-pass filter and creates a hybrid image. Returns
    the low frequency content of image1, the high frequency content of image 2,
    and the hybrid image.

    Args:
        image1: numpy nd-array of dim (m, n, c)
        image2: numpy nd-array of dim (m, n, c)
        filter_kernel: numpy nd-array of dim (x, y)
    Returns:
        low_frequencies: numpy nd-array of shape (m, n, c)
        high_frequencies: numpy nd-array of shape (m, n, c)
        hybrid_image: numpy nd-array of shape (m, n, c)

    HINTS:
    - You will use your my_imfilter function in this function.
    - You can get just the high frequency content of an image by removing its low
      frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values of the hybrid image are between
      0 and 1. This is known as 'clipping'.
    - If you want to use images with different dimensions, you should resize them
      in the notebook code.
    """
    # 获取 image1 的低频内容
    low_frequencies = apply_filter(image1, filter_kernel)

    # 获取 image2 的低频内容，然后从原图中减去得到高频内容
    image2_low_frequencies = apply_filter(image2, filter_kernel)
    high_frequencies = image2 - image2_low_frequencies

    # 混合图像 = 低频 + 高频
    hybrid_image = low_frequencies + high_frequencies

    # 裁剪到 [0, 1] 范围
    hybrid_image = np.clip(hybrid_image, 0, 1)

    return low_frequencies, high_frequencies, hybrid_image


if __name__ == "__main__":
    from configs.config import APP_CONFIG

    # load image
    image_path = cast(Path,APP_CONFIG.paths.image_pairs[0].image_a)
    image = load_image(image_path)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    show_image(ax[0], image, "Original Image")

    # create a Gaussian kernel with cutoff frequency 1
    cutoff_frequency = 7
    kernel = create_Gaussian_kernel(cutoff_frequency)
    print("\033[1;32m Gaussian Kernel: \033[0m")
    print(kernel)

    # apply the Gaussian kernel to the image
    filtered_image = apply_filter(image, kernel)
    show_image(ax[1], filtered_image, f"Filtered Image (cutoff frequency={cutoff_frequency})")

    # show the original and filtered images
    plt.show()

