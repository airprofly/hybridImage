from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ImagePairConfig:
    """
    用途: 管理单个图像对的输入文件路径.

    Attributes:
        image_a (str | Path): 图像对中的第一个图像文件名, 例如 "1a_dog.bmp".
        image_b (str | Path): 图像对中的第二个图像文件名, 例如 "1b_cat.bmp".
    """

    image_a: str | Path
    image_b: str | Path

    def __post_init__(self) -> None:
        if isinstance(self.image_a, str):
            object.__setattr__(self, "image_a", Path(self.image_a))
        if isinstance(self.image_b, str):
            object.__setattr__(self, "image_b", Path(self.image_b))


@dataclass(frozen=True)
class HybridConfig:
    """
    用途: 管理混合图像的截止频率参数配置.

    Attributes:
        best_cutoffs (list[float]): 每个图像对的最佳截止频率值列表, 按配对顺序 (1-5) 对应.
                                    例如: [20.0, 25.0, 30.0, 35.0, 40.0] 表示 5 对图像各自的最佳截止频率.
    """

    best_cutoffs: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class PathConfig:
    """
    用途: 管理项目输入输出路径配置.

    Attributes:
        data_dir (str | Path): 输入图像数据目录路径, 默认 ./data, 支持字符串或 Path 对象输入.
        output_dir (str | Path): 混合图像输出目录路径, 默认 ./outputs/hybrid, 支持字符串或 Path 对象输入.
        image_pairs (list[ImagePairConfig]): 所有图像对的文件名配置列表, 按 pair_id 顺序排列.
    """

    data_dir: str | Path = Path("./data")
    output_dir: str | Path = Path("./outputs/hybrid")
    image_pairs: list[ImagePairConfig] = field(
        default_factory=lambda: [
            ImagePairConfig(image_a="1a_dog.bmp", image_b="1b_cat.bmp"),
            ImagePairConfig(image_a="2a_motorcycle.bmp", image_b="2b_bicycle.bmp"),
            ImagePairConfig(image_a="3a_plane.bmp", image_b="3b_bird.bmp"),
            ImagePairConfig(image_a="4a_einstein.bmp", image_b="4b_marilyn.bmp"),
            ImagePairConfig(image_a="5a_submarine.bmp", image_b="5b_fish.bmp"),
        ]
    )

    def __post_init__(self) -> None:
        # 先转换 data_dir 和 output_dir 为 Path 对象
        if isinstance(self.data_dir, str):
            object.__setattr__(self, "data_dir", Path(self.data_dir))
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

        # 使用 Path() 确保类型为 Path (Path(Path) 返回自身)
        data_dir = Path(self.data_dir)

        # 将 data_dir 拼接到 image_pairs 中每个文件名, 生成完整路径
        updated_pairs = [
            ImagePairConfig(
                image_a=data_dir.joinpath(pair.image_a),
                image_b=data_dir.joinpath(pair.image_b),
            )
            for pair in self.image_pairs
        ]
        object.__setattr__(self, "image_pairs", updated_pairs)


@dataclass(frozen=True)
class AppConfig:
    """
    用途: 聚合全局配置入口.

    Attributes:
        paths (PathConfig): 路径相关配置, 包括数据目录、输出目录和图像对配置.
        hybrid (HybridConfig): 混合图像相关配置, 包含每个图像对的最佳截止频率值.
    """

    paths: PathConfig = field(default_factory=PathConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)

    @classmethod
    def load_from_yaml(cls, yaml_path: str | Path) -> "AppConfig":
        """
        从 YAML 文件加载配置.

        Args:
            yaml_path (str | Path): YAML 配置文件的路径.

        Returns:
            AppConfig: 加载完成的配置对象.

        Raises:
            ValueError: 当 YAML 格式错误或文件不存在时抛出异常.
        """
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"\033[1;91mInvalid YAML format in {yaml_path}: {e}\033[0m") from e
        except FileNotFoundError as e:
            raise ValueError(f"\033[1;91mConfiguration file not found: {yaml_path}\033[0m") from e

        paths_dict = config_dict.get("paths", {})
        hybrid_dict = config_dict.get("hybrid", {})

        image_pairs_dicts = paths_dict.get("image_pairs", [])
        image_pairs = [
            ImagePairConfig(**pair_dict) for pair_dict in image_pairs_dicts
        ] if image_pairs_dicts else None

        return cls(
            paths=PathConfig(
                data_dir=paths_dict.get("data_dir", "./data"),
                output_dir=paths_dict.get("output_dir", "./outputs/hybrid"),
                image_pairs=image_pairs if image_pairs else field(default_factory=list),
            ),
            hybrid=HybridConfig(
                best_cutoffs=hybrid_dict.get("best_cutoffs", []),
            ),
        )

# 全局配置实例 (模块导入时加载一次, 全局唯一)
# 其他模块通过 `from configs.config import APP_CONFIG` 导入使用
_temp_dir = Path(__file__).parent
_yaml_path = _temp_dir.joinpath("config.yaml")
try:
    APP_CONFIG = AppConfig.load_from_yaml(_yaml_path)
except ValueError as e:
    print(f"\033[1;93m[WARNING] 加载 YAML 失败, 使用默认配置: {e}\033[0m")
    APP_CONFIG = AppConfig()