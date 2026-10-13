# DocMind —— 检索：向量嵌入

"""
向量嵌入模块（Phase 1）

【 BM25 缺什么？为什么还需要向量？】

上一阶段我们用 BM25 检索，它靠"字面是否出现同一个词"来判断相关性。
这有个致命缺陷，叫 **词汇鸿沟（vocabulary mismatch）**：

    用户问："这个方法比传统手段快多少？"
    文档写："所提算法的收敛速度显著优于高斯正则化方法。"

    这两句话没有一个词是重合的，BM25 打 0 分，完全漏召回。
    但语义上它们高度相关 —— 这正是 embedding 要解决的问题。

【embedding 在做什么？】

把一段文字映射成一个固定长度的稠密向量（bge-m3 是 1024 维），
语义相近的文本在这个向量空间里距离也近。检索时就变成了"找最近邻"。

注意：它和 BM25 是**互补**而不是替代关系。真实数据里：
    - 专有名词 / 精确术语 -> BM25 更强
    - 同义改写 / 口语化描述 -> 向量更强
所以生产系统一定做双路召回再融合，这也是我们这阶段要做的。

【为什么选 bge-m3？】

1. 多语言（中英混排场景必需，你的语料是英文学术论文）
2. 1024 维，检索精度显著高于常见的 384 维小模型
3. max_length 8192，能吃下超长段落
4. 由 BAAI 开源，中文检索榜长期霸榜，简历上写这个不算野鸡模型

显存占用：fp16 模式下约 1.2GB，你的 RTX 5060 8GB 完全够。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


# 中国网络环境下 HuggingFace 直连不通，走镜像。
# 这行必须在 import sentence_transformers / huggingface_hub 之前执行才生效。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"

# 远端仓库地址（本地没有时才会用到）
MODEL_PRESETS = {
    "small": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # ~470MB，384维
    "base": "BAAI/bge-base-zh-v1.5",                                          # ~400MB，768维
    "m3": "BAAI/bge-m3",                                                      # ~2.2GB，1024维，推荐
}

# preset -> 本地期望目录名，与 scripts/download_model.sh 的下载目标保持一致
_LOCAL_DIRNAME = {
    "small": "paraphrase-multilingual-MiniLM-L12-v2",
    "base": "bge-base-zh-v1.5",
    "m3": "bge-m3",
}


def resolve_model_path(preset: str) -> str:
    """优先使用本地已下载的模型，避免每次都走网络。

    为什么要设计这层 fallback？
        国内网络环境下 HuggingFace 下载极不稳定（详见 scripts/download_model.sh 的说明），
        一旦模型已在本地，就不应该再有任何网络依赖 —— 否则你跑一次实验卡一次网络。
    """
    dirname = _LOCAL_DIRNAME.get(preset)
    if dirname:
        local = _MODEL_DIR / dirname
        if (local / "config.json").exists() and (
            (local / "pytorch_model.bin").exists()
            or (local / "model.safetensors").exists()
        ):
            return str(local)
    return MODEL_PRESETS.get(preset, preset)


class EmbeddingModel:
    """文本 -> 向量的封装。

    用法:
        model = EmbeddingModel("m3")
        vecs = model.encode(["第一段", "第二段"])
        qvec = model.encode_query("用户的问题")
    """

    def __init__(self, preset: str = "m3", batch_size: int = 8, use_fp16: bool = True, device: str | None = None):
        """
        参数
        ----
        preset      : "small" / "base" / "m3"，见 MODEL_PRESETS
        batch_size  : 一次送几段文本进 GPU。显存不够就调小；显存富余就调大提速
        use_fp16    : 半精度。显存占用减半、速度接近翻倍，精度损失可忽略
        """
        from sentence_transformers import SentenceTransformer
        import torch

        self.model_name = resolve_model_path(preset)
        self.batch_size = batch_size

        # 设备选择：有 CUDA 就用 GPU。
        # RTX 5060 是 Blackwell 架构(sm_120)，需要 PyTorch >= 2.7 + CUDA 12.8 才能识别，
        # 装驱动时务必装匹配的 torch 版本，否则 torch.cuda.is_available() 会是 False。
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Embedding] 设备: {self.device}  模型: {self.model_name}")
        if self.device == "cuda":
            print(f"[Embedding] 显卡: {torch.cuda.get_device_name(0)}")

        self.model = SentenceTransformer(self.model_name, device=self.device)
        if use_fp16 and self.device == "cuda":
            self.model = self.model.half()  # 转 fp16

        # sentence-transformers 6.x 改了 API 名，做兼容处理避免升级时炸掉
        try:
            self.dim = self.model.get_embedding_dimension()
        except AttributeError:
            self.dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: list[str],
        is_query: bool = False,
        show_progress: bool = True,
    ) -> np.ndarray:
        """把文本列表编码成向量矩阵，形状 (N, dim)。

        返回的向量已做 L2 归一化 —— 这样余弦相似度就等于向量点积，
        检索时可以直接用矩阵乘法算，快很多而且不容易出错。
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        prefix = "Represent this sentence for searching relevant passages: " if is_query else ""
        if prefix:
            texts = [prefix + t for t in texts]

        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 归一化
            show_progress_bar=show_progress,
        )
        return vecs.astype(np.float32)

    def to(self, device: str) -> None:
        """在 GPU / CPU 之间迁移模型。

        为什么需要这个方法（这是真实踩过的坑）：

        本项目运行时有两个模型要共存：
            bge-m3 (embedding, fp16 约 1.2GB) + Qwen2.5-7B (4bit 约 5.2GB)

        在 8GB 显卡上，先加载 bge-m3 再加载 LLM，bitsandbytes 会发现
        剩余显存"看起来不够"，于是把部分 LLM 层 offload 到 CPU。
        一旦发生 offload，每生成一个 token 都要在 CPU 和 GPU 之间搬数据，
        速度会从 25 tok/s 掉到 2-3 tok/s —— 慢 10 倍且很难察觉原因。

        解法：加载 LLM 之前，把 embedding 模型先挪到 CPU。
        检索时 query 编码在 CPU 上跑（单条约 0.3-1 秒，完全可接受），
        让 LLM 独占 GPU。

        注意 fp16 的问题：半精度在 CPU 上几乎没有硬件加速，反而比 fp32 慢，
        所以迁到 CPU 时要转回 fp32，迁回 GPU 再转 fp16。
        """
        import torch

        self.model = self.model.to(device)
        if device == "cpu":
            self.model = self.model.float()      # CPU 上 fp16 反而慢
        elif torch.cuda.is_available():
            self.model = self.model.half()       # GPU 上 fp16 省显存又快
        self.device = device
        torch.cuda.empty_cache()

    def encode_query(self, query: str) -> np.ndarray:
        """编码单条查询。

        为什么要和 encode 分开？
        因为 bge 系列是「非对称检索」模型：查询和文档的角色不同。
        给查询加特定的 instruction 前缀能明显提升召回率（通常 2-5 个点）。
        这是很多人忽略的细节，也是能体现你对模型理解的点。
        """
        return self.encode([query], is_query=True, show_progress=False)[0]

    def save_index(self, vectors: np.ndarray, chunk_ids: list[str], path: str | Path) -> None:
        """保存向量索引。

        为什么不把向量塞进 chunks.json？
        因为 1024 维 float32 向量占 4KB，一万个 chunk 就是 40MB JSON，
        解析慢且没法做高效的矩阵运算。二进制的 .npz 才是正确做法。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=vectors.astype(np.float32),
            chunk_ids=np.array(chunk_ids),
        )
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"[Embedding] 向量索引已保存: {path}  ({size_mb:.1f} MB)")

    @staticmethod
    def load_index(path: str | Path) -> tuple[np.ndarray, list[str]]:
        data = np.load(path, allow_pickle=True)
        return data["vectors"], list(data["chunk_ids"])
