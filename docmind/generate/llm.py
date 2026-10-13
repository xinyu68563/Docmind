# DocMind —— 生成：本地 LLM 封装

"""
本地大模型加载与推理（Phase 2 基础）

【为什么需要量化：把 7B 塞进 8GB 显存】

Qwen2.5-7B 有 76 亿个参数。模型权重在磁盘/显存里是以浮点数存的：

    fp32（4 字节/参数）  7.6B × 4B = 30.4 GB   <- 你显卡只有 8GB
    fp16（2 字节/参数）  7.6B × 2B = 15.2 GB   <- 还是放不下
    int8（1 字节/参数）  7.6B × 1B =  7.6 GB   <- 刚好卡在边缘，没空间给 KV cache
    int4（0.5 字节/参数）7.6B × 0.5B = 3.8 GB  <- 可行！

4-bit 量化把每个权重从 16 bit 压到 4 bit，只存 16 个离散值（2^4=16）。
bitsandbytes 用的 NF4（Normal Float 4）不是均匀切分，而是**按正态分布分位
来取这 16 个档位** —— 神经网络权重天然接近正态分布，所以靠近 0 的地方档位密、
远离 0 的地方档位疏，同样的 4 bit 能保留更多信息。这是 NF4 比 INT4 好的原因。

实测（RTX 5060 8GB）：加载后占用约 4.5GB，剩余 3.5GB 给 KV cache 和激活值。

【一个必须注意的坑：compute dtype 和 storage dtype 是两回事】

    bnb_4bit_compute_dtype = torch.bfloat16

权重的**存储**是 4 bit，但**计算**时会反量化回 bf16 做矩阵乘法。
如果你不加这个参数，默认用 fp32 计算，速度会慢一大截，而且更容易爆显存。

【为什么不直接用 fp16 而是 bf16】

bf16（Brain Float）指数位和 fp32 一样多（8 bit），只是尾数少，
所以数值范围大、不容易溢出；fp16 尾数多但指数只有 5 bit，
在量化模型的反量化过程中容易上溢成 inf。

【device_map 在量化模式下的陷阱】

写 device_map={"":0} 或者 device_map="cuda" 会和 bitsandbytes 打架，
导致部分层被 offload 到 CPU，推理速度掉 10 倍。
正确写法是 device_map="auto"，让 accelerate 自己决定。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch


# 默认模型目录。放在项目内的 data/models/，保证运行时零网络依赖
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"


def resolve_llm_path(name: str = "Qwen2.5-7B-Instruct") -> Path:
    """定位本地 LLM 目录。

    和 embeddings.py 里的 resolve_model_path 一个思路：先看项目内有没有，
    有就用（离线可跑），没有再退回 HF 缓存。
    """
    local = DEFAULT_MODEL_DIR / name
    if (local / "config.json").exists():
        return local
    raise FileNotFoundError(
        f"本地没有模型: {local}\n"
        f"先运行: python scripts/download_llm.py 7b"
    )


@dataclass
class GenerationStats:
    """一次生成的统计信息。

    为什么要记录这些？因为 Phase 3 做消融实验时要对比
    "不同策略花了多少时间、生成了多少 token"，
    没有这些数据就只能靠感觉说"好像快了一点"。
    """

    prompt_tokens: int = 0
    output_tokens: int = 0
    elapsed: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed <= 0 or self.output_tokens == 0:
            return 0.0
        return self.output_tokens / self.elapsed


class LocalLLM:
    """本地 Qwen2.5 推理封装。

    用法:
        llm = LocalLLM()
        reply = llm.chat("用一句话解释采样定理")

    设计取舍：这里刻意只暴露 chat() 和 generate() 两个方法，
    不做流式输出。原因是 Phase 2 的 Agent 需要的是"一次性拿到完整结果
    再判断下一步"，流式输出反而让逻辑变复杂。
    等 Phase 4 做 Web 服务时再单独加流式。
    """

    def __init__(
        self,
        model_name: str = "Qwen2.5-7B-Instruct",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ):
        """
        参数
        ----
        load_in_4bit  : 是否做 4-bit 量化。8GB 显存必须开。
                        如果你有 24GB 显卡，可以设 False 用 fp16，效果略好。
        temperature   : 默认压到 0.1 —— 这是 RAG 场景的重要默认值。
                        我们要的是"忠实于检索到的证据"，不是创意写作，
                        高温会让模型编造文档里没有的内容。
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_path = resolve_llm_path(model_name)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",           # NF4 而非 fp4，见文件头说明
                bnb_4bit_use_double_quant=True,      # 对量化常数再量化一次，省约 0.4GB
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), trust_remote_code=True
        )

        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            quantization_config=quant_config,
            device_map="auto",          # 关键：量化模式下必须 auto，见文件头说明
            trust_remote_code=True,
        )
        self.load_seconds = time.time() - t0
        self.model.eval()               # 关掉 dropout。推理时忘了这行会引入随机性

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        system: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, GenerationStats]:
        """单轮对话。返回 (回复文本, 统计信息)。"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})
        return self._run(messages, max_new_tokens, temperature)

    def chat_messages(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, GenerationStats]:
        """多轮对话。messages 形如 [{"role":"user","content":"..."}, ...]"""
        return self._run(messages, max_new_tokens, temperature)

    def _run(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None,
        temperature: float | None,
    ) -> tuple[str, GenerationStats]:
        # apply_chat_template 会把 messages 拼成 Qwen 训练时的格式：
        #   <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...
        # 手写拼接很容易漏掉 <|im_end|>，模型就会"跑题"，所以必须用它。
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        n_prompt = inputs["input_ids"].shape[1]

        temp = temperature if temperature is not None else self.temperature

        # temperature = 0 意味着"每次都选概率最高的那个 token"（贪心解码）。
        # 这里有个坑：不能直接把 temperature=0.0 传给 generate()。
        # transformers 会拒绝它（TemperatureLogitsWarper 要求严格正数），
        # 正确做法是 do_sample=False 且**根本不传** temperature 参数。
        #
        # 为什么 Agent 的规划/反思/校验节点要用贪心？
        # 这些节点输出的是结构化 JSON，我们要的是稳定可复现的判断，
        # 而不是每次跑出不一样的结果。只有最终回答节点用低温采样。
        do_sample = temp > 0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temp

        t0 = time.time()
        with torch.no_grad():   # 推理不求导。省显存也省时间
            out = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - t0

        # 截掉 prompt 部分，只保留新生成的 token
        new_tokens = out[0][n_prompt:]
        reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        stats = GenerationStats(
            prompt_tokens=n_prompt,
            output_tokens=len(new_tokens),
            elapsed=elapsed,
        )
        return reply.strip(), stats

    # ------------------------------------------------------------------
    # 显存管理
    # ------------------------------------------------------------------

    @property
    def memory_gb(self) -> float:
        """当前模型占用的显存（GB）"""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 ** 3)

    def unload(self) -> None:
        """释放显存。

        什么时候需要？本项目的流程是"先 embedding 检索，再 LLM 生成"。
        两个模型同时在显存上：bge-m3 约 1.2GB + Qwen 7B 4bit 约 4.5GB = 5.7GB，
        8GB 显卡只剩 2.3GB 给 KV cache，虽然够用但很紧。
        如果后面要跑批量评测，建议生成完就 unload。
        """
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
