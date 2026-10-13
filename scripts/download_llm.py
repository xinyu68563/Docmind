#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2 准备：下载本地 LLM（Qwen2.5-Instruct 系列）

【为什么单独写一个脚本，而不是复用 download_model.sh】
bge-m3 那个脚本走的是 hf-mirror。本次实测发现国内源速度差异极大：

    源                  实测速度        下 15GB 需要
    ------------------  ------------    ------------
    hf-mirror.com       ~0.43 MB/s     ~10 小时   <- 之前 bge-m3 这么下的
    modelscope.cn       ~37 MB/s       ~7 分钟    <- 快 86 倍

ModelScope（魔搭）是阿里官方模型库，国内有 CDN，能跑满带宽。
所以 LLM 这种十几 GB 的大件必须走 ModelScope，别再走 hf-mirror 了。

【为什么用标准库 urllib 而不是 requests / huggingface_hub】
1. requests 在当前 venv 里没装，为一个下载脚本装依赖不划算
2. huggingface_hub 走 Xet CDN，国内会 401（bge-m3 已经踩过这个坑）
3. urllib 支持 Range 请求，断点续传自己实现只要几行

【用法】
    python scripts/download_llm.py              # 默认 7B
    python scripts/download_llm.py 3b           # 下 3B（显存紧张时的退路）
    python scripts/download_llm.py 7b --verify  # 只校验已下载文件是否完整
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# 每个 preset 需要的文件清单。
# safetensors 权重是分片的，必须全部下齐，少一个都加载不了。
# tokenizer 相关的小文件（vocab.json / merges.txt / tokenizer.json）不能漏，
#   否则 transformers 会退化成慢速的 Python tokenizer，且 Qwen 的 BPE 会不准。
PRESETS = {
    "7b": {
        "repo": "qwen/Qwen2.5-7B-Instruct",
        "files": [
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        ],
    },
    "3b": {
        "repo": "qwen/Qwen2.5-3B-Instruct",
        "files": [
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
    },
}

# ModelScope 的文件下载 API。注意 repo 名是小写 qwen（不是 HuggingFace 的 Qwen）
API = "https://modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"

CHUNK = 1 << 20  # 1MB 读缓冲。太小会让 Python 循环次数变多拖慢速度


def human(n: float) -> str:
    """把字节数转成人类可读的形式"""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024


def remote_size(url: str) -> int:
    """用 HEAD 探一下远端文件大小，用来判断本地文件是否下完整"""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers.get("Content-Length", 0))


def download(url: str, dest: Path, total: int) -> bool:
    """
    带断点续传的下载。

    断点续传的原理：发请求时带上 `Range: bytes=<已下载字节数>-`，
    服务器就只返回剩余部分。这样网络中断后重跑脚本不用从头来，
    对 15GB 的模型来说是刚需。
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = tmp.stat().st_size if tmp.exists() else 0

    headers = {}
    mode = "wb"
    if done > 0:
        # 已下载的部分比远端还大，说明是脏数据（比如上次下载被截断后又被追加过）
        if done > total > 0:
            tmp.unlink()
            done = 0
        else:
            headers["Range"] = f"bytes={done}-"
            mode = "ab"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            # 服务器忽略 Range 时会返回 200 而不是 206，这时必须从头写，
            # 否则会把完整内容追加到旧文件后面，得到一个损坏的模型。
            if resp.status == 200 and done > 0:
                tmp.unlink()
                done = 0
                mode = "wb"

            start = time.time()
            last_t = start
            last_n = done
            with open(tmp, mode) as f:
                while True:
                    buf = resp.read(CHUNK)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)

                    now = time.time()
                    if now - last_t >= 0.5:  # 每 0.5s 刷新一次进度，避免刷屏
                        speed = (done - last_n) / (now - last_t)
                        pct = done / total * 100 if total else 0
                        eta = (total - done) / speed if speed > 0 else 0
                        sys.stdout.write(
                            f"\r  {human(done)} / {human(total)}  "
                            f"{pct:5.1f}%  {human(speed)}/s  "
                            f"剩余 {int(eta)}s   "
                        )
                        sys.stdout.flush()
                        last_t, last_n = now, done
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"\n  网络中断: {e}")
        print(f"  已下载 {human(done)}，重新运行本脚本会自动续传")
        return False

    sys.stdout.write("\r" + " " * 90 + "\r")

    # 大小对不上就是坏了，删掉 .part 让下次重下，别让它污染后续加载
    if total and done != total:
        print(f"  大小不符: 期望 {human(total)} 实际 {human(done)}，删除重下")
        tmp.unlink()
        return False

    tmp.rename(dest)  # 只有下完整了才去掉 .part 后缀
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", nargs="?", default="7b", choices=list(PRESETS))
    ap.add_argument("--verify", action="store_true", help="只校验不下载")
    args = ap.parse_args()

    spec = PRESETS[args.preset]
    repo = spec["repo"]
    root = Path(__file__).resolve().parent.parent
    dest_dir = root / "data" / "models" / repo.split("/")[-1]
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"模型 : {repo}")
    print(f"目标 : {dest_dir}")
    print(f"源   : ModelScope (国内 CDN)")
    print()

    ok, missing, total_all = True, [], 0
    for name in spec["files"]:
        target = dest_dir / name
        url = API.format(repo=repo, path=name)

        try:
            total = remote_size(url)
        except Exception as e:
            print(f"  [跳过] {name} 无法获取大小: {e}")
            ok = False
            missing.append(name)
            continue
        total_all += total

        if target.exists() and (not total or target.stat().st_size == total):
            print(f"  [已有] {name} ({human(target.stat().st_size)})")
            continue

        if args.verify:
            print(f"  [缺失] {name}")
            missing.append(name)
            ok = False
            continue

        print(f"  [下载] {name} ({human(total)})")
        if not download(url, target, total):
            ok = False
            missing.append(name)

    print()
    print(f"总计 {human(total_all)}")
    if ok:
        print("全部完成。")
        print()
        print("下一步测试加载（4-bit 量化，8GB 显存够用）:")
        print(f'  .venv/Scripts/python.exe -c "from transformers import AutoModelForCausalLM, AutoTokenizer; '
              f'm=AutoModelForCausalLM.from_pretrained(\'data/models/{repo.split("/")[-1]}\', '
              f'load_in_4bit=True, device_map=\'auto\'); print(m.device)"')
    else:
        print(f"未完成: {missing}")
        print("重新运行本脚本可断点续传。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
