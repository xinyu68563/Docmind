<div align="center">

# DocMind

**本地运行的论文问答与原文溯源工具**

[![Source checks](https://github.com/xinyu68563/Docmind/actions/workflows/ci.yml/badge.svg)](https://github.com/xinyu68563/Docmind/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

[下载](https://github.com/xinyu68563/Docmind/raw/refs/heads/main/downloads/docmind-0.1.0-local-source.zip) · [使用指南](docs/USAGE.md) · [安装说明](docs/TRYOUT.md) · [问题反馈](https://github.com/xinyu68563/Docmind/issues)

</div>

DocMind 让你直接向论文提问，并通过引用片段和 PDF 原文检查回答。上传一篇或多篇论文，选择本次阅读范围，即可查询方法、实验设置、定义与结论。

当前版本支持本地 Qwen2.5-7B 推理，也提供无需显卡的流程演示模式。

## 功能

- **论文问答**：围绕选中的论文提出问题，回答附带引用编号。
- **多篇阅读**：一次选择最多 10 篇论文，在指定范围内检索和归纳。
- **原文溯源**：点击引用查看完整片段、对应 PDF 物理页，支持翻页与缩放。
- **文档管理**：上传文本 PDF，查看解析和索引进度，重复文件复用索引。
- **阅读记录**：回看历史任务、导出问题与回答，保留证据便于复核。
- **本地运行**：模型准备完成后，论文处理和推理在本机执行。

## 快速开始

### 环境准备

| 依赖 | 已验证配置 |
| --- | --- |
| Python | 3.13 |
| Java | 21 |
| Node.js | 22，含 npm |
| Maven | 3.9 系列 |

Windows 启动方式如下。完整安装、模型下载和故障排查见[安装说明](docs/TRYOUT.md)。

### 下载并启动

下载[源码试用包](https://github.com/xinyu68563/Docmind/raw/refs/heads/main/downloads/docmind-0.1.0-local-source.zip)并解压，在项目根目录打开 PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.venv\Scripts\python.exe scripts/build_frontend.py
service\mvnw.cmd -f service/pom.xml package
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_docmind.ps1 -Backend stub -ServicePort 8088 -WorkerPort 8788
```

浏览器打开 **http://127.0.0.1:8088**。

> 以上启动演示模式，可体验上传、任务和原文查看。界面标注“桩后端 · 非模型”，回答为固定模拟内容。使用自己的论文进行真实问答，请继续完成[真实模型配置](docs/TRYOUT.md#真实模型模式)。

### 开始阅读

1. 上传 PDF，等待文档状态变为“可提问”。
2. 选中要阅读的论文，输入问题并点击“检索并回答”。
3. 点击回答中的引用编号，查看证据片段和原 PDF。

例如：

> 这篇论文的方法包含哪些步骤？请引用原文。
>
> 实验使用了什么数据集、硬件和训练设置？未报告的项目请明确指出。
>
> 这两篇论文采用了哪些不同的假设？请分别提供依据。

更多操作见[使用指南](docs/USAGE.md)。

## 本地模型

真实模式使用 **Qwen2.5-7B-Instruct（NF4）** 生成回答，结合 **bge-m3、BM25 和 RRF** 检索论文片段。已在 RTX 5060 8GB 上验证；其他显卡和驱动组合需实测。模型权重单独下载，首次安装与下载需要联网。

## 使用范围

当前支持能选中文字的 PDF：单文件最多 **20MB / 200页 / 1200个切分片段**。扫描件需先 OCR；复杂公式和表格建议直接查看 PDF。模型回答和自校验可能出错，重要结论请核对原文。

当前为个人本地试用版本，尚未提供账号、多用户在线服务或页内框选高亮。

## 文档

- [使用指南](docs/USAGE.md)：上传、范围选择、引用查看和任务记录。
- [安装说明](docs/TRYOUT.md)：演示模式、真实模型与常见安装问题。
- [技术架构](docs/ARCHITECTURE.md)：检索、Agent 与服务组成。
- [测试说明](docs/TESTING.md)：回归测试及评测范围。

## 反馈与贡献

欢迎通过 [Issues](https://github.com/xinyu68563/Docmind/issues) 报告问题或提出建议。请附操作步骤、运行模式和必要的报错信息；涉及未公开论文时，请使用可分享的最小复现材料。

## License

DocMind 使用 [GNU AGPL-3.0-or-later](LICENSE) 许可证。第三方组件和模型遵循各自许可，详见[第三方说明](THIRD_PARTY_NOTICES.md)。
