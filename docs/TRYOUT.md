# 朋友本地试用

下载仓库Code → Download ZIP并解压。请使用Python 3.11+、Java 21、Node.js（含npm）、Apache Maven。安装这些官方工具后重新打开终端使PATH生效。不要在GitHub页面直接运行Python。

## 演示模式（无GPU和模型下载）

在解压后的项目根目录打开PowerShell，执行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.venv\Scripts\python.exe scripts/check_environment.py
.venv\Scripts\python.exe scripts/build_frontend.py
service\mvnw.cmd -f service/pom.xml package
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_docmind.ps1 -Backend stub -ServicePort 8088 -WorkerPort 8788
```

需要真实问答时，先完成同样的环境安装和构建，再继续下方“真实模型模式”。

浏览器地址8088，worker8788；默认真实工作台8089/8789可同时保留。浏览器左上角显示“桩后端 · 非模型”，固定回答以STUB开头。生成token为0。

运行 `.venv\Scripts\python.exe scripts/make_demo_pdf.py` 生成原创合成测试PDF，上传 `examples/demo.pdf`，等待就绪后选择它并提交问题。演示模式的上传向量是确定性占位数据，不用LLM，也没有语义检索质量含义。切换real不会复用stub索引。

关闭启动脚本不会自动停止后台服务。用任务管理器核对命令行后结束相应DocMind Java和Python进程；不要直接终止所有Java/Python程序。运行取消不能立即中止GPU正在执行的生成。

## 真实模型模式

需兼容NVIDIA CUDA显卡，建议8GB或更多显存，并预留至少25GB磁盘供模型和环境。已验证环境的PyTorch为2.14.1+cu130；这不是全部显卡的通用配置。按PyTorch官方选择器为自己的驱动/显卡安装对应版本：https://pytorch.org/get-started/locally/ 。RTX50系列需要支持其架构的CUDA/PyTorch。

已验证机器的安装命令：

```powershell
.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/download_models.py
.venv\Scripts\python.exe scripts/check_environment.py --real
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_docmind.ps1
```

下载失败可自行下载官方模型快照，分别完整放入 `data/models/bge-m3` 和 `data/models/Qwen2.5-7B-Instruct`。仓库不带权重；下载文件完整后上传自己的文本PDF。新安装的文档库为空，上传后务必勾选该论文再提问。首次问题需加载模型，等待较久。没有训练或微调模型。

## 常见问题

- 找不到Java/Maven/npm：安装对应官方工具，重新打开终端，再运行环境检查。Java须21。
- 缺jar：先执行前端构建及Maven package；构建失败不能跳过。
- 端口占用/后端不符：不要复用其他程序端口，使用ServicePort/WorkerPort参数更换一对端口。
- 依赖版本无法获取：保留报错，提供Python版本、系统与pip输出；未在你的组合上完成安装验证，不能以开发者现有venv代替。
- 无引用/校验失败：打开原PDF核对，保留原答案反馈，不把状态绿灯当真值。

## 反馈模板

操作系统/显卡/模式；安装或操作步骤；预期和实际表现；报错截图；问题及引用页。公开Issue只附你允许公开的材料，勿包含私人论文、密钥或日志中的私人问题。
