# v0.1.0 本地试用验证（2026-10-13）

- 从白名单源码ZIP解压到独立目录；未带旧data、library、日志或模型。
- Python 3.13新建venv，实际安装requirements-dev.txt成功；未安装torch/transformers。
- 公开Python测试82项通过，覆盖首次空库/不完整索引/上传限定范围、解析、Agent控制流、worker边界、任务持久化及共享embedding。
- 独立源码目录实际npm ci + Vue构建成功；Java21/Maven package成功，23项Java测试通过。Maven验证复用已下载的公共依赖缓存及开发者镜像配置；没有证明首次在线Maven下载在所有网络环境成功。
- 隔离端口实际启动Java8098/Python8798，明确stub后端；HTTP确认初始空库、合成PDF上传和就绪、原PDF读取、页图、限定范围任务完成、0模型token、关闭journal、阶段SSE及Vue静态资源。
- 此轮没有在干净新venv重装CUDA模型环境或运行真实Qwen推理；真实模型能力来自此前开发环境验证。首次使用的真实索引选择结构由受控测试验证，真实新机器/不同GPU仍需实测。
- 未停止原8089/8789工作台；未改变私人论文或历史gold。源码包由白名单生成，没有包含个人原始数据或模型权重。

版本为源码试用版，需要安装前提和构建；不是独立exe。演示模式可以无GPU运行，但不展示真实回答质量。
