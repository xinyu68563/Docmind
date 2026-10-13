# 测试与评测范围

公开仓库不带私人语料和历史gold。公开CI运行解析、Agent控制流、任务边界、持久化、embedding共享与首次使用测试，使用fake模型，不测真实模型质量。

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest tests -q
service\mvnw.cmd -f service/pom.xml test
```

开发工作区中更早的历史语料测试不进入源码包。公开测试列表以打包清单为准。真实推理需要本地模型和GPU，不能由stub链路通过推断真实推理通过。

历史真实模型在RTX5060 8GB验证了上传、限定范围问答和留痕。公开论文9题为便利样本且基础模型可能已见过论文；WorkBuddy辅助分42/45，Codex对其中一项扣分有异议，建议43/45，用户尚未最终评分。二者都不是整体准确率，不证明Agent收益。

发布验证的实际命令、数量和限制见docs/RELEASE_VALIDATION.md。原始私人数据及历史实验保留在开发工作区，没有随仓库传输。

公开源码的首次安装验证见[README部署验证](DEPLOYMENT_CHECK.md)。

Windows还提供2项启动器回归：不完整JAR拒绝启动、运行副本与构建输出隔离及副本完整性检查。Linux CI中这两项会明确跳过，Windows环境通过powershell执行。
