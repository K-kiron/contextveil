# ContextVeil

一个以英文为主、在本地运行的文字脱敏框架，提供 Python SDK 和命令行，采用 MIT 许可证。

核心方法是**只替换选中的敏感片段，不让生成模型重写全文**。同一实体在批次中使用一致的带类型代号；显式别名表可以关联全名、简称和昵称。金额、否定、关系、顺序等未被选中的文本保持原样。

它提供有边界的保真保证，而不声称“自动找全所有 PII”或“任何任务都完全不掉精度”。检测器是否找全、转换是否改坏内容、下游任务精度是否下降，是三个分别验证的问题。

## 快速使用

```sh
git clone https://github.com/K-kiron/contextveil.git
cd contextveil
python -m pip install -e ".[nlp,formats,vault,benchmark,dev]"
python -m spacy download en_core_web_sm
contextveil anonymize examples/story.txt --glossary examples/glossary.json --output-dir output
```

输出目录必须尚不存在。输入文件不会被覆盖，输出使用中性文件名并附带检查报告。安装时下载依赖和指定模型，处理文档时不调用云服务。

支持 TXT、Markdown、日志、JSON、JSONL、CSV、TSV、YAML、HTML、XML、DOCX 和带文字层的 PDF。其中前几种输出同类格式；HTML/XML/Word/PDF 输出新的 TXT，不复制原文件的隐藏内容、附件或元数据。这些提取格式始终标记为需要复核，执行时需明确加上 `--allow-review`。

JSON/JSONL/YAML 的小数不会先变为二进制浮点数再损失精度。未选中的标量类型、数组及行列顺序保留；敏感键名或数字型 ID 改成代号可能改变下游结构，报告会指出并要求复核。YAML 不支持别名和自定义标签，丢弃注释时也会提示。

## 精度怎样验证

1. **检测质量**：用人工标注的敏感片段计算 precision、recall、F1 和敏感字符覆盖率；可设置失败门槛。
2. **转换保真**：验证选中片段之外的文字不变、实体代号一致、结构化输出可重新解析、普通数值类型不变。
3. **业务精度**：对原文与脱敏文本运行同一任务，对照同一标准答案，比较准确率和原本正确答案的保留率。

```sh
contextveil evaluate examples/evaluation.jsonl
contextveil evaluate your-labeled-cases.jsonl --min-recall 0.99 --min-precision 0.95
python examples/utility_evaluation.py
```

随附的 13 条合成检测样例与业务示例用于回归，不代表真实业务验证。`checks_passed` 只表示声明的检查通过，不能解释为“没有任何个人信息”。默认模型仍然会误判或漏判；日期、年龄、罕见事件等保留事实也可能导致重新识别。

0.2 版增加了完整公开测试集评测：Gretel 5,000 篇、TAB 127 篇（555 个独立标注层），以及 AG News 7,600 条新闻的下游分类。评测在测试前固定了检测参数、代码和模型版本，并对同一分类器比较原文和实际脱敏输出，统计准确率、macro-F1、变化样本子集、配对置信区间、失败与复核比例。结果和复现命令见 [公开评测报告](docs/public-evaluation.md)。这份报告包含精度下降和未通过的验收门槛，不把“非敏感文字没变”当作业务精度证明。

使用本地 PII 专用模型（在源码目录运行）：

```sh
python -m pip install -e ".[neural,formats,vault]"
python -m benchmarks.download --dest ../contextveil-assets model
contextveil anonymize examples/story.txt --neural-model-dir ../contextveil-assets/models/gliner-pii --output-dir neural-output
```

模型约 1.16 GB；运行时校验固定版本的权重和分词器哈希，不联网。安装兼容的 PyTorch CUDA 后可加 `--device cuda`，默认使用 CPU。该检测配置也会遮盖日期等间接标识，不能保证时间推理等任务无损；残留检查要求复核时，命令默认不输出，只有确认范围后才使用 `--allow-review`。

## 恢复与扩展

默认不保存恢复映射。加上 `--recoverable` 可生成使用口令加密的恢复文件；口令通过交互输入或指定环境变量读取，不写入命令行。恢复文件加密保存规范化后的原文，保证别名拼写及数字类型能正确恢复；不恢复原 PDF/Word 的版式和二进制结构。恢复时还需要匹配报告中的随机文档 ID，命令行默认自动读取同名 `.report.json`，避免串用恢复文件。

检测器、字段策略、别名表、业务验收函数和配对任务评估都可通过 SDK 扩展。日期平移、自动推测代词指代、图像/OCR、在线服务和网页界面不在此版本范围。

详细接口、安全边界和命令见 [英文 README](README.md) 与 [保真约定](docs/assurance.md)。
