# 3T-SFT：通过文本通信训练多智能体

**只使用最终答案标签，让前面负责规划、求解或验证的 Agent 也能得到训练信号。**

项目名称为 **3T-SFT（Training Through Text）**。当前论文将核心方法称为 **RADST-SFT**，即考虑重分词的双重直通估计监督微调。

普通多 Agent 系统通过文字交流，但采样、解码和重新分词会切断梯度。这里保留实际生成的文字，在训练阶段增加一条近似梯度通路，将最终答案损失传给上游 Agent；推理阶段继续使用普通文本通信。

## 已经落地的内容

- **真实的双重直通估计与手动梯度传递**：不仅有接口，还包含前向一致性与梯度等价性测试。
- **不同分词器之间的对齐**：按 UTF-8 字节跨度计算重叠，处理提示词边界、特殊 token 和无有效重叠的位置。
- **多 Agent 联合训练**：在有限执行图上逆序传播任务梯度，累加分支贡献，并将局部稳定项与递归梯度分开。
- **完整的小模型训练入口**：包含 LoRA、优化器、梯度裁剪、验证集选模、检查点、日志和推理脚本。
- **末端 SFT 基线**：相同初始化和数据划分下，比较是否能把最终答案梯度传给前面的 Agent。

## 运行

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/train_tiny.py
python scripts/train_tiny.py --mode terminal --output outputs/tiny-terminal
python scripts/infer_tiny.py
```

CPU 示例采用三名小型 GRU Agent，不需要下载大模型。它的作用是检查训练机制是否真正连通；它不是论文实验模型，也不能替代正式基准测试。

本地验证中，完整方法的三个 Agent 都获得了非零任务梯度，而末端 SFT 的前两个 Agent 梯度为零。具体环境、结果和适用范围见 [验证记录](docs/validation.md)。

## 阅读建议

先看 [方法说明](docs/method.md)，再读 `transport.py` 和 `engine.py`；想了解训练如何执行，可从 `scripts/train_tiny.py` 进入。接入实际大模型前，请阅读 [集成说明](docs/integration.md)。

仓库是依据论文新整理的参考实现，提交历史对应本次实际开发过程。未包含原论文训练权重、完整实验数据或未经验证的复现成绩。
