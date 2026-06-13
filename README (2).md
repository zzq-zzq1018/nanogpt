# GPT 训练与生成框架

本仓库包含一个极简、自包含的 GPT（生成式预训练 Transformer）语言模型实现，灵感来自 [nanoGPT](https://github.com/karpathy/nanoGPT)。支持从零开始训练、从检查点恢复、加载 OpenAI GPT‑2 权重以及文本采样/生成。代码可运行在单 GPU 或多 GPU（分布式数据并行，DDP）环境。

## 文件说明

| 文件               | 描述                                                                                   |
|------------------|--------------------------------------------------------------------------------------|
| `model.py`       | GPT 模型的完整定义（Transformer、注意力、MLP、权重绑定等）。                                         |
| `train.py`       | 训练脚本，支持单 GPU、多 GPU（DDP）、日志记录、检查点保存、学习率衰减等。                                     |
| `sample.py`      | 生成脚本，加载训练好的模型并生成文本样本。                                                          |
| `bench.py`       | 简单的基准测试脚本，测量吞吐量和模型 FLOPs 利用率（MFU）。                                            |
| `configurator.py`| “穷人的配置器”，通过命令行或配置文件覆盖超参数。                                                      |

## 环境要求

- Python 3.8+
- PyTorch 2.0+（需要 `torch.compile` 和 Flash Attention）
- `tiktoken`（GPT‑2 分词器）
- `numpy`
- `wandb`（可选，用于日志记录）
- `transformers`（可选，仅用于加载预训练的 GPT‑2 权重）

安装依赖：
```bash
pip install torch numpy tiktoken wandb transformers
数据准备
脚本期望使用 nanoGPT 格式的 tokenized 数据。默认数据集为 openwebtext。你需要将 train.bin 和 val.bin 放在类似 data/openwebtext/ 的目录下。

可参照 nanoGPT 数据预处理步骤 准备 OpenWebText 数据集。你也可以修改代码以适应任何 tokenized 数据集。

训练
单 GPU（调试模式）
bash
python train.py --batch_size=32 --compile=False
多 GPU（DDP）单节点（4 张 GPU）
bash
torchrun --standalone --nproc_per_node=4 train.py
多节点训练（示例：2 节点，每节点 4 张 GPU）
节点 0（主节点）

bash
torchrun --nnodes=2 --node_rank=0 --master_addr=192.168.1.26 --master_port=1234 --nproc_per_node=4 train.py
节点 1

bash
torchrun --nnodes=2 --node_rank=1 --master_addr=192.168.1.26 --master_port=1234 --nproc_per_node=4 train.py
从检查点恢复训练
在 train.py 中设置 init_from = 'resume'，或通过命令行覆盖：

bash
python train.py --init_from=resume
主要训练超参数（位于 train.py）
参数	默认值	描述
batch_size	12	微批次大小（每张 GPU）
gradient_accumulation_steps	40	梯度累积步数（用于模拟更大的批次）
block_size	1024	上下文长度（每个序列的最大 token 数）
n_layer	12	Transformer 层数
n_head	12	注意力头数
n_embd	768	嵌入维度
learning_rate	6e-4	最大学习率（warmup + 余弦衰减）
max_iters	600000	总训练迭代次数
warmup_iters	2000	线性 warmup 步数
compile	True	使用 torch.compile 加速（需要 PyTorch 2.0+）
所有参数均可通过命令行覆盖：

bash
python train.py --n_layer=24 --n_head=16 --n_embd=1024 --batch_size=8
采样 / 生成
使用 sample.py 从训练好的模型生成文本。

基本用法（从 out/ 目录的检查点恢复）
bash
python sample.py --init_from=resume --out_dir=out --num_samples=5 --max_new_tokens=500
使用预训练的 GPT‑2 模型
bash
python sample.py --init_from=gpt2 --start="The meaning of life is"
选项说明
--start：提示字符串（或 FILE:prompt.txt 从文件加载）。

--num_samples：生成的独立样本数量。

--max_new_tokens：每个样本生成的 token 数量。

--temperature：采样温度（越高越随机）。

--top_k：Top‑k 采样（只保留概率最高的 k 个 token）。

--compile：启用 torch.compile 加速生成。

基准测试
运行 bench.py 测量模型吞吐量和 MFU（模型 FLOPs 利用率）。

bash
python bench.py --batch_size=12 --compile=True
默认使用 OpenWebText 数据集（真实数据）并执行一个简短的训练循环。设置 --real_data=False 可使用随机合成数据，从而隔离计算与 I/O 的影响。

启用 PyTorch profiler 并生成 TensorBoard 日志：

bash
python bench.py --profile=True
日志记录（Weights & Biases）
在 train.py 中设置 wandb_log = True，或通过命令行覆盖：

bash
python train.py --wandb_log=True --wandb_project=my_project --wandb_run_name=my_run
检查点
检查点保存在 out_dir（默认为 out/）中。每 eval_interval 次迭代，如果验证损失改善或 always_save_checkpoint 为 True，则保存检查点。

检查点包含模型状态字典、优化器状态、模型参数、迭代次数、最佳验证损失以及完整的配置。

自定义模型
通过 n_layer、n_head、n_embd、dropout、bias 修改架构。

默认词汇表大小为 50304（为了效率向上取整到 64 的倍数）。对于字符级训练，设置 dataset='shakespeare_char'，词汇表大小将变为 65。