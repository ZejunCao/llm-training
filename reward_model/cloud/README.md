<div align="center">

# 🌩️ Cloud Reward Model Implementation

**基于 Cloud 论文的奖励模型训练实现与改进**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2408.11791)
[![Original](https://img.shields.io/badge/Original-GitHub-blue)](https://github.com/zankner/CLoud)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

*基于 Cloud 论文的完整训练实现，包含数据处理、模型训练和优化改进*

</div>

---

## 📖 项目简介

本项目是基于 [Cloud 论文](https://arxiv.org/abs/2408.11791) 的完整实现，提供了从数据收集到模型训练的完整流程。Cloud 是一种生成式奖励模型，对模型回答先生成一段批判文本，然后将其拼接回模型，通过一个奖励头（线性层）获取奖励分数。

### 🎯 核心特性

- **完整训练流程**: 从数据收集到最终模型的三阶段训练
- **数据处理工具**: 提供批判数据收集、清洗和重组脚本
- **训练优化**: 针对显存和性能的优化改进
- **实用技巧**: 基于实验总结的最佳实践

### 🏗️ 训练架构

![Cloud Architecture](https://s3.bmp.ovh/imgs/2025/08/25/8c276c3d022c226b.png)

Cloud 模型训练分为三个阶段：

1. **批判学习阶段**: 首先设计批判 prompt，用更大的模型对 question+response 生成批判，然后将 question+response 作为输入，生成的批判作为输出，使用 sft 训练 cloud 模型，让模型学会生成批判。

2. **数据重构阶段**: 用 cloud 模型对数据集重新生成批判，用于替换原始数据集中的批判。

3. **奖励训练阶段**: 用自生成的批判和偏好数据训练模型，采用 SFT+RM 混合 Loss 训练，此阶段还引入 SFT Loss 目的是为了保持批判能力，防止只拟合奖励分数。

![Training Process](https://s3.bmp.ovh/imgs/2025/08/25/0267e0a72a51c29d.png)

---

## 🚀 快速开始

### 环境要求

- Python 3.11
- PyTorch 2.5.1

### 安装依赖

#### 使用 uv 安装（推荐）

```bash
# git 下载项目之后进入主目录
uv sync  # 同步安装依赖
# 或
uv add -r requirements.txt
```

#### 使用 pip 安装

```bash
pip install -r requirements.txt
```

---

## 📚 训练流程概览

整个训练流程分为五个阶段：

1. 📝 **收集原始批判**：使用另外的大模型生成原始批判，用于蒸馏批判供小模型学习

2. 🎓 **第一阶段 SFT 训练**：用原始批判数据进行 SFT。

3. 🤖 **获取自生成批判**：部署训练好的 SFT 模型，生成自批判数据替换原始批判。

4. 🏆 **奖励模型训练**：引入奖励头，联合训练 SFT + Reward Loss。

5. 🚀 **推理部署**：最终部署，应用到实际任务场景。

## 🛠️ 完整训练流程

注意：以下除了两阶段训练之外，中间的数据处理和权重重组代码均在 `数据处理_权重重组.ipynb` 中。

### 1. 收集原始批判

首先需要从规模更大的模型中蒸馏出批判，然后训练规模较小的奖励模型。

首先可以下载一个数据集，或者使用自己业务的偏好数据集，这里使用 `Skywork-Reward-Preference-80K-v0.1`:
```bash
pip install modelscope

modelscope download --dataset Skywork/Skywork-Reward-Preference-80K-v0.1 --local_dir reward_model/data/Skywork-Reward-Preference-80K-v0.1
```

* **部署大模型服务**: 下载规模更大的模型，或者使用 api，这里使用 `Qwen2.5-72B-Instruct`，下载到本地后使用 vllm 部署，注意替换python环境和模型地址。

    > **重要提示**: `--max-model-len`需要限制一下，实验发现模型可能会出现重复生成的情况，持续让他生成到最大长度会极大拖慢推理时间，且数据无意义后期还需切除。在实际训练时长度过长对显存压力也较大，我这里只使用2048长度训练，可根据自己需求调整。`max-model-len`代表数据总长度，新生成长度可在调用时设置。

    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --port 5001 \
        --served-model-name qwen25 \
        --model Qwen/Qwen2.5-72B-Instruct \
        --tensor_parallel_size 4 \
        --gpu-memory-utilization 0.9 \
        --max-model-len 2048
    ```

* **多进程推理**: 当数据量很大时，推理生成原始批判速度较慢，此时可以部署多个节点，使用 `multi_process_reason_for_vllm.py` 脚本多进程推理，已在 jupyter 中调用。

* **数据清洗**: 得到的批判需要反复清洗，这步很关键，批判数据越脏模型越难学习。本仓库根据数据情况创建启发式规则，包括json结构判断、token长度判断、不完整句式判断等。

### 2. 第一阶段 SFT 训练

第一阶段就是常规的 SFT 训练，学习如何对数据进行批判，可直接使用 `sh train_sft1.sh` 脚本训练。

* **训练周期**：通常模型在 2 epoch 左右能收敛到最低，第 3 个 epoch 开始过拟合。

* **权重重组**：trainer只保存了权重，直接用 vllm 部署缺少一些 tokenizer 的文件（应该也有一些别的办法），所以需要将原始模型的一些文件与训练权重结合。

### 3. 获取自生成批判

重组权重之后用vllm部署，将新生成的子批判替换原始批判得到新数据集，并将其成对整理成偏好数据格式，用于第二阶段训练。

* **目的**：这是为了使模型更符合推理场景，防止更大规模的模型的一些批判小模型学不会，导致强行拟合效果变差。此外，在训练的时候还要同时引入 sft+reward 两种损失，防止只拟合奖励分数。

### 4. 奖励模型训练

第一阶段sft训练完之后模型已经具备了批判的能力，现在需要进行第二阶段训练奖励头了，可直接使用 `sh train_sft2.sh` 脚本训练。

* **vllm适配**：为了使奖励头部分能适配 vllm 加载，需要将其命名为 `score` 或 `classify`。

* **训练**：通常训练 2 epoch 就可以，且该阶段由于双塔结构，消耗显存较高，可降低 batch_size 大小。

### 5. 推理部署

vllm 支持判别式奖励模型部署，cloud 模型的奖励头部分与其类似，所以可分别部署生成部分和奖励头部分，然后串行获取奖励分数。

* **性能**：经测试比直接 transformers 加载快得多（ray 部署），奖励头：生成模型 显存占比可在 1：8 左右。

* **部署**：修改 `cloud_infer_start.sh` 中的参数，然后运行 `sh cloud_infer_start.sh` 启动两个模型和fastapi服务，使用 jupyter 脚本调用测试。

* **vllm改造**：目前尝试过改造vllm，使得先生成批判，然后再经过线性层得到奖励分数，但没有成功，有厉害的大神可以尝试一下。

---

## 🎯 实验总结与最佳实践

### 原始批判 Prompt 设计技巧

我做了很多次实验，总结了以下原始批判prompt设计技巧：

1. **不建议单独输入chosen和rejected去独立生成批判**，这样得到的某些批判，人都很难区分哪个好哪个差，包括cloud论文中附录的批判prompt，经测试后效果一般。

2. **不要过于极端的批判**，例如对chosen过度的夸奖，rejected过度的批评，这样模型很难学到正常的批判，往往第一个token模型就认定是好还是差了。

3. **同时输入chosen和rejected**，让模型了解到这两种的差别，然后对chosen夸的多一点，对rejected批评的多一点，这样有好有坏的批判模型才能学会。

### 性能优化建议

- **显存管理**: 根据GPU显存调整 batch_size，特别是第三阶段的双塔结构
- **长度控制**: 限制 max_model_len 避免重复生成，建议使用2048长度
- **多进程推理**: 大规模数据使用 `multi_process_reason_for_vllm.py` 并行处理
- **数据清洗**: 批判数据质量直接影响模型效果，需要严格清洗

---

## 📁 项目结构

```
reward_model/cloud/
├── train_sft1.py                    # 第一阶段训练脚本
├── train_sft2.py                    # 第二阶段训练脚本
├── train_sft1.sh                    # 第一阶段训练配置
├── train_sft2.sh                    # 第二阶段训练配置
├── 数据处理_权重重组.ipynb             # 数据处理与权重重组脚本
├── prompt.py                        # 批判提示词
├── multi_process_reason_for_vllm.py # 多进程推理脚本
├── cloud_vllm_api.py                # 模型推理服务部署脚本
├── cloud_infer_start.sh             # vllm模型启动脚本
├── arguments.py                     # 训练参数配置
├── utils.py                         # 工具函数
├── deepspeed/                       # DeepSpeed配置
└── data/                            # 数据
```

---

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目基于 MIT 许可证开源 - 查看 [LICENSE](../../LICENSE) 文件了解详情。

---

## 🙏 致谢

- 感谢 [Cloud 论文](https://arxiv.org/abs/2408.11791) 的作者们提供的研究基础
- 感谢 [Skywork](https://modelscope.cn/datasets/Skywork/Skywork-Reward-Preference-80K-v0.1) 提供的数据集
- 感谢开源社区的支持

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐️**

</div>