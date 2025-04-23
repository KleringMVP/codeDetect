# 🔍 基于复杂场景下的代码检测实证研究 + 基于统计指标的权重分配方案  
### Empirical Study on Code Detection in Complex Scenarios + A New Statistical-Weighted Detection Method

---

## 🧠 项目简介 | Project Overview

本项目旨在探索 **复杂提示场景（Prompt-based Complex Scenarios）下的代码检测能力**，并提出一种基于统计指标融合的新型检测方法。我们从两个层面展开工作：

- **实证研究：** 设计五种变体提示词，模拟现实中更复杂的生成情况，评估现有检测器在这些变体下的鲁棒性；
- **方法创新：** 结合统计特征，开发加权集成模型，提升检测准确率，最终构建一个可直接使用的检测工具 Demo。

---

This project explores the **robustness of code detectors in prompt-based complex generation scenarios**, and proposes a new statistical-weighted fusion method. It consists of:

- **Empirical Study:** We design five prompt variants to simulate real-world perturbations and test the resilience of existing detectors.
- **Method Innovation:** We build a weighted fusion model using statistical indicators to improve detection accuracy and offer an easy-to-use demo.

---

## 🧪 数据设计 | Prompt Design & Data

在 `post-processed data/` 文件夹中，我们设计了五类复杂提示（Prompt）变体，分别模拟以下生成扰动：

1. 增加时间复杂度（Higher time complexity）  
2. 修改变量命名为单一风格（Homogenized variable names）  
3. 增加冗余逻辑（Redundant logic added）  
4. 替换注释风格（Comment style modification）  
5. 改写结构但语义不变（Rephrased logic, same semantics）

---

All prompt variants are saved in JSONL format under the `post-processed data/` folder. Each variation tests how well detectors handle code obfuscation or manipulation while preserving core logic.

---

## 🛠️ 检测实证 | Detection Baseline Study

我们对当前主流代码检测模型（如 DetectGPT、LLMDet 等）进行了实证分析，结果保存在 `detectgpt4code/` 文件夹中，包含各个检测器在五类复杂提示下的表现对比。

---

Baseline detection results for each scenario are saved in the `detectgpt4code/` folder. We benchmark several detectors to understand their weaknesses under perturbed prompts.

---

## 🔧 新方法与集成 Demo | New Method & Integration Demo

下一步工作包括：

- **特征提取：** 从检测结果中提取关键统计指标（如 token 重复率、编辑距离、结构保真度等）；
- **加权策略：** 通过加权融合策略组合多个指标，形成更强鲁棒性的综合评分；
- **可用工具：** 实现一个命令行/网页 Demo，用户可上传代码，即可得出是否为生成代码的判定结果。

---

Planned developments:

- **Feature Extraction:** Use statistical indicators like token repetition rate, edit distance, and structural fidelity.
- **Weighted Fusion:** Apply weight assignment based on statistical metrics for a more robust judgment.
- **Accessible Demo:** Build a CLI or web demo where users can upload code to detect whether it's AI-generated.

---
