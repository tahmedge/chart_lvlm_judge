# LVLM Judge for Chart Model Evaluation

[![ACL 2025](https://img.shields.io/badge/ACL%202025-Industry%20Track-blue)]()
[![EMNLP 2025](https://img.shields.io/badge/EMNLP%202025-Industry%20Track-green)]()
[![License](https://img.shields.io/badge/License-Research-orange)]()

Official repository for two papers investigating **Large Vision-Language Models (LVLMs) as judges for chart comprehension and reasoning tasks**:

1. 📄 **"Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning?"** — *ACL 2025 Industry Track*
2. 📄 **"Deploying Tiny LVLM Judges for Real-World Evaluation of Chart Models: Lessons Learned and Best Practices"** — *EMNLP 2025 Industry Track*

---

## 📖 Overview

Evaluating LVLMs on chart-related tasks (chart QA, captioning, reasoning) typically relies on costly human annotation or expensive closed-source models like GPT-4. This repository provides:

- 🏛️ **A comprehensive evaluation framework** ("LVLM-as-a-Judge") for chart comprehension with clear pairwise & pointwise rubrics.
- 🧪 **Large-scale benchmarking** of 13+ open-source LVLMs (2B–9B) against GPT-4o and LLaVA-Critic-70B reference judgments.
- 🎯 **ChartJudge-2B** — a fine-tuned 2B-parameter LVLM judge specialized for chart evaluation, enabling cost-efficient deployment.
- 🔬 **Multi-criteria prompting** — an optimization strategy that fuses multiple evaluation dimensions into a single query.
- 📊 **Chart-Instruct-Eval** — a new benchmark for assessing instruction-following evaluation in chart tasks.

---

## 🌟 Highlights

### Paper 1: Judging the Judges (ACL 2025 Industry)
- First systematic study of state-of-the-art **open-source LVLMs as judges** across diverse chart benchmarks.
- Evaluated **13 LVLMs** (2B–9B parameters) including LLaVA-Critic, Qwen2-VL, InternLM-XComposer, mPLUG-Owl3, Phi-3.5-Vision, XGen-MM, Janus-Pro, MiniCPM-V, PaliGemma, ChartGemma, Idefics, and more.
- Over **100K judgment annotations** generated using GPT-4o and LLaVA-Critic-70B on OpenCQA and VisText datasets.
- In-depth analysis of **position bias, length bias, format adherence, and instruction-following**.

### Paper 2: Deploying Tiny LVLM Judges (EMNLP 2025 Industry)
- Proposes **two cost-efficient deployment strategies**: (i) multi-criteria prompting, (ii) domain-adaptive fine-tuning.
- Introduces **ChartJudge-2B** — a lightweight, fine-tuned 2B LVLM that rivals much larger judges on chart tasks.
- Demonstrates that **multi-criteria prompting breaks many 7B LVLMs** (including LLaVA-Critic), but ChartJudge-2B handles it robustly.
- **2× faster and 2× cheaper** than 7B LVLM judges (deployable on 8GB VRAM / T4 GPU).

---

## 📊 Datasets

| Dataset | Source | Task | Size | Purpose |
|---|---|---|---|---|
| **OpenCQA** | Kantharaj et al. (2022) | Open-ended chart QA | 1.1K test instances | Evaluation |
| **VisText (L1)** | Tang et al. (2023) | Chart captioning (structural) | 1.2K test instances | Evaluation |
| **VisText (L2/L3)** | Tang et al. (2023) | Chart captioning (insights) | 1.2K test instances | Evaluation |
| **Chart-Instruct-Eval** | *Ours (ACL 2025)* | Instruction-following evaluation | 400 samples | Evaluation |
| **ChartJudge Training Set** | *Ours (EMNLP 2025)* | Synthetic judgments from Chart-to-Text | ~9.7K (single-criterion) + ~2.8K (multi-criteria) | Fine-tuning |

Chart images can be downloaded from [here](https://drive.google.com/drive/folders/10HkZmjkTojIauKUb5B7BuEDj_cVS5_fc?usp=sharing).

---

## 🏗️ Evaluation Framework

### Evaluation Rubric

- **Pairwise**: Judge selects the better of two candidate responses.
- **Pointwise**: Judge rates a single response on a 1–5 Likert scale.
- **With / Without Reference**: Judge evaluates with or without a gold answer.

### Evaluation Criteria
- Factual Correctness
- Informativeness
- Relevance
- Multidimensional (overall quality)

### Evaluation Metrics
- Judgment Accuracy (pairwise)
- Error Distance (pointwise)
- Position Bias
- Length Bias
- Instruction-Following Evaluation Accuracy
- Format Adherence (JSON compliance)

---

## 🤖 Models Evaluated

**Open-source LVLMs (≤10B parameters):**

Qwen2-VL (2B / 7B), Phi-3.5-Vision-3.8B, XGen-MM-Phi3-3.8B, PaliGemma-3B, ChartGemma-3B, Janus-Pro-7B, InternLM-XComposer2d5-7B, LLaVA-Next-v1.6-Mistral-7B, LLaVA-Critic-7B, mPLUG-Owl3-7B, MiniCPM-V-2.6-8B, Idefics-9B-Instruct

**Reference Judges:** GPT-4o, LLaVA-Critic-70B

---

## 🏆 Key Results

### Paper 1: LVLM-as-a-Judge Performance (Avg. Pairwise Accuracy)

| Model | OpenCQA | VisText L1 | VisText L2/L3 |
|---|---|---|---|
| **LLaVA-Critic-7B** | **79.5%** | **79.1%** | **77.1%** |
| LLaVA-Next-v1.6-Mistral-7B | 75.9% | 75.1% | 75.1% |
| XGen-MM-Phi3-3.8B-Instruct | 71.6% | 75.4% | 70.7% |
| InternLM-XComposer2d5-7B | 64.5% | 72.0% | 75.6% |
| Qwen2-VL-7B-Instruct | 66.9% | 57.6% | 70.0% |
| PaliGemma-3B / ChartGemma-3B | 0.0% | 0.0% | 0.0% |
| Idefics-9B-Instruct | 20.3% | 20.9% | 24.3% |

### Paper 2: ChartJudge-2B vs. Baselines

| Model | OpenCQA (Pair ↑) | VisText L1 (Pair ↑) | VisText L2/L3 (Pair ↑) |
|---|---|---|---|
| Qwen2-VL-2B-Instruct (base) | 54.0% | 27.2% | 3.0% |
| **ChartJudge-2B (ours)** | **61.7%** | **64.6%** | **52.3%** |
| LLaVA-Critic-7B | 79.5% | 79.1% | 77.1% |

### Multi-Criteria Prompting Results (OpenCQA)

| Model | Single-Criterion (Pair ↑) | Multi-Criteria (Pair ↑) |
|---|---|---|
| LLaVA-Critic-7B | 76.78% | **0.00%** (fails) |
| LLaVA-Next-v1.6-Mistral-7B | 72.13% | **0.93%** (fails) |
| Qwen2-VL-7B-Instruct | 66.74% | 40.62% |
| **ChartJudge-2B (Multi-Criteria fine-tuned)** | 66.35% | **46.86%** |

---

## 🔑 Key Findings

- ✅ **Some 7B open-source LVLMs match GPT-4o level** judging performance (~80% agreement), making them viable for privacy-sensitive industrial settings.
- ⚠️ **Specialized chart models (ChartGemma, PaliGemma) fail as judges** — they do not generalize to judging tasks.
- 📉 **Multi-criteria prompting exposes severe fragility** — 7B LVLMs including specialized judges like LLaVA-Critic collapse to near-0% accuracy.
- 🎯 **ChartJudge-2B generalizes across datasets** even when distilled from a different LVLM (Gemini-1.5-Pro) than used at evaluation (GPT-4o / LLaVA-Critic-70B).
- 🧭 **LLaVA-Critic-70B correlates more strongly with human judgments than GPT-4o** (avg. error distance 0.81 vs. 0.93) — a strong open-source alternative for proprietary data.
- ⚖️ **Bias persists across all judges** — position bias and length bias remain challenges even for top performers.
- 💰 **ChartJudge-2B is 2× faster and 2× cheaper** than 7B judges (runs on 8GB VRAM / T4 GPU).
- 🔄 **Domain-adaptive fine-tuning even lifts extremely weak base models** — PaliGemma-3B jumps from 0% → 55.9% pairwise accuracy on VisText after fine-tuning.

---

## 📝 Citation

If you use this code, data, or models, please cite both papers:

```bibtex
@inproceedings{laskar2025judging,
  title     = {Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning?},
  author    = {Laskar, Md Tahmid Rahman and Islam, Mohammed Saidul and Mahbub, Ridwan and Masry, Ahmed
               and Rahman, Mizanur and Bhuiyan, Md Amran Hossen and Nayeem, Mir Tafseer
               and Joty, Shafiq and Hoque, Enamul and Huang, Jimmy Xiangji},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 6: Industry Track)},
  pages     = {1203--1216},
  year      = {2025},
  address   = {Vienna, Austria},
  publisher = {Association for Computational Linguistics}
}

@inproceedings{laskar2025deploying,
  title     = {Deploying Tiny LVLM Judges for Real-World Evaluation of Chart Models: Lessons Learned and Best Practices},
  author    = {Laskar, Md Tahmid Rahman and Islam, Mohammed Saidul and Mahbub, Ridwan
               and Rahman, Mizanur and Bhuiyan, Md Amran Hossen and Jahan, Israt
               and Nayeem, Mir Tafseer and Joty, Shafiq and Hoque, Enamul and Huang, Jimmy Xiangji},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track},
  pages     = {1906--1918},
  year      = {2025},
  publisher = {Association for Computational Linguistics}
}
```

---

## 👥 Authors

Md Tahmid Rahman Laskar¹, Mohammed Saidul Islam¹, Ridwan Mahbub¹, Ahmed Masry¹, Mizanur Rahman¹, Md Amran Hossen Bhuiyan¹, Israt Jahan¹, Mir Tafseer Nayeem², Shafiq Joty³, Enamul Hoque¹, Jimmy Xiangji Huang¹

¹ York University  ² University of Alberta  ³ Salesforce AI Research

📧 Contact: `{tahmedge, enamulh, jhuang}@yorku.ca`

---

## 🙏 Acknowledgments

This research is supported by the Natural Sciences and Engineering Research Council (NSERC) of Canada, the York Research Chairs (YRC) program, the Canada Foundation for Innovation (CFI), Google's Gemini Academic Program, and Compute Canada.

---

## ⚖️ Ethical Considerations

All models in this repository are used **only as judges** to evaluate existing LVLM-generated responses, not to generate content about people or sensitive topics. All datasets are publicly available. Human annotations were conducted by the authors, with expertise in NLP and Computer Vision. Licensing requirements are maintained for all third-party artifacts (OpenAI, Gemini, Anthropic, HuggingFace).

---




