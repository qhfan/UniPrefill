# UniPrefill: Universal Long-Context Prefill Acceleration via Block-wise Dynamic Sparsification

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-paper-red?logo=arxiv" alt="arXiv"></a>
  <a href="#"><img src="https://img.shields.io/badge/vLLM-v0.16.0-blue?logo=github" alt="vLLM version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
</p>

## Abstract

As large language models (LLMs) continue to advance rapidly, they are becoming increasingly capable while simultaneously demanding ever-longer context lengths. To improve the inference efficiency of long-context processing, several novel low-complexity hybrid architectures have recently been proposed, effectively alleviating the computational burden of long-context inference. However, existing research on long-context prefill acceleration remains predominantly focused on sparse attention mechanisms, which achieve their maximum speedup only on full-attention models. When transferred to emerging architectures — such as linear/full attention hybrids or sliding window/full attention hybrids — these prefill acceleration approaches suffer significant performance degradation. Furthermore, such methods are generally incompatible with continuous batching, making them difficult to integrate into modern inference engines such as vLLM.

To this end, we propose **UniPrefill**, a prefill acceleration framework applicable to virtually any model architecture, which directly accelerates the model's computation at the token level. We further implement UniPrefill as a continuous batching operator and extend vLLM's scheduling strategy to natively support prefill-decode co-processing and tensor parallel for UniPrefill, enabling its seamless integration into vLLM. UniPrefill achieves up to **2.1× speedup** in Time-To-First-Token (TTFT), with the acceleration becoming increasingly pronounced as the number of concurrent requests grows.

---

## Key Features

- **Architecture-agnostic prefill acceleration** — works on full-attention, linear/full hybrids, and sliding-window/full hybrids, unlike sparse-attention-only approaches
- **Continuous batching compatible** — implemented as a drop-in batching operator, natively supported by vLLM's scheduling pipeline
- **Tensor parallel support** — UniPrefill's scheduling strategy is extended to support multi-GPU tensor parallelism
- **Prefill-decode co-processing** — simultaneous prefill and decode within the same engine, improving GPU utilization
- **Up to 2.1× TTFT speedup** — gains grow with the number of concurrent requests

---

## Installation

This implementation is based on **vLLM v0.16.0**.

~~~bash
git clone https://github.com/qhfan/UniPrefill.git
pip install -r requirements.txt
cd UniPrefill/vllm-releases-v0.16.0
bash setup.sh
~~~

> **Note:** We recommend using a clean conda environment with Python 3.10+ and CUDA 12.1+ before running `setup.sh`.

---

## Supported Models

| Model Family | File Modified |
|---|---|
| LLaMA-3.1 | `vllm/model_executor/models/llama.py` |
| Qwen3-Next | `vllm/model_executor/models/qwen3_next.py` |
| Gemma3 | `vllm/model_executor/models/gemma3.py` |

---

## Code Changes Overview

UniPrefill's modifications to the vLLM codebase are minimal and well-contained. The key changed files are listed below:

| File | Description |
|---|---|
| `vllm/model_executor/layers/fused_top_p_selection_tp_pd.py` | Core UniPrefill operator: block-wise dynamic sparsification with tensor parallel and prefill-decode support |
| `vllm/model_executor/models/llama.py` | LLaMA-3.1 model integration |
| `vllm/model_executor/models/qwen3_next.py` | Qwen3-Next model integration |
| `vllm/model_executor/models/gemma3.py` | Gemma3 model integration |
| `vllm/v1/attention/backends/flash_attn.py` | Modified `FlashAttnImpl.forward` and KV cache update logic |
| `vllm/v1/attention/backends/triton_attn.py` | Modified `TritonAttnImpl.forward` and KV cache update logic |
| `vllm/v1/worker/gpu_model_runner.py` | Per-request per-layer sequence length tracking across requests |
| `vllm/forward_context.py` | Per-layer token count variable maintained across the forward pass |
| `native_imp.py` | PyTorch reference implementation for algorithm illustration only; supports `batch_size == 1` solely |

---

## ⚠️ Known Issues

> **Warning:** When running with tensor parallelism (`tp > 1`), there is an intermittent bug that may cause the model to output repeated `!` characters (e.g., `!!!!!!!!!!!!!`). The root cause is under investigation. Please use `tp > 1` with caution in production environments and validate outputs when deploying at scale.

---

## Citation

If you find UniPrefill useful in your research, please cite our paper:

~~~bibtex
@article{uniprefill2026,
  title     = {UniPrefill: Universal Long-Context Prefill Acceleration via Block-wise Dynamic Sparsification},
  author    = {Qihang Fan, Huaibo Huang, Zhiying Wu, Bingning Wang, Ran He},
  journal   = {arXiv preprint arXiv:2605.06221},
  year      = {2026}
}
~~~

---

## Acknowledgements

This project builds on top of [vLLM](https://github.com/vllm-project/vllm). We thank the vLLM team for their excellent open-source inference engine.
