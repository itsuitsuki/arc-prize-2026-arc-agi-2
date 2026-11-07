# Third-party notices

Our own code and notebook are released under [CC BY 4.0](LICENSE). This file lists what the notebook loads and the parts of our code that we adapted from others. Each of these keeps its own licence.

## Model weights

All weights the notebook loads are under the [Qwen Community License 1.0](licenses/QWEN_COMMUNITY_LICENSE.txt) (Copyright (c) 2026 Qwen), the licence of the base model, and every copy must keep its copyright and permission notice. The licence is not OSI-approved; section 2.5.a.3 of the competition rules exempts pretrained models with an incompatible licence from the open-source licence that the rules require for the submission.

| Component | Source | Kaggle input |
| --- | --- | --- |
| Qwen3.8-Flash-Next | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (revision de4b8e4) | used through the W4A16 checkpoint |
| W4A16 AutoRound checkpoint | [Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) | [dfranzen/intel-qwen3.8-flash-next-w4a16-autoround](https://www.kaggle.com/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround) (resharded, weights unchanged) |
| MTP drafter | RadixArk's MTP head from [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4), with its routed experts quantized to INT4 by albucino (`runtime/mtp-int4-g32` in [albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE)) | [dfranzen/albucino-qwen3-8-flash-next-drafter](https://www.kaggle.com/models/dfranzen/albucino-qwen3-8-flash-next-drafter) |
| FP8 n-gram table | RadixArk's FP8 table, as assembled by albucino (the `model-plefp8-*` shards of [albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE)) | [nisshokuitsuki/arc-fp8-ngram-table](https://www.kaggle.com/models/nisshokuitsuki/arc-fp8-ngram-table) |
| AZ-Focus weight override | 50 bf16 tensors of [AxonZeta/Qwen3.8-Flash-Next-AZ-Focus](https://huggingface.co/AxonZeta/Qwen3.8-Flash-Next-AZ-Focus) | [nisshokuitsuki/arc-az-focus-override](https://www.kaggle.com/models/nisshokuitsuki/arc-az-focus-override) |

## Engine

- **Merlin.** halt95's fork of [vLLM](https://github.com/vllm-project/vllm) v0.30.0, [halt95/merlin-engine](https://github.com/halt95/merlin-engine) at tag v2.5.2, under the [Apache License 2.0](licenses/APACHE-2.0.txt) with this [NOTICE](licenses/NOTICE-merlin-v2.5.2.txt). We built the wheel `vllm-0.30.0+merlin252.sm89` from that tag with one change: `vllm/models/qwen4_exp/nvidia/qsa.py` accepts compute capability 8.9 as well as 8.6 (`!= 86` became `not in (86, 89)`). We also rebuilt its two compiled extensions for sm89.
- **vLLM.** Apache License 2.0. It bundles `flash_linear_attention` under the MIT License (Copyright (c) 2023-2025, Songlin Yang, Yu Zhang).
- **FlashInfer.** [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) v0.6.18.post1, under the Apache License 2.0, with the NOTICE "FlashInfer / Copyright 2025-2026 NVIDIA / Copyright 2023-2026 FlashInfer community". We compiled `flashinfer-aot-sm89-sampling.so` and `flashinfer-aot-sm89-norm.so` for sm89 from the unmodified source.

## Python packages

The wheel dataset holds 199 wheels. Every wheel except the Merlin wheel is the file published on PyPI. [licenses/wheels.csv](licenses/wheels.csv) lists each wheel with its licence and source.

- 174 wheels are under permissive OSI licences (MIT, BSD, Apache-2.0, PSF-2.0, ISC, MPL-2.0 and others).
- pycountry 26.2.16 is under the LGPL-2.1, shipped unmodified.
- 24 wheels from NVIDIA's CUDA stack are under NVIDIA's own licences: the CUDA Toolkit EULA, the cuDNN and cuSPARSELt software license agreements, the NVIDIA SDK License Agreement (NVSHMEM 3.4.5), and the CUTLASS Python DSL EULA. Section 2.5.a.2 of the competition rules covers commercially available software such as these.
- Some wheels bundle other libraries under their own licences: opencv-python-headless and pynvvideocodec bundle FFmpeg, scipy and numpy bundle the GCC runtime libraries (GPL-3.0 with the GCC Runtime Library Exception), and Pillow bundles FreeType (FreeType License or GPL-2.0).

## Code adapted from others

The adapted parts remain under the licences below, and each says so in a comment where it starts.

| Part | Adapted from | Licence |
| --- | --- | --- |
| `pin_waiting_hit` in `solution_files/engine_patches/arc_qsa_host_kv.py` | `pin_waiting_hit` in Merlin's `kv_cache_manager.py` | Apache-2.0 |
| `solution_files/engine_patches/arc_last_state.py` | part of Merlin's `GDNRecoverSSMCommitContext.commit`, rewritten at run time | Apache-2.0 |
| `serve_command` in `submission.ipynb` | `scripts/serve-v2.5.sh` of Merlin v2.5.2 | Apache-2.0 |
| `solution_files/prompts/direct.txt` and `task.txt` | the prompt of [arcprize/arc-agi-benchmarking](https://github.com/arcprize/arc-agi-benchmarking) | [MIT](licenses/arc-agi-benchmarking-MIT.md) |
| the answer parsing in `submission.ipynb` | the answer parsing of [arcprize/arc-agi-benchmarking](https://github.com/arcprize/arc-agi-benchmarking) | [MIT](licenses/arc-agi-benchmarking-MIT.md) |

The other engine patches replace or wrap functions of vLLM 0.30 and Merlin at run time.

## Public work we credit

The agent's design follows these, but we did not copy code from them.

| Work | Licence |
| --- | --- |
| Adarsh's notebook [arc-qwen-3-8-27b-agent](https://www.kaggle.com/code/madarshbb/arc-qwen-3-8-27b-agent) | Apache 2.0 |
| Russell Kirk's notebook [arcagi2-qwen38-freeform-direct](https://www.kaggle.com/code/russcore/arcagi2-qwen38-freeform-direct) | Apache 2.0 |
| Tufa Labs' [Duck harness](https://github.com/Tufalabs/duck-harness) | the repository has no licence file |
