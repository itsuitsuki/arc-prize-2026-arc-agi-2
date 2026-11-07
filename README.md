# ARC Prize 2026 ARC-AGI-2: Qwen3.8-Flash-Next agent on 4 L4 GPUs

This repository holds our solution to the Kaggle competition [ARC Prize 2026 ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2). The solution is the Kaggle notebook `submission.ipynb`. On the four L4 GPUs that Kaggle gives a submission, it runs a public finetuned checkpoint of Qwen3.8-Flash-Next as an agent that writes and tests Python programs for each task. The notebook scored 77.22 on the public leaderboard when the competition closed.

[WRITEUP.md](WRITEUP.md) (also [writeup.pdf](writeup.pdf)) describes the method and the measurements.

## Files

| Path | Contents |
| --- | --- |
| `submission.ipynb` | The competition notebook. |
| `solution_files/` | The files of the Kaggle dataset [arc-solution-files](https://www.kaggle.com/datasets/nisshokuitsuki/arc-solution-files), which the notebook reads. |
| `solution_files/engine_patches/` | 13 Python modules that change the inference engine. The notebook adds them to the installed engine in the order of its `ENGINE_PATCHES` list. |
| `solution_files/sandbox.py` | Checks the model's Python programs and runs them in a separate process with limits on CPU time, files and output. |
| `solution_files/prompts/` | The system prompt, the task prompts, and the text added to the task prompt in later attempts. |
| `solution_files/tools.json` | The two tools the model can call: the Python executor and the answer submission. |
| `solution_files/reap448_keep.json` | The 448 of 512 routed experts that the notebook keeps in each layer. |
| `solution_files/qwen38_arc_16384.json` | The 16340 tokens that the drafter can propose. |
| `WRITEUP.md`, `writeup.pdf` | The writeup. |
| `THIRD_PARTY_NOTICES.md`, `licenses/` | The licences of the model weights, the engine, the Python wheels and the code we adapted. |
| `LICENSE` | CC BY 4.0. |

## Running the notebook on Kaggle

The notebook reads seven Kaggle inputs. Kaggle mounts each under `/kaggle/input/`, and the first code cell lists their paths and checks that each exists.

| Input | Kaggle page |
| --- | --- |
| Competition data | [arc-prize-2026-arc-agi-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data) |
| Engine and Python wheels | dataset [nisshokuitsuki/arc-merlin-wheels](https://www.kaggle.com/datasets/nisshokuitsuki/arc-merlin-wheels) |
| `solution_files/` of this repository | dataset [nisshokuitsuki/arc-solution-files](https://www.kaggle.com/datasets/nisshokuitsuki/arc-solution-files) |
| W4A16 checkpoint | model [dfranzen/intel-qwen3.8-flash-next-w4a16-autoround](https://www.kaggle.com/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround) |
| MTP drafter | model [dfranzen/albucino-qwen3-8-flash-next-drafter](https://www.kaggle.com/models/dfranzen/albucino-qwen3-8-flash-next-drafter) |
| FP8 n-gram table | model [nisshokuitsuki/arc-fp8-ngram-table](https://www.kaggle.com/models/nisshokuitsuki/arc-fp8-ngram-table) |
| AZ-Focus weight override | model [nisshokuitsuki/arc-az-focus-override](https://www.kaggle.com/models/nisshokuitsuki/arc-az-focus-override) |

The Kaggle notebook is [here](https://www.kaggle.com/code/nisshokuitsuki/arc-prize-2026-solution?scriptVersionId=365659112) (version 1, which scored 83.06 as a late submission). It uses Kaggle's paths, checks for four L4 GPUs, and installs engine wheels built for compute capability 8.9.

## Running the agent on other tasks

To answer tasks in a saved version, set `SCORING = True` in the first code cell and point `CHALLENGES` at a file in the format of the competition's challenge files, for example `arc-agi_evaluation_challenges.json` in the competition data. The run writes `/kaggle/working/submission.json`. To score it, compare it with `arc-agi_evaluation_solutions.json`: a test output counts as solved when one of its two attempts equals the solution.

## Licence

The notebook and `solution_files/` are under [CC BY 4.0](LICENSE), except the parts adapted from others. Those parts, the model weights, the engine and the wheels keep their own licences, which [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists.
