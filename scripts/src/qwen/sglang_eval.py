"""Task-unit direct answers on eval-120 from an SGLang server, written like the vLLM arms (one file per test output:
request_outputs, prompt/completion/thinking tokens, finish reason, prediction), so analyze_arms compares them output by
output. Same prompt (src.qwen.direct.task_prompt through the model's chat template), sampling (T 0.7, top-p 0.95,
top-k 20), 64K thinking budget with the answer forced after it (src.qwen.direct.budgeted) and per-task seeds as
q38_direct64k_task_20261011. The server runs from config["python"] (an env with sglang) on config["tp"] GPUs (default 1)
with config["server_args"]. config["attempt"] (default 0) labels the result files; with config["second_view"] a later
attempt sees the whole task under the fixed non-identity D8 symmetry of its first output (src.qwen.direct.second_view,
as the runtime's job_unit "task" second attempts and q38_direct64k_taskview2_20261011) and its grids are mapped back.
    python sglang_eval.py CONFIG.json"""
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from src.qwen.direct import answer_grids, atomic_json, second_view, sglang_completion, task_prompt
from src.utils.grid import View, apply_task, invert_grid


def main(path: str) -> None:
    from transformers import AutoTokenizer

    config = json.loads(Path(path).read_text())
    output = Path(config["output"])
    (output / "results").mkdir(parents=True, exist_ok=True)
    tasks = json.loads(Path(config["challenges"]).read_text())
    if any("output" in test for task in tasks.values() for test in task["test"]):
        raise ValueError("test labels must not reach the prompts")
    attempt = config.get("attempt", 0)
    jobs = [{"task_id": task_id, "attempt": attempt, "seed": config["seed"] + number} for number, task_id in enumerate(tasks)]
    shard, shards = config["shard"]
    jobs = [job for number, job in enumerate(jobs) if number % shards == shard
            and not (output / "results" / f"{job['task_id']}_0_{attempt}.json").exists()]
    tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer", config["model"]))
    think_start, think_end = tokenizer.convert_tokens_to_ids("<think>"), tokenizer.convert_tokens_to_ids("</think>")
    port = config["port"]
    server = subprocess.Popen([config["python"], "-m", "sglang.launch_server", "--model-path", config["model"],
                               "--host", "127.0.0.1", "--port", str(port), "--tp-size", str(config.get("tp", 1)),
                               "--context-length", str(config["max_model_len"]),
                               "--max-running-requests", str(config["concurrency"]), "--skip-server-warmup",
                               *config.get("server_args", [])],
                              stdout=open(output / f"server_{shard}.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(trust_env=False) as http:
            for _ in range(1800):
                try:
                    if http.get(base + "/health", timeout=5).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if server.poll() is not None:
                    raise RuntimeError(f"sglang server exited with {server.returncode}")
                # a failed scheduler can leave the launcher alive (its cleanup may hit /proc permissions)
                if "Received sigquit from a child process" in (output / f"server_{shard}.log").read_text(errors="replace"):
                    raise RuntimeError("sglang scheduler failed at startup (see the server log)")
                time.sleep(2)
            else:
                raise RuntimeError("sglang server not ready after 1 h")
            settings = {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "thinking_token_budget": config["thinking_token_budget"],
                        "request_timeout": 6 * 3600}

            def one(job):
                try:
                    answer_one(job)
                except Exception as error:  # one failed request must not end the shard
                    print(f"{job['task_id']}: {type(error).__name__}: {error}", flush=True)

            def answer_one(job):
                task, view = tasks[job["task_id"]], None
                if attempt and config.get("second_view"):
                    transpose, k = second_view(job["task_id"], 0)
                    view = View(transpose, k, tuple(range(10)), tuple(range(len(task["train"]))))
                    task = apply_task(task, view)
                text = tokenizer.apply_chat_template([{"role": "user", "content": task_prompt(task)}],
                                                     tokenize=False, add_generation_prompt=True, enable_thinking=True)
                prompt = tokenizer.encode(text, add_special_tokens=False)
                started = time.time()
                tokens, finish = sglang_completion(http, base + "/generate", prompt, settings, job["seed"],
                                                   config["max_tokens"], think_start, think_end)
                end = tokens.index(think_end) if think_end in tokens else None
                answer = tokenizer.decode(tokens[end + 1:] if end is not None else tokens, skip_special_tokens=False)
                count = len(task["test"])
                for index, grid in enumerate(answer_grids(answer, count)):
                    grid = invert_grid(grid, view) if view and grid is not None else grid
                    atomic_json(output / "results" / f"{job['task_id']}_{index}_{attempt}.json", {
                        **job, "test_index": index, "request_outputs": count, "prompt_tokens": len(prompt),
                        "completion_tokens": len(tokens), "thinking_tokens": end, "finish_reason": finish, "prediction": grid,
                        "answer_text": answer[-4000:], "seconds": round(time.time() - started, 1)})

            with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
                list(pool.map(one, jobs))
    finally:
        os.killpg(server.pid, signal.SIGTERM)  # the scheduler and detokenizer are children of the launcher
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)


if __name__ == "__main__":
    main(sys.argv[1])
