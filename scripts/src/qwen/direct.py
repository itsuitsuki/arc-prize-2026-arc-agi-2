"""Direct (transductive) answers: the model thinks and writes the test output grid, without tools.

The prompt and the answer parsing follow arcprize/arc-agi-benchmarking (prompts/system_prompt.txt with
convert_task_pairs_to_prompt, sent as the user message; utils/parsing.py: \\boxed{} first, else the last bracketed JSON
list of lists). The induction deployment (src/code/native_tool_deployment.py) runs these before its tool trajectories
and ranks their grids below verified programs and above unverified ones.
"""
import hashlib
import json
from pathlib import Path
import re
import time

from src.utils.grid import D8, View, apply_task, invert_grid, is_valid_grid

DIRECT_PROMPT = (
    "You are participating in a puzzle solving competition. You are an expert at solving puzzles.\n\n"
    "Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation "
    "in the training examples that maps the input to the output, then apply that pattern to the test input to give a "
    "final output.\n\nRespond in the format of the training output examples\n\n--Training Examples--\n"
    "{training_examples}\n--End of Training Examples--\n\n--Test Input--\n{test_input}\n--End of Test Input--\n\n"
    "Your response:")
# One request for all test inputs of a task (job_unit "task"): the same instructions, every test input in order, and one
# labelled output per test input. A task with one test input gets DIRECT_PROMPT unchanged.
TASK_PROMPT = (
    "You are participating in a puzzle solving competition. You are an expert at solving puzzles.\n\n"
    "Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation "
    "in the training examples that maps the input to the output, then apply that pattern to each test input to give its "
    "final output.\n\nRespond in the format of the training output examples, with one output per test input, in order, "
    "each after its own line \"--Test Output k--\" (k = 1, 2, ...)\n\n--Training Examples--\n"
    "{training_examples}\n--End of Training Examples--\n\n{test_inputs}\n\nYour response:")
SEED_OFFSET = 10_000_000  # keeps direct-answer seeds apart from the tool trajectories' seeds


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def format_grid(grid: list, grid_format: str = "json") -> str:
    """"json": the benchmark's json.dumps (about 3 tokens per cell); "compact": one line of digits per row (1 token per cell)."""
    if grid_format == "compact":
        return "\n".join("".join(str(cell) for cell in row) for row in grid)
    return json.dumps(grid)


def direct_prompt(task: dict, index: int, grid_format: str = "json") -> str:
    examples = "".join(f"--Example {i}-- \n\n INPUT: \n\n{format_grid(pair['input'], grid_format)}\n\nOUTPUT: \n\n"
                       f"{format_grid(pair['output'], grid_format)}\n\n" for i, pair in enumerate(task["train"]))
    return DIRECT_PROMPT.format(training_examples=examples, test_input=format_grid(task["test"][index]["input"], grid_format))


def task_prompt(task: dict, grid_format: str = "json") -> str:
    """The prompt for all test inputs of a task at once (DIRECT_PROMPT when there is one)."""
    if len(task["test"]) == 1:
        return direct_prompt(task, 0, grid_format)
    examples = "".join(f"--Example {i}-- \n\n INPUT: \n\n{format_grid(pair['input'], grid_format)}\n\nOUTPUT: \n\n"
                       f"{format_grid(pair['output'], grid_format)}\n\n" for i, pair in enumerate(task["train"]))
    tests = "\n\n".join(f"--Test Input {k}--\n{format_grid(test['input'], grid_format)}\n--End of Test Input {k}--"
                         for k, test in enumerate(task["test"], 1))
    return TASK_PROMPT.format(training_examples=examples, test_inputs=tests)


def all_grids(text: str) -> list:
    """Every top-level bracketed JSON list of lists that is a valid ARC grid, in order."""
    grids, depth, start = [], 0, None
    for position, char in enumerate(text):
        if char == "[":
            if depth == 0:
                start = position
            depth += 1
        elif char == "]" and depth:
            depth -= 1
            if depth == 0:
                try:
                    grid = json.loads(text[start:position + 1])
                except json.JSONDecodeError:
                    continue
                if is_valid_grid(grid):
                    grids.append(grid)
    return grids


def answer_grids(text: str, count: int, grid_format: str = "json") -> list:
    """`count` grids for the test outputs of a task answer: the grid after each "--Test Output k--" line, else (no
    labels) the last `count` grids in order; a missing one is None."""
    parts = re.split(r"--\s*Test Output\s*(\d+)\s*--", text)
    found = {}
    for label, body in zip(parts[1::2], parts[2::2]):
        found[int(label)] = answer_grid(body, grid_format)
    if found:
        return [found.get(k) for k in range(1, count + 1)]
    if grid_format == "compact":
        return [answer_grid(text, grid_format)] + [None] * (count - 1) if count else []
    grids = all_grids(text)[-count:] if count else []
    return grids + [None] * (count - len(grids))


def last_digit_grid(text: str) -> list | None:
    """The last block of consecutive lines that are each only digits, all of one length (fences and blank lines end a
    block); kept only when it is a valid ARC grid."""
    rows, blocks = [], []
    for line in text.splitlines() + [""]:
        line = line.strip().strip("`")
        if line.isdigit() and (not rows or len(line) == len(rows[0])):
            rows.append(line)
        else:
            if rows:
                blocks.append(rows)
            rows = [line] if line.isdigit() else []
    for rows in reversed(blocks):
        grid = [[int(cell) for cell in row] for row in rows]
        if is_valid_grid(grid):
            return grid
    return None


def last_grid(text: str) -> list | None:
    """\\boxed{...} first, else the last bracketed JSON list of lists; kept only when it is a valid ARC grid."""
    candidates = [match.group(1) for match in re.finditer(r"\\boxed\{(.*?)\}", text, re.DOTALL)][:1]
    end = text.rfind("]")
    if end >= 0:
        depth = 0
        for start in range(end, -1, -1):
            depth += {"]": 1, "[": -1}.get(text[start], 0)
            if depth == 0:
                candidates.append(text[start:end + 1])
                break
    for candidate in candidates:
        try:
            grid = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if is_valid_grid(grid):
            return grid
    return None


def answer_grid(text: str, grid_format: str = "json") -> list | None:
    """last_grid; with the compact format the last digit block first, then last_grid."""
    if grid_format == "compact":
        grid = last_digit_grid(text)
        if grid is not None:
            return grid
    return last_grid(text)


def parse_answer(tokenizer, tokens: list[int], grid_format: str = "json") -> tuple[str, list | None, int | None]:
    """(answer text after </think>, its grid or None, the index of </think> or None) for a completion's token ids;
    the deployment and the RL reward (src/rl/verl/direct_loop.py) both parse with this."""
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    end = tokens.index(think_end) if think_end in tokens else None
    answer = tokenizer.decode(tokens[end + 1:] if end is not None else tokens, skip_special_tokens=False,
                              clean_up_tokenization_spaces=False)
    return answer, answer_grid(answer, grid_format), end


def second_view(task_id: str, index: int) -> list:
    """[transpose, k] of a fixed non-identity D8 symmetry per test output (from a hash of task and output index)."""
    return list(D8[1 + int(hashlib.sha256(f"{task_id}_{index}".encode()).hexdigest(), 16) % 7])


def direct_jobs_for(tasks: dict, settings: dict) -> list[dict]:
    """settings["direct_attempts"] (default 0) direct jobs per test output, attempt-major. With settings["second_view"],
    attempts after the first carry "view": the task is shown under that symmetry and the answer mapped back."""
    jobs = []
    for attempt in range(settings.get("direct_attempts", 0)):
        for task_id, task in tasks.items():
            # job_unit "task": one job for all test inputs of a task (test_index None, task_prompt)
            for index in [None] if settings.get("job_unit") == "task" else range(len(task["test"])):
                jobs.append({"key": f"d{len(jobs):04d}", "task_id": task_id, "test_index": index, "attempt": attempt,
                             "seed": settings["seed"] + SEED_OFFSET + len(jobs)})
                if attempt and settings.get("second_view"):
                    jobs[-1]["view"] = second_view(task_id, 0 if index is None else index)
    return jobs


def sglang_completion(http, url: str, prompt: list[int], settings: dict, seed: int, max_tokens: int, think_start: int,
                      think_end: int) -> tuple[list[int], str]:
    """(token ids, finish reason) from SGLang's /generate with vLLM's thinking_token_budget behaviour (budgeted)."""
    def generate(ids: list[int], limit: int) -> tuple[list[int], str]:
        body = {"input_ids": ids, "sampling_params": {
            "temperature": settings["temperature"], "top_p": settings["top_p"], "top_k": settings["top_k"],
            "max_new_tokens": limit, "skip_special_tokens": False, "sampling_seed": seed}}
        response = http.post(url, json=body, timeout=settings["request_timeout"])
        response.raise_for_status()
        out = response.json()
        return list(out["output_ids"]), out["meta_info"]["finish_reason"]["type"]

    return budgeted(generate, prompt, settings, max_tokens, think_start, think_end)


def llamacpp_completion(http, url: str, prompt: list[int], settings: dict, seed: int, max_tokens: int, think_start: int,
                        think_end: int) -> tuple[list[int], str]:
    """The same from llama.cpp's /completion (prompt as token ids). llama.cpp's default sampler chain differs from vLLM's
    (min_p 0.05, top-k and top-p before the temperature), so the chain is set to vLLM's: temperature, then top-k and
    top-p on the scaled distribution, nothing else."""
    def generate(ids: list[int], limit: int) -> tuple[list[int], str]:
        body = {"prompt": ids, "n_predict": limit, "temperature": settings["temperature"], "top_k": settings["top_k"],
                "top_p": settings["top_p"], "min_p": 0.0, "samplers": ["temperature", "top_k", "top_p"], "seed": seed,
                "return_tokens": True, "cache_prompt": True}
        response = http.post(url, json=body, timeout=settings["request_timeout"])
        response.raise_for_status()
        out = response.json()
        return list(out["tokens"]), "length" if out.get("stop_type") == "limit" else "stop"

    return budgeted(generate, prompt, settings, max_tokens, think_start, think_end)


def budgeted(generate, prompt: list[int], settings: dict, max_tokens: int, think_start: int,
             think_end: int) -> tuple[list[int], str]:
    """vLLM's thinking_token_budget behaviour for engines without it, over generate(ids, limit) -> (new ids, finish):
    tokens after the prompt's last <think> count toward the budget, and when they reach it </think> is forced and the
    answer continues up to max_tokens. The first request stops at the budget; if it ends there, a second request
    continues from prompt + output (+ </think> when thinking is still open)."""
    budget = settings.get("thinking_token_budget")
    opened = prompt[::-1].index(think_start) if think_start in prompt else None  # tokens after the last <think>
    first = max_tokens if budget is None or opened is None else min(max_tokens, max(0, budget - opened))
    tokens, finish = generate(prompt, first)
    if finish == "length" and len(tokens) < max_tokens:
        forced = [] if think_end in tokens else [think_end]
        rest, finish = generate(prompt + tokens + forced, max_tokens - len(tokens) - len(forced))
        tokens = tokens + forced + rest
    return tokens, finish


def direct_answer(job: dict, task: dict, root: Path, model: str, settings: dict, tokenizer, client) -> dict:
    """One completion with the agent's sampling, thinking budget and token cap; writes root/<key>.json. With
    settings["max_model_len"] the token cap shrinks so prompt plus answer fit the server's context. With
    settings["engine"] == "sglang", client is an httpx client for settings["generate_url"] (sglang_completion)."""
    if any("output" in test for test in task["test"]):
        raise ValueError("Test labels must not enter the prompt")
    view = View(job["view"][0], job["view"][1], tuple(range(10)), tuple(range(len(task["train"])))) if job.get("view") else None
    task = apply_task(task, view) if view else task
    grid_format = settings.get("grid_format", "json")
    content = task_prompt(task, grid_format) if job["test_index"] is None else direct_prompt(task, job["test_index"], grid_format)
    text = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=True)
    prompt = tokenizer.encode(text, add_special_tokens=False)
    extra = {"top_k": settings["top_k"], "return_token_ids": True, "add_special_tokens": False, "skip_special_tokens": False}
    if settings.get("thinking_token_budget") is not None:
        extra["thinking_token_budget"] = settings["thinking_token_budget"]
    result, started = {**job, "status": "RUNNING", "prediction": None}, time.monotonic()
    max_tokens = min(settings["max_tokens"], settings.get("max_model_len", 1 << 30) - len(prompt))
    try:
        if settings.get("engine") == "sglang":
            tokens, finish = sglang_completion(client, settings["generate_url"], prompt, settings, job["seed"], max_tokens,
                                               tokenizer.convert_tokens_to_ids("<think>"), tokenizer.convert_tokens_to_ids("</think>"))
        else:
            response = client.completions.create(
                model=model, prompt=prompt, n=1, temperature=settings["temperature"], top_p=settings["top_p"],
                max_tokens=max_tokens, seed=job["seed"], extra_body=extra, timeout=settings["request_timeout"]).model_dump()
            choice = response["choices"][0]
            tokens, finish = choice.get("token_ids") or [], choice.get("finish_reason")
        answer, grid, thinking = parse_answer(tokenizer, tokens, grid_format)
        grids = [grid] if job["test_index"] is not None else answer_grids(answer, len(task["test"]), grid_format)
        grids = [invert_grid(g, view) if view and g is not None else g for g in grids]
        if job["test_index"] is not None:
            result["prediction"] = grids[0]
        else:
            result["predictions"] = grids
        result.update(status="ANSWERED" if any(g is not None for g in grids) else "NO_GRID", prompt_tokens=len(prompt),
                      completion_tokens=len(tokens), thinking_tokens=thinking, finish_reason=finish,
                      answer_tail=answer[-2000:])
    except Exception as error:
        result.update(status="ERROR", error=f"{type(error).__name__}: {error}")
    result["elapsed_seconds"] = time.monotonic() - started
    atomic_json(root / f"{job['key']}.json", result)
    return result
