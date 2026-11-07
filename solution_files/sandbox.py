from __future__ import annotations

import ast
import builtins
import collections
import contextlib
import functools
import importlib
import io
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import numpy as np

Grid = list[list[int]]


def normalize_grid(value: Any) -> Grid | None:
    try:
        array = np.asarray(value)
        if array.ndim != 2 or array.size == 0 or any(not 0 < size <= 30 for size in array.shape):
            return None
        if not np.issubdtype(array.dtype, np.integer):
            if not np.all(np.equal(array, np.floor(array))):
                return None
        array = array.astype(int)
        if array.min() < 0 or array.max() > 9:
            return None
        return array.tolist()
    except (TypeError, ValueError, OverflowError):
        return None


_ALLOWED_IMPORTS = {
    "collections",
    "functools",
    "itertools",
    "math",
    "numpy",
}
FREE_IMPORTS = _ALLOWED_IMPORTS | {"bisect", "copy", "fractions", "heapq", "json", "random", "re", "statistics"}
# with scipy_ndimage, scipy.ndimage (image morphology, labelling, filters; no file access) is also predefined as ndimage and importable, never scipy itself
NDIMAGE = "scipy.ndimage"
_FREE_ATTRIBUTES = {"dumps", "loads"}  # json's string forms; the file forms (dump, load) stay blocked
_F3_ATTRIBUTES = {"remove", "replace"}
_F3_BLOCKED = {"os", "sys", "subprocess", "builtins", "codecs", "types", "f2py", "ctypes", "ctypeslib", "operator",
               "attrgetter", "methodcaller", "exec", "eval", "gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame",
               "ag_code", "f_back", "f_globals", "f_locals", "f_builtins", "f_code", "tb_frame"}
_F3_BUILTIN_NAMES = {"chr", "frozenset", "repr", "type"}  # and id, which numbers objects in call order (_stable_id)
F3_FUNCTION = re.compile(r"(?:transform|solve)\d*")  # the functions verified when a program defines them
# generators seeded from the OS or the clock, blocked so that a replay prints the same; the module-level generators stay, seeded (a missing seed becomes 0, see _free_namespace)
_FREE_BLOCKED = {"BitGenerator", "Generator", "MT19937", "PCG64", "PCG64DXSM", "Philox", "Random", "RandomState",
                 "SFC64", "SeedSequence", "SystemRandom", "datetime64"}
STDOUT_LIMIT = 4000
PYTHON = sys.executable  # the notebook sets the engine environment's Python
_ADDRESS = re.compile(r' at 0x[0-9a-fA-F]+(?=>)')  # object addresses differ per process; replays must match byte for byte
_BLOCKED_NAMES = {
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "quit",
    "setattr",
    "type",
    "vars",
}
_BLOCKED_ATTRIBUTES = {
    "chdir",
    "chmod",
    "connect",
    "ctypeslib",
    "DataSource",
    "dump",
    "dumps",
    "fromfile",
    "fromregex",
    "fromtextfile",
    "genfromtxt",
    "load",
    "loads",
    "loadtxt",
    "memmap",
    "mkdir",
    "open",
    "open_memmap",
    "popen",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "save",
    "savetxt",
    "savez",
    "savez_compressed",
    "socket",
    "spawn",
    "system",
    "tofile",
    "unlink",
    "write",
}
_SAFE_BUILTIN_NAMES = {
    "Exception",
    "IndexError",
    "KeyError",
    "RuntimeError",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "int",
    "isinstance",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "pow",
    "range",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self, ndimage: bool = False) -> None:
        self.error: str | None = None
        self.ndimage = ndimage
        self.imports = FREE_IMPORTS
        self.attributes = ((((_BLOCKED_ATTRIBUTES - _FREE_ATTRIBUTES) | _FREE_BLOCKED) - _F3_ATTRIBUTES - {"compile"})
                           | _F3_BLOCKED)
        self.names = _BLOCKED_NAMES

    def reject(self, message: str) -> None:
        if self.error is None:
            self.error = message

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if self.ndimage and alias.name == NDIMAGE:
                # bare "import scipy.ndimage" binds the whole scipy package (scipy.io reads and writes files)
                if alias.asname is None:
                    self.reject("import is not allowed: scipy.ndimage without a name; use from scipy import ndimage")
                continue
            root, *parts = alias.name.split(".")
            # import numpy.f2py as x would bypass the attribute check
            if root not in self.imports or any(part in self.attributes or part.startswith("_") for part in parts):
                self.reject(f"import is not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # type(x) with one positional argument; the name type anywhere else (type(name, bases, namespace), t = type) stays blocked by visit_Name
        if (isinstance(node.func, ast.Name) and node.func.id == "type" and len(node.args) == 1
                and not node.keywords and not isinstance(node.args[0], ast.Starred)):
            self.visit(node.args[0])
            return
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self.ndimage and not node.level and (node.module == NDIMAGE or (node.module == "scipy" and node.names
                                                                         and all(a.name == "ndimage" for a in node.names))):
            for alias in node.names:
                if alias.name == "*" or alias.name.startswith("_") or alias.name in self.attributes:
                    self.reject(f"import is not allowed: {alias.name}")
            self.generic_visit(node)
            return
        root, *parts = (node.module or "").split(".")
        # from numpy.ctypeslib import x would bypass the attribute check as import numpy.f2py would
        if node.level or root not in self.imports or any(part in self.attributes or part.startswith("_") for part in parts):
            self.reject(f"import is not allowed: {node.module}")
        for alias in node.names:  # from numpy import load would bypass the attribute check
            if alias.name == "*" or alias.name.startswith("_") or alias.name in self.attributes:
                self.reject(f"import is not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.names or node.id.startswith("__"):
            self.reject(f"name is not allowed: {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_") or node.attr in self.attributes:
            self.reject(f"attribute is not allowed: {node.attr}")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.reject("global declarations are not allowed")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.reject("nonlocal declarations are not allowed")


def safety_error(code: str, ndimage: bool = False) -> str | None:
    if not isinstance(code, str) or not code.strip():
        return "program is empty"
    try:
        tree = ast.parse(code)
        visitor = _SafetyVisitor(ndimage)
        visitor.visit(tree)
    except SyntaxError as exc:
        return f"syntax error: {exc.msg}"
    except RecursionError:
        return "syntax tree exceeds the verifier recursion limit"
    return visitor.error


def grid_problem(value: Any) -> str | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        value = [row.tolist() if hasattr(row, "tolist") else row for row in value]
    if not isinstance(value, list) or not all(isinstance(row, (list, tuple)) for row in value):
        return "it is not a list of rows"
    if not value:
        return "it has no rows"
    if len(value) > 30:
        return f"it has {len(value)} rows (at most 30)"
    lengths = [len(row) for row in value]
    common = collections.Counter(lengths).most_common(1)[0][0]
    odd = [f"row {number} has {length} value{'s' if length != 1 else ''}" for number, length in enumerate(lengths, 1)
           if length != common]
    if odd:
        return ", ".join(odd[:5]) + (", ..." if len(odd) > 5 else "") + f", the others {common}"
    if not 0 < common <= 30:
        return f"its rows have {common} values (1 to 30)"
    for number, row in enumerate(value, 1):
        for cell in row:
            if isinstance(cell, np.generic):  # numpy scalars count as their Python value
                cell = cell.item()
            integral = (isinstance(cell, int) and not isinstance(cell, bool)) or (isinstance(cell, float) and cell.is_integer())
            if not integral or not 0 <= cell <= 9:
                # only plain values are shown as written: any other object's repr is its own code
                shown = repr(cell)[:40] if type(cell) in (int, float, bool, str, type(None)) else type(cell).__name__
                return f"row {number} holds {shown} (cells are integers 0-9)"
    return None


def _restricted_import(name: str, globals_: Any = None, locals_: Any = None,
                       fromlist: Sequence[str] = (), level: int = 0, ndimage: bool = False) -> Any:
    root = name.split(".", 1)[0]
    if ndimage and not level and (name == NDIMAGE or (name == "scipy" and fromlist and set(fromlist) == {"ndimage"})):
        return builtins.__import__(name, globals_, locals_, fromlist, level)
    if level or root not in FREE_IMPORTS:
        raise ImportError(f"import is not allowed: {name}")
    return builtins.__import__(name, globals_, locals_, fromlist, level)


def _stable_id():
    seen: dict[int, tuple[int, Any]] = {}

    def stable(value: Any) -> int:
        # keeps every object passed to id() alive so that a number is never reused; a worker serves only one call
        return seen.setdefault(builtins.id(value), (len(seen), value))[0]
    return stable


def _safe_builtins(ndimage: bool = False) -> dict[str, Any]:
    namespace = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES | _F3_BUILTIN_NAMES}
    namespace["__import__"] = functools.partial(_restricted_import, ndimage=ndimage)
    namespace["print"] = builtins.print
    namespace["id"] = _stable_id()
    return namespace


class _BoundedOutput(io.StringIO):

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit, self.truncated = limit, False

    def write(self, text: str) -> int:
        text = _ADDRESS.sub(" at <address>", text)
        room = self.limit - self.tell()
        if len(text) > room:
            self.truncated = True
        return super().write(text[:max(0, room)]) if room > 0 else 0


def _error_line(exc: Exception) -> str:
    import traceback

    lines = [frame.lineno for frame in traceback.extract_tb(exc.__traceback__) if frame.filename == "<model-program>"]
    message = _ADDRESS.sub(" at <address>", str(exc))
    where = f" (line {lines[-1]})" if lines else ""
    return f"{type(exc).__name__}{where}: {' '.join(message.split())[:240]}"


def _fresh(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _free_namespace(train: list[dict], test: list[dict], ndimage: bool = False) -> dict:
    seed_random, seed_numpy, numpy_rng = random.seed, np.random.seed, np.random.default_rng
    random.seed = lambda a=None, version=2: seed_random(0 if a is None else a, version)
    np.random.seed = lambda seed=None: seed_numpy(0 if seed is None else seed)
    np.random.default_rng = lambda seed=None: numpy_rng(0 if seed is None else seed)
    random.seed(0)
    np.random.seed(0)
    return {"__builtins__": _safe_builtins(ndimage), "collections": __import__("collections"),
            "functools": functools, "itertools": itertools, "math": math, "np": np, "numpy": np,
            "train": _fresh(train), "test": _fresh(test)}


def _apply_function(function, grids: list) -> tuple[list, list]:
    outputs, errors = [], []
    for grid in grids:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = normalize_grid(function([list(row) for row in grid]))
            outputs.append(result)
            errors.append(None if result is not None else "InvalidGrid: function did not return a valid ARC grid")
        except Exception as exc:
            outputs.append(None)
            errors.append(_error_line(exc))
    return outputs, errors


def _plain(grid: Any) -> list[list]:
    rows = grid.tolist() if isinstance(grid, np.ndarray) else grid
    return [[cell.item() if isinstance(cell, np.generic) else cell for cell in row] for row in rows]


def objects(grid: Any, diagonal: bool = False, background: int | None = None) -> list[dict]:
    grid = _plain(grid)
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if diagonal else [])
    seen, found = set(), []
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if (r, c) in seen or color == background:
                continue
            seen.add((r, c))
            cells, stack = [], [(r, c)]
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for near in ((y + dy, x + dx) for dy, dx in steps):
                    if (near not in seen and 0 <= near[0] < len(grid) and 0 <= near[1] < len(grid[near[0]])
                            and grid[near[0]][near[1]] == color):
                        seen.add(near)
                        stack.append(near)
            cells.sort()
            columns = [x for _, x in cells]
            found.append({"color": color, "cells": cells, "bbox": (cells[0][0], min(columns), cells[-1][0], max(columns)),
                          "size": len(cells)})
    return found


def diff(a: Any, b: Any) -> str:
    a, b = _plain(a), _plain(b)
    if a == b:
        return "same"
    if [len(row) for row in a] != [len(row) for row in b]:
        shapes = [f"{len(grid)}x{len(grid[0]) if grid else 0}" + (" (rows of unequal length)" if len({*map(len, grid)}) > 1 else "")
                  for grid in (a, b)]
        return "shape differs: {} vs {}".format(*shapes)
    cells = [(r, c, x, y) for r, (row_a, row_b) in enumerate(zip(a, b)) for c, (x, y) in enumerate(zip(row_a, row_b)) if x != y]
    return (f"{len(cells)} of {sum(map(len, a))} cells differ, {'the first 50 ' if len(cells) > 50 else ''}as "
            f"(row, col, a, b): {cells[:50]}")


def _f3_session(train: list[dict], test: list[dict], session: Sequence, ndimage: bool = False) -> dict:
    namespace, checks = _free_namespace(train, test, ndimage), {}
    given = {"objects": objects, "diff": diff}
    if ndimage:
        given["ndimage"] = sys.modules[NDIMAGE]  # imported by the worker before the limits are set
    for earlier, functions in session:
        if earlier:
            namespace.update(train=_fresh(train), test=_fresh(test), checks=_fresh(checks), **given)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exec(compile(earlier, "<earlier-program>", "exec"), namespace)
        checks.update(functions)
    namespace.update(train=_fresh(train), test=_fresh(test), checks=_fresh(checks), **given)
    return namespace


def _run_f3(code: str, train: list[dict], test: list[dict], session: Sequence, submit: list | None = None,
            call_seconds: int | None = None, ndimage: bool = False) -> dict:
    namespace = _f3_session(train, test, session, ndimage)
    if call_seconds:
        _set_worker_limits(call_seconds, narrow=True)
    if submit is not None:
        return _run_submission(namespace, train, test, submit)
    tree = ast.parse(code)
    output, error, completed = _BoundedOutput(STDOUT_LIMIT), None, 0
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        try:
            for statement in tree.body:
                exec(compile(ast.Module([statement], []), "<model-program>", "exec"), namespace)
                completed += 1
        except Exception as exc:
            error = _error_line(exc)
    names = dict.fromkeys(node.name for node in tree.body[:completed]
                          if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and F3_FUNCTION.fullmatch(node.name))
    inputs = [pair["input"] for pair in train] + [pair["input"] for pair in test]
    functions = {name: dict(zip(("outputs", "errors"), _apply_function(namespace[name], inputs)))
                 for name in names if callable(namespace.get(name))}
    return {"ok": True, "stdout": output.getvalue(), "truncated": output.truncated, "error": error, "completed": completed,
            "functions": functions}


def _run_submission(namespace: dict, train: list[dict], test: list[dict], entries: list) -> dict:
    results, failing = [], {}
    for index, kind, name in entries:
        value, problem = namespace.get(name), None
        what = f"variable {name}" if kind == "variable" else f"the output of {name} for test input {index + 1}"
        if kind == "variable" and name not in namespace:
            problem = f"no variable named {name} is defined"
        elif kind == "function" and not callable(value):
            problem = f"no function named {name} is defined" if value is None else f"{name} is not a function"
        elif kind == "function":
            if name not in failing:
                outputs, _ = _apply_function(value, [pair["input"] for pair in train])
                failing[name] = [number for number, (actual, pair) in enumerate(zip(outputs, train), 1) if actual != pair["output"]]
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    value = value([list(row) for row in test[index]["input"]])
            except Exception as exc:
                problem = f"{name} raised on test input {index + 1}: {_error_line(exc)}"
        grid = None
        try:  # the value can be any session object: reading it as a grid can run its code
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                grid = None if problem else normalize_grid(value)
                if problem is None and grid is None:
                    problem = f"{what} is not a valid grid: {grid_problem(value) or 'not a list of rows of integers 0-9'}"
        except Exception as exc:
            grid, problem = None, f"{what} could not be read as a grid: {_error_line(exc)}"
        results.append({"grid": grid, "problem": problem})
    return {"ok": True, "entries": results, "failing": failing}


def _set_worker_limits(cpu_seconds: int, narrow: bool = False) -> None:
    with contextlib.suppress(ImportError, OSError, ValueError):
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        consumed = usage.ru_utime + usage.ru_stime
        soft = _absolute_cpu_limit(cpu_seconds, consumed)
        current = resource.getrlimit(resource.RLIMIT_CPU)[0]
        if narrow and current != resource.RLIM_INFINITY:
            soft = min(soft, current)
        resource.setrlimit(resource.RLIMIT_CPU, (soft, soft + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def _absolute_cpu_limit(cpu_seconds: int, consumed_seconds: float) -> int:
    return max(1, math.ceil(consumed_seconds + max(1, cpu_seconds)))


def _worker_main() -> int:
    try:
        payload = json.load(sys.stdin)
        code = payload["code"]
        earlier = [program for program, _ in payload["session"] if program]
        programs = [*earlier, *([] if "submit" in payload else [code])]
        ndimage = bool(payload.get("ndimage"))
        error = next(filter(None, (safety_error(program, ndimage) for program in programs)), None)
        if error:
            json.dump({"ok": False, "error": error}, sys.stdout)
            return 2
        if ndimage:  # before the limits are set, so that no program's time limit includes scipy's import
            try:
                importlib.import_module(NDIMAGE)
            except Exception:  # if scipy does not import, the programs run without ndimage and the worker still runs
                ndimage = False
        _set_worker_limits(int(payload.get("cpu_seconds", 5)))
        # nothing the programs write (fd 1 included) can reach the result: it goes to a copy of fd 1, and fd 1 and sys.stdout point at /dev/null before any program runs; os._exit skips the objects' finalizers
        result = os.dup(1)
        os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
        sys.stdout = open(os.devnull, "w")
        output = _run_f3(code, payload["train"], payload["test"], payload["session"], payload.get("submit"),
                         payload.get("call_seconds"), ndimage)
        with os.fdopen(result, "w") as stream:
            json.dump(output, stream)
            stream.flush()
        os._exit(0)
    except Exception as exc:
        json.dump({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, sys.stdout)
        return 3


def run_session(code: str, train: list[dict], test: list[dict], timeout: float = 5.0, session: Sequence = (),
                submit: list | None = None, ndimage: bool = False) -> dict | None:
    earlier = [program for program, _ in session if program]
    if (timeout <= 0 or any(set(pair) != {"input"} for pair in test)
            or any(safety_error(program, ndimage) for program in earlier + ([] if submit is not None else [code]))):
        return None
    allowance = timeout * (1 + len(earlier))
    result = _run_worker({"code": code, "train": train, "test": test, "session": [list(entry) for entry in session],
                          **({"submit": submit} if submit is not None else {}), "cpu_seconds": max(1, math.ceil(allowance)),
                          "call_seconds": max(1, math.ceil(timeout)), "ndimage": ndimage}, allowance)
    return result if _session_shape(result, code, len(train) + len(test), submit) else None


def _session_shape(result: Any, code: str, count: int, submit: list | None) -> bool:
    grid = lambda value: value is None or normalize_grid(value) is not None  # noqa: E731
    text = lambda value: value is None or isinstance(value, str)  # noqa: E731
    if not isinstance(result, dict) or result.get("ok") is not True:
        return False
    if submit is not None:
        entries, failing = result.get("entries"), result.get("failing")
        return (isinstance(entries, list) and len(entries) == len(submit)
                and all(isinstance(entry, dict) and set(entry) == {"grid", "problem"} and grid(entry["grid"])
                        and text(entry["problem"]) and (entry["grid"] is None) != (entry["problem"] is None) for entry in entries)
                and isinstance(failing, dict) and all(isinstance(pairs, list) for pairs in failing.values()))
    completed, functions = result.get("completed"), result.get("functions")
    return (isinstance(result.get("stdout"), str) and isinstance(result.get("truncated"), bool) and text(result.get("error"))
            and type(completed) is int and 0 <= completed <= len(ast.parse(code).body)
            and isinstance(functions, dict)
            and all(isinstance(run, dict) and set(run) == {"outputs", "errors"}
                    and all(isinstance(run[key], list) and len(run[key]) == count for key in run)
                    and all(map(grid, run["outputs"])) and all(map(text, run["errors"])) for run in functions.values()))


_WORKER_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "PYTHONIOENCODING": "utf-8",
    "BLIS_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def ndimage_error(timeout: float = 120.0) -> str | None:
    try:
        done = subprocess.run([PYTHON, "-c", f"import {NDIMAGE}"],
                              capture_output=True, text=True, env=_WORKER_ENVIRONMENT, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return f"{type(error).__name__}: {error}"[:300]
    if done.returncode == 0:
        return None
    return (done.stderr.strip().splitlines() or [f"exit code {done.returncode}"])[-1][:300]


def _run_worker(request: dict, timeout: float) -> dict | None:
    payload = json.dumps(request)
    worker_path = os.path.abspath(__file__)
    try:
        with tempfile.TemporaryDirectory(prefix="arc-codeexec-") as workdir:
            completed = subprocess.run(
                [PYTHON, worker_path, "--worker"],
                input=payload,
                text=True,
                capture_output=True,
                cwd=workdir,
                env=_WORKER_ENVIRONMENT,
                timeout=timeout + 1.0,
                check=False,
            )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 or not completed.stdout:
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


if __name__ == "__main__" and "--worker" in sys.argv:
    raise SystemExit(_worker_main())
