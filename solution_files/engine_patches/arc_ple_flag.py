import os
import sys
import time


def patch_ple_offload_load(module) -> None:
    original = module.PleOffloadRunner._load_weights

    def _load_weights(self) -> None:
        started = time.time()
        original(self)
        with open(os.environ["ARC_VLLM_PLE_LOADED_FLAG"] + ".part", "w") as handle:
            handle.write(f"{time.time()}\n")
        os.replace(os.environ["ARC_VLLM_PLE_LOADED_FLAG"] + ".part", os.environ["ARC_VLLM_PLE_LOADED_FLAG"])
        memory = {line.split(":")[0]: int(line.split()[1]) for line in open("/proc/meminfo")}
        print(f"PLE table loaded in {time.time() - started:.1f} s; MemAvailable {memory.get('MemAvailable', 0) / 2**20:.1f} GiB, "
              f"Cached {memory.get('Cached', 0) / 2**20:.1f} GiB", file=sys.stderr, flush=True)

    module.PleOffloadRunner._load_weights = _load_weights


PLE_WAIT_SECONDS = 10800  # Merlin's own PLE ready timeout, VLLM_PLE_OFFLOAD_READY_TIMEOUT in serve_command


def wait_for_file(path: str, timeout: float) -> float | None:
    started = time.time()
    while not os.path.exists(path):
        if time.time() - started >= timeout:
            return None
        time.sleep(1)
    return time.time() - started


def patch_wait_ple_loaded(module) -> None:
    worker = module.Worker
    original = worker.determine_available_memory

    def determine_available_memory(self):
        flag = os.environ["ARC_VLLM_PLE_LOADED_FLAG"]
        waited = wait_for_file(flag, PLE_WAIT_SECONDS)
        print(f"PLE table flag {flag}: " + (f"present after {waited:.1f} s" if waited is not None
                                            else f"absent after {PLE_WAIT_SECONDS} s, profiling anyway"), file=sys.stderr, flush=True)
        return original(self)

    worker.determine_available_memory = determine_available_memory

from arc_patches import chain
if os.environ.get("ARC_VLLM_PLE_LOADED_FLAG"):
    chain("vllm.v1.ple_offload.worker", patch_ple_offload_load)
    chain("vllm.v1.worker.gpu_worker", patch_wait_ple_loaded)
