# pin_waiting_hit is adapted from the Merlin engine's kv_cache_manager.py (Apache License 2.0) and modified by us; see THIRD_PARTY_NOTICES.md.
import logging
import os
import time

logger = logging.getLogger("vllm.arc_qsa_host_kv")

ROLE = "arc_qsa_host_kv"  # KVCacheGroupSpec.role of the host K/V group, mapped to QSAHostKVManager
SMEM_LIMIT = 99 * 1024  # dynamic shared memory per block on sm89; the swap-in kernel needs more as H grows
ALIGN = 256  # byte alignment of every buffer carved from a worker's one fixed GPU allocation
STATS_EVERY = 1024  # target-model forwards between two log lines of the device buffer's hit rate (0: no such lines)
# Registered host buffers are never freed: freeing one without cudaHostUnregister leaves a stale registration that breaks later copies from memory reusing those addresses ("invalid argument"). Pools live as long as the server.
_REGISTERED = []


def register_host_memory(tensor) -> None:
    import torch

    flags = 1 | 2  # cudaHostRegisterPortable | cudaHostRegisterMapped
    error = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.numel() * tensor.element_size(), flags)
    if error != torch.cuda.cudart().cudaError.success:
        raise RuntimeError(f"cudaHostRegister of {tensor.numel() * tensor.element_size() / 2**30:.2f} GiB failed: {error}")
    _REGISTERED.append(tensor)


def host_budget_bytes() -> int:
    return int(os.environ.get("ARC_VLLM_QSA_HOST_KV_SIZE", 10 * 10**9))


def resolve_smem_bytes(capacity: int, width: int) -> int:
    chunks = -(-capacity // 32)
    return 4 * (width + 2 * (chunks + 1) + 2 * width + 3) + 2 * (capacity + 2 * width)


def device_steps(q: int) -> int:
    steps = int(os.environ.get("ARC_VLLM_QSA_DEVICE_STEPS", "0"))
    if steps < 0:
        raise ValueError(f"ARC_VLLM_QSA_DEVICE_STEPS={steps} is not a row step count")
    return min(steps, q) if steps else q


def device_tokens() -> int:
    value = int(os.environ.get("ARC_VLLM_QSA_DEVICE_TOKENS", "0") or 0)
    if value < 0 or value % 16:
        raise ValueError(f"ARC_VLLM_QSA_DEVICE_TOKENS={value} is not a non-negative multiple of 16")
    return value


def device_capacity(rows_per_request: int, width: int, override: int = 0) -> int:
    need = rows_per_request * width
    capacity = override or -(-need // 16) * 16
    if capacity % 16:
        raise ValueError(f"H={capacity} is not a multiple of 16")
    if capacity < need:
        raise ValueError(f"H={capacity} is below {rows_per_request} rows x {width} entries = {need}")
    if capacity > 32767:
        raise ValueError(f"H={capacity} does not fit the swap-in kernel's int16 LRU slots")
    if resolve_smem_bytes(capacity, width) > SMEM_LIMIT:
        raise ValueError(f"H={capacity} needs {resolve_smem_bytes(capacity, width)} B of shared memory per block (max {SMEM_LIMIT})")
    return capacity


def staging_blocks(max_model_len: int, lookahead: int, block_size: int, override: int = 0) -> int:
    one = -(-(max_model_len + lookahead) // block_size)
    blocks = override or 2 * one
    if blocks < one:
        raise ValueError(f"ARC_VLLM_QSA_STAGING_BLOCKS={blocks} cannot hold one context of {one} blocks")
    return blocks


def buffer_layout(layers: int, geo: dict) -> tuple[dict[str, tuple[int, int]], int]:
    cells, slots, rows, width = layers * geo["slots"] * geo["capacity"], geo["slots"], geo["rows"], geo["width"]
    sizes = dict(device_kv=cells * geo["row_bytes"], slab=geo["staging"] * geo["page_bytes"], keys=4 * cells,
                 lru=2 * cells, lru_init=2 * geo["capacity"], staging_table=4 * slots * geo["staging"], states=4 * slots,
                 request_ids=4 * slots, identity=4 * slots, zeros=4 * rows, attention=4 * rows * (width + 1),
                 device_indices=4 * rows * width, swap_host=4 * rows * width, swap_device=4 * rows * width,
                 swap_counts=4 * rows, stats=16)
    layout, end = {}, 0
    for name, size in sizes.items():
        layout[name] = (end, size)
        end += -(-size // ALIGN) * ALIGN
    return layout, end


def allocator_bytes(size: int) -> int:
    if size <= 1 << 20:
        return 2 << 20
    if size < 10 << 20:
        return 20 << 20
    return -(-size // (2 << 20)) * (2 << 20)


def reserve_bytes(layers: int, geo: dict) -> int:
    return allocator_bytes(buffer_layout(layers, geo)[1]) if layers else 0


def host_num_blocks(budget_bytes: int, block_bytes_per_worker: list[int]) -> int:
    return min(budget_bytes // b for b in block_bytes_per_worker if b)


def merge_state_groups(layer_sets: list[list[str]], keys: list, layer_bytes: dict[str, int],
                       worker_layers: list[set[str]], strides: list[int]) -> list[list[int]]:
    merged = []  # [key, member indices, bytes per worker]
    for i, (names, key) in enumerate(zip(layer_sets, keys)):
        own = [sum(layer_bytes[name] for name in names if name in layers) for layers in worker_layers]
        for entry in merged:
            if key is not None and entry[0] == key and all(a + b <= s for a, b, s in zip(entry[2], own, strides)):
                entry[1].append(i)
                entry[2] = [a + b for a, b in zip(entry[2], own)]
                break
        else:
            merged.append([key, [i], own])
    return [entry[1] for entry in merged]


def packed_page(state_bytes: int, strides: list[int]) -> int | None:
    half = min(s for s in strides if s) // 2 // ALIGN * ALIGN
    return half if state_bytes <= half else None


def tight_mamba_groups(kvu, groups, worker_specs):
    import dataclasses

    states = [s.real_page_size_bytes for g in groups for s in kvu._group_layer_specs(g) if isinstance(s, kvu.MambaSpec)]
    if not states:
        return groups, kvu._worker_strides(groups, worker_specs)
    page = -(-max(states) // ALIGN) * ALIGN

    def resize(spec):
        return dataclasses.replace(spec, page_size_padded=page) if isinstance(spec, kvu.MambaSpec) else spec

    out = []
    for group in groups:
        spec = group.kv_cache_spec
        if isinstance(spec, kvu.UniformTypeKVCacheSpecs):
            spec = kvu.UniformTypeKVCacheSpecs(block_size=spec.block_size,
                                               kv_cache_specs={n: resize(s) for n, s in spec.kv_cache_specs.items()})
        out.append(dataclasses.replace(group, kv_cache_spec=resize(spec)))
    return out, [max(s, 2 * page) if s else 0 for s in kvu._worker_strides(out, worker_specs)]


def encode_host_copy(src: int, dst: int) -> tuple[int, int]:
    return -1 - src, -1 - dst


def split_copies(pairs) -> tuple[list, list[tuple[int, int]]]:
    gpu = [pair for pair in pairs if pair[0] >= 0]
    host = [(-1 - pair[0], -1 - pair[1]) for pair in pairs if pair[0] < 0]
    if any(pair[1] < 0 for pair in gpu) or any(dst < 0 for _, dst in host):
        raise ValueError(f"QSA host KV: a block copy mixes host and GPU ids: {list(pairs)}")
    return gpu, host


def fold_warmup_block_ids(output, host_groups: list[int], num_blocks: int, packed: dict[int, int] | None = None):
    tables = [request.block_ids for request in output.scheduled_new_reqs]
    tables += [ids for ids in output.scheduled_cached_reqs.new_block_ids if ids is not None]
    for table in tables:
        for group in host_groups:
            table[group][:] = [1 + (block - 1) % (num_blocks - 1) for block in table[group]]
        for group, slots in (packed or {}).items():
            table[group][:] = [block * slots for block in table[group]]
    return output


def row_steps(num_rows: int, rows_per_request: int) -> list[tuple[int, int]]:
    return [(j, len(range(j, num_rows, rows_per_request))) for j in range(min(rows_per_request, num_rows))]


def staging_plan(seq_lens: list[int], block_size: int, capacity: int) -> list[list[tuple[int, int, int]]]:
    chunks, current, used = [], [], 0
    for i, seq_len in enumerate(seq_lens):
        blocks = -(-seq_len // block_size)
        if blocks > capacity:
            raise ValueError(f"QSA host KV: a context of {blocks} blocks exceeds the staging slab ({capacity} blocks)")
        if used + blocks > capacity:
            chunks.append(current)
            current, used = [], 0
        current.append((i, used, blocks))
        used += blocks
    if current:
        chunks.append(current)
    return chunks


def geometry(vllm_config, spec) -> dict:
    text = vllm_config.model_config.hf_text_config
    width = int(text.indexer_budget) + int(text.indexer_compress_ratio) - 1
    lookahead = int(vllm_config.num_speculative_tokens or 0)
    q = 1 + lookahead
    slots = int(vllm_config.scheduler_config.max_num_seqs)
    page_bytes = int(spec.page_size_bytes)
    return dict(
        width=width, q=q, slots=slots, rows=slots * q, page_bytes=page_bytes, block_size=int(spec.block_size),
        row_bytes=page_bytes // int(spec.block_size),
        steps=device_steps(q), capacity=device_capacity(device_steps(q), width, device_tokens()),
        staging=staging_blocks(vllm_config.model_config.max_model_len, lookahead, int(spec.block_size),
                               int(os.environ.get("ARC_VLLM_QSA_STAGING_BLOCKS", "0"))),
    )


def worker_budgets(available_memory: list[int], layers_per_worker: list[int], geo: dict) -> tuple[list[int], list[int]]:
    reserves = [reserve_bytes(layers, geo) for layers in layers_per_worker]
    budgets = [available - reserve for available, reserve in zip(available_memory, reserves)]
    if min(budgets) <= 0:
        raise ValueError(f"ARC_VLLM_QSA_HOST_KV: the KV budget {available_memory} does not cover the fixed GPU buffers "
                         f"{reserves}")
    return budgets, reserves


def merge_mamba_groups(kvu, groups, worker_specs):
    import dataclasses

    mode = os.environ.get("ARC_VLLM_GDN_PACK", "0")
    pack = mode in ("1", "tight")
    groups, strides = (tight_mamba_groups(kvu, groups, worker_specs) if mode == "tight"
                       else (groups, kvu._worker_strides(groups, worker_specs)))
    keys, layer_bytes, layer_specs, halved = [], {}, {}, 0
    for group in groups:
        specs = kvu._group_layer_specs(group)
        mergeable = (specs and group.slots_per_block == 1
                     and all(isinstance(s, kvu.MambaSpec) and not s.tp_replicated for s in specs)
                     and len({kvu._mamba_schedule_key(s) for s in specs}) == 1)
        if pack and mergeable:
            pages = [packed_page(s.real_page_size_bytes, strides) for s in specs]
            specs = [s if page is None else dataclasses.replace(s, page_size_padded=page) for s, page in zip(specs, pages)]
            halved += sum(page is not None for page in pages)
        layer_specs.update(zip(group.layer_names, specs))
        layer_bytes.update((name, spec.page_size_bytes) for name, spec in zip(group.layer_names, specs))
        keys.append((kvu._mamba_schedule_key(specs[0]), group.is_eagle_group) if mergeable else None)
    members = merge_state_groups([g.layer_names for g in groups], keys, layer_bytes,
                                 [set(spec) for spec in worker_specs], strides)
    if len(members) == len(groups):
        return groups
    merged = []
    for indices in members:
        if len(indices) == 1:
            merged.append(groups[indices[0]])
            continue
        specs = {name: layer_specs[name] for i in indices for name in groups[i].layer_names}
        merged.append(kvu.KVCacheGroupSpec(
            list(specs), kvu.UniformTypeKVCacheSpecs(block_size=groups[indices[0]].kv_cache_spec.block_size,
                                                     kv_cache_specs=specs),
            is_eagle_group=groups[indices[0]].is_eagle_group))
    logger.info("QSA host KV: %d KV cache groups -> %d (Mamba groups paired across workers%s, block bytes per worker "
                "at most %s)", len(groups), len(merged), f", {halved} Mamba layers on half-block pages" if halved else "", strides)
    return merged


def add_host_group(kvu, config, layers: dict, spec, num_blocks: int, eagle: bool) -> None:
    from vllm.v1.hisparse.layout import _build_hisparse_kv_cache_tensors
    from vllm.v1.kv_cache_layout import KVCacheLayout

    group = kvu.KVCacheGroupSpec(list(layers), spec, is_eagle_group=eagle and bool(layers), host_resident=True,
                                 role=ROLE)
    config.kv_cache_groups.append(group)
    config.hisparse_host_num_blocks = num_blocks
    if layers:
        block_bytes = len(layers) * spec.page_size_bytes
        config.kv_cache_tensors.extend(_build_hisparse_kv_cache_tensors(
            [group], num_blocks, block_bytes * num_blocks, KVCacheLayout.LBNHC, block_bytes, host_resident=True))


def patch_kv_cache_utils(kvu) -> None:
    configs_fn, groups_fn = kvu.get_kv_cache_configs, kvu.get_kv_cache_groups
    context = {}

    def get_kv_cache_groups(vllm_config, kv_cache_spec):
        groups = groups_fn(vllm_config, kv_cache_spec)
        worker_specs = context.get("worker_specs")
        return groups if worker_specs is None or not groups else merge_mamba_groups(kvu, groups, worker_specs)

    def get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory):
        host = [{n: s for n, s in specs.items() if type(s) is kvu.FullAttentionSpec} for specs in kv_cache_specs]
        if not any(host):
            return configs_fn(vllm_config, kv_cache_specs, available_memory)
        if vllm_config.attention_config.hisparse_config is not None or vllm_config.kv_transfer_config is not None:
            raise NotImplementedError("ARC_VLLM_QSA_HOST_KV: no native HiSparse and no KV connector")
        spec = kvu.FullAttentionSpec.merge([s for layers in host for s in layers.values()])
        geo = geometry(vllm_config, spec)
        budgets, reserves = worker_budgets(list(available_memory), [len(layers) for layers in host], geo)
        host_blocks = host_num_blocks(host_budget_bytes(), [len(layers) * geo["page_bytes"] for layers in host])
        per_request = -(-(vllm_config.model_config.max_model_len + geo["q"] - 1) // geo["block_size"])
        if host_blocks <= per_request:
            raise ValueError(f"ARC_VLLM_QSA_HOST_KV_SIZE holds {host_blocks} host blocks, one request needs {per_request}"
                             " + the null block")
        context["worker_specs"] = [{n: s for n, s in specs.items() if n not in layers}
                                   for specs, layers in zip(kv_cache_specs, host)]
        try:
            configs = configs_fn(vllm_config, context["worker_specs"], budgets)
        finally:
            context.clear()
        # The drafter's K/V layer registers last, as its index cache does on the index group, which vLLM flags as the EAGLE group: the host group shares that flag.
        eagle = any(g.is_eagle_group and all(isinstance(s, kvu.FullAttentionSpec) for s in kvu._group_layer_specs(g))
                    for config in configs for g in config.kv_cache_groups if g.layer_names)
        for config, layers in zip(configs, host):
            add_host_group(kvu, config, layers, spec, host_blocks, eagle)
        logger.info(
            "QSA host KV: %s QSA K/V layers per worker in host memory, %d host blocks (%.1f GB in all, %d tokens "
            "per worker); GPU KV budget %s B - fixed %s B (H=%d rows x %d request slots, staging %d blocks) -> %d GPU "
            "blocks of %s B", [len(layers) for layers in host], host_blocks,
            sum(len(layers) for layers in host) * geo["page_bytes"] * host_blocks / 1e9, host_blocks * geo["block_size"],
            list(available_memory), reserves, geo["capacity"], geo["slots"], geo["staging"], configs[0].num_blocks,
            [kvu._get_kv_cache_bytes_per_block([g for g in c.kv_cache_groups if not g.host_resident]) for c in configs])
        return configs

    kvu.get_kv_cache_groups, kvu.get_kv_cache_configs = get_kv_cache_groups, get_kv_cache_configs


def patch_single_type_manager(module) -> None:
    class QSAHostKVManager(module.FullAttentionManager):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._record_new_block_ids = False  # host ids never reach the GPU zeroer
            self.device_pool = self.block_pool
            self.host_shortfall = 0  # of the last get_num_blocks_to_allocate, read by the coordinator

        def take_new_block_ids(self) -> list[int]:
            self.new_block_ids = []  # _apply_cow records copy-on-write targets here unconditionally
            return []

        def bind_host_pool(self, num_blocks: int) -> None:
            device_pool = self.device_pool
            self.block_pool = module.SharedEventQueueBlockPool(
                num_gpu_blocks=num_blocks, enable_caching=self.enable_caching and self.kv_cache_spec.prefix_cacheable,
                hash_block_size=device_pool.hash_block_size, enable_kv_cache_events=device_pool.enable_kv_cache_events,
                metrics_collector=device_pool.metrics_collector, medium=module.MEDIUM_CPU, event_owner=device_pool)
            self._null_block = self.block_pool.null_block

        def take_pending_cow_copies(self):
            return []  # host ids: take_host_cow_copies hands them over separately

        def take_host_cow_copies(self):
            copies, self._pending_cow_copies = self._pending_cow_copies, []
            return copies

        def get_num_blocks_to_allocate(self, request_id, num_tokens, new_computed_blocks, total_computed_tokens,
                                       num_local_computed_tokens, num_tokens_main_model, apply_admission_cap=False):
            required = super().get_num_blocks_to_allocate(
                request_id, num_tokens, new_computed_blocks, total_computed_tokens, num_local_computed_tokens,
                num_tokens_main_model, apply_admission_cap=apply_admission_cap)
            # Host blocks are not GPU blocks: the coordinator turns a host shortfall into a GPU-pool need that fails
            self.host_shortfall = max(required - self.block_pool.get_num_free_blocks(), 0)
            return 0

    QSAHostKVManager.__qualname__ = QSAHostKVManager.__name__
    module.QSAHostKVManager = QSAHostKVManager
    module.KVCacheSpecRegistry.register_role_manager(ROLE, QSAHostKVManager)


def patch_coordinator(module) -> None:
    init, count = module.KVCacheCoordinator.__init__, module.KVCacheCoordinator.get_num_blocks_to_allocate

    def __init__(self, kv_cache_config, *args, **kwargs):
        init(self, kv_cache_config, *args, **kwargs)  # subclasses' hit-lookup setup reads the pools after this
        for group, manager in zip(kv_cache_config.kv_cache_groups, self.single_type_managers):
            if group.role == ROLE:
                manager.bind_host_pool(kv_cache_config.hisparse_host_num_blocks)

    def get_num_blocks_to_allocate(self, *args, **kwargs):
        need = count(self, *args, **kwargs)
        short = sum(getattr(manager, "host_shortfall", 0) for manager in self.single_type_managers)
        # The scheduler measures the need against the GPU pool's free blocks. A host shortfall puts it past them by that many blocks, so the allocation fails and last_alloc_shortfall is the GPU deficit plus the host deficit: what released pins must cover (the scheduler's yield test compares it with num_yieldable_blocks over both pools).
        return max(need, self.block_pool.get_num_free_blocks()) + short if short else need

    module.KVCacheCoordinator.__init__ = __init__
    module.KVCacheCoordinator.get_num_blocks_to_allocate = get_num_blocks_to_allocate


def patch_kv_cache_manager(module) -> None:
    take = module.KVCacheManager.take_kv_cache_block_copies

    def pin_waiting_hit(self, request) -> int:
        request_id = request.request_id
        if (request_id in self._pinned_hits or not self.enable_caching or not self.prefix_cache_lookup_enabled(request)
                or len(self._pinned_hits) >= self.MAX_PINNED_WAITERS):
            return 0
        computed, num_hit, _ = self.get_computed_blocks(request)
        blocks = [block for group in computed.blocks for block in group if not block.is_null]
        if num_hit == 0 or not blocks:
            return 0
        proof = num_hit // self.block_pool.hash_block_size
        if proof < len(request.block_hashes):
            seen = {id(block) for block in blocks}
            for group_id, manager in enumerate(self.coordinator.single_type_managers):
                cached = manager.block_pool.get_cached_block(request.block_hashes[proof], [group_id])
                if cached and id(cached[0]) not in seen:
                    seen.add(id(cached[0]))
                    blocks.append(cached[0])
        for block in blocks:
            (block.pool or self.block_pool).touch([block])
        self._pinned_hits[request_id] = blocks
        return len(blocks)

    def take_kv_cache_block_copies(self):
        copies, retained = take(self)
        pairs = [pair for manager in self.coordinator.single_type_managers
                 if hasattr(manager, "take_host_cow_copies") for pair in manager.take_host_cow_copies()]
        if not pairs:
            return copies, retained
        # Retained endpoints are freed through the GPU pool, which routes each block to its own pool
        return ([*copies, *(module.KVCacheBlockCopy(*encode_host_copy(s.block_id, d.block_id)) for s, d in pairs)],
                [*retained, *(block for pair in pairs for block in pair)])

    module.KVCacheManager.take_kv_cache_block_copies = take_kv_cache_block_copies
    module.KVCacheManager.pin_waiting_hit = pin_waiting_hit


_FORCED_ROWS: list[int] = []  # _select_config's row count while a launch covers part of a batch
_WORKER = {}  # "vllm_config", "host_group_ids" (set by initialize_kv_cache), "state" (HostKV)


class HostKV:

    def __init__(self, kv_cache_config, host_tensors, device, vllm_config):
        import torch
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        from vllm.v1.kv_cache_interface import create_kv_cache_views
        from vllm.v1.kv_cache_layout import KVCacheLayout

        import vllm._C_stable_libtorch  # noqa: F401 (registers torch.ops._C_cache_ops.hisparse_*)

        self.ops = torch.ops._C_cache_ops
        group = next(g for g in kv_cache_config.kv_cache_groups if g.role == ROLE)
        spec = group.kv_cache_spec
        geo = geometry(vllm_config, spec)
        self.geo, self.device, self.dtype = geo, device, spec.dtype
        self.kv_heads, self.head_size = spec.num_kv_heads, spec.head_size
        n, capacity, slots = len(group.layer_names), geo["capacity"], geo["slots"]
        num_blocks = kv_cache_config.hisparse_host_num_blocks
        cols = geo["row_bytes"] // spec.dtype.itemsize

        started = time.perf_counter()
        self.host = torch.empty(host_tensors[0].size, dtype=torch.int8)
        register_host_memory(self.host)
        registered = time.perf_counter() - started
        layout, size = buffer_layout(n, geo)
        reserved = torch.cuda.memory_reserved(device)
        self.buffer = torch.empty(size, dtype=torch.uint8, device=device)  # one allocation, rounded once as reserve_bytes plans
        reserved = torch.cuda.memory_reserved(device) - reserved

        def carve(name, dtype, *shape):
            start, nbytes = layout[name]
            return self.buffer[start:start + nbytes].view(dtype).view(shape)

        device_kv = carve("device_kv", spec.dtype, n, slots, capacity, cols)
        keys = carve("keys", torch.int32, n, slots, capacity).fill_(-1)
        self.lru_init = carve("lru_init", torch.int16, capacity).copy_(torch.arange(capacity, dtype=torch.int16))
        lru = carve("lru", torch.int16, n, slots, capacity).copy_(self.lru_init.expand(n, slots, capacity))
        self.keys, self.lru = keys, lru
        self.layers, self.views = {}, {}
        index = 0
        for tensor in host_tensors:
            for name, view in zip(tensor.layers, create_kv_cache_views(self.host, spec, num_blocks, KVCacheLayout.LBNHC,
                                                                       tensor)):
                uva = get_accelerator_view_from_cpu_tensor(view)
                self.views[name] = uva
                rows = view.transpose(1, 2)  # [blocks, tokens, heads, width]: one contiguous row per token
                self.layers[name] = dict(
                    host=rows.reshape(-1, cols), device=device_kv[index], keys=keys[index], lru=lru[index],
                    device_kv=device_kv[index].view(slots, capacity, self.kv_heads, -1).split(self.head_size, dim=-1),
                    pages=uva.transpose(1, 2).view(torch.uint8).reshape(num_blocks, -1).view(torch.int64))
                index += 1
        rows, width = geo["rows"], geo["width"]
        self.states = carve("states", torch.int32, slots).fill_(-1)  # batch request -> request slot
        self.device_table = self.states.view(slots, 1)  # one device block of H rows per request slot
        self.request_ids = carve("request_ids", torch.int32, slots).copy_(torch.arange(slots, dtype=torch.int32))
        self.identity = carve("identity", torch.int32, 1, slots).copy_(self.request_ids.view(1, slots))
        self.zeros = carve("zeros", torch.int32, rows).zero_()
        self.attention = carve("attention", torch.int32, rows, width + 1).fill_(-1)
        self.device_indices = carve("device_indices", torch.int32, rows, width)
        self.swap_host = carve("swap_host", torch.int32, rows, width)
        self.swap_device = carve("swap_device", torch.int32, rows, width)
        self.swap_counts = carve("swap_counts", torch.int32, rows).zero_()
        self.stats_int = carve("stats", torch.int64, 2).zero_()  # uint64 has few torch ops
        self.stats = self.stats_int.view(torch.uint64)  # hits, misses (the swap-in kernel's atomics)
        self.slab = carve("slab", spec.dtype, geo["staging"], geo["block_size"], self.kv_heads, cols // self.kv_heads)
        self.slab_kv = self.slab.split(self.head_size, dim=-1)
        self.slab_pages = self.slab.view(torch.uint8).view(geo["staging"], -1).view(torch.int64)
        self.staging_entries = carve("staging_table", torch.int32, slots * geo["staging"])
        self.forwards = 0
        # ARC_VLLM_QSA_SWAP_OVERLAP: the next row-step group's swap-in on a side stream while the current group attends
        self.overlap_stream = None
        if os.environ.get("ARC_VLLM_QSA_SWAP_OVERLAP") == "1":
            if geo["capacity"] < 2 * geo["steps"] * geo["width"]:
                raise ValueError(f"ARC_VLLM_QSA_SWAP_OVERLAP needs H >= 2 x {geo['steps']} row steps x {geo['width']} "
                                 f"entries, H is {geo['capacity']}")
            self.overlap_stream = torch.cuda.Stream(device)
        planned = reserve_bytes(n, geo)
        (logger.warning if reserved > planned else logger.info)(
            "QSA host KV: %d layers x %d host blocks = %.2f GB registered in %.1f s; device buffers (H=%d x %d slots), "
            "maps, row buffers and staging slab (%d blocks): %.3f GB in one allocation, %.3f GB newly reserved by the "
            "allocator, %.3f GB reserved in pool sizing", n, num_blocks, self.host.numel() / 1e9, registered, capacity, slots,
            geo["staging"], size / 1e9, reserved / 1e9, planned / 1e9)


    def set_requests(self, idx_mapping, num_reqs: int) -> None:
        if num_reqs:
            self.states[:num_reqs].copy_(idx_mapping[:num_reqs])
        self.states[num_reqs:].fill_(-1)
        every = STATS_EVERY
        if not num_reqs or every <= 0:
            return
        self.forwards += 1
        if self.forwards % every == 0:
            hits, misses = self.stats_int.tolist()
            self.stats_int.zero_()
            if hits + misses:
                logger.info("QSA host KV: last %d forwards: %.1f%% of %d selected entries served from the device "
                            "buffer, %.1f MB read from host memory per forward", every, 100.0 * hits / (hits + misses),
                            hits + misses, misses * self.geo["row_bytes"] / every / 1e6)

    def reset_slot(self, req_index: int) -> None:
        self.keys[:, req_index].fill_(-1)
        self.lru[:, req_index].copy_(self.lru_init)

    def copy_host_blocks(self, pairs: list[tuple[int, int]]) -> None:
        import torch

        src = torch.tensor([s for s, _ in pairs], dtype=torch.int64).to(self.device, non_blocking=True)
        dst = torch.tensor([d for _, d in pairs], dtype=torch.int64).to(self.device, non_blocking=True)
        for layer in self.layers.values():
            layer["pages"].index_copy_(0, dst, layer["pages"].index_select(0, src))


    def attend(self, layer, host: dict, query, attn_metadata, side, output, token_to_req, use_prefill_config,
               output_gate) -> None:
        num_tokens = attn_metadata.num_actual_tokens
        selected = layer.topk_indices_buffer[:num_tokens]
        written = attn_metadata.slot_mapping[:num_tokens]
        self.ops.hisparse_invalidate_written_slots(host["keys"], self.states, token_to_req[:written.shape[0]], written)
        decode_rows = side.num_decode_tokens if side.decode_query_len else 0
        partial = num_tokens if decode_rows < num_tokens else None
        q = side.decode_query_len
        if decode_rows and self.geo["steps"] >= q:
            self.resolve(host, selected, decode_rows, q, attn_metadata.block_table)
            self.launch(layer, query[:decode_rows], host["device_kv"], self.attention[:decode_rows], self.identity,
                        self.zeros[:decode_rows], use_prefill_config, output[:decode_rows],
                        output_gate[:decode_rows], partial)
        elif decode_rows:  # ARC_VLLM_QSA_DEVICE_STEPS: H holds `steps` row steps, so each group attends before the next
            import torch

            steps, config_rows = self.geo["steps"], partial or decode_rows
            requests = torch.arange(decode_rows // q, device=query.device)[:, None] * q
            groups = [range(first, min(first + steps, q)) for first in range(0, q, steps)]
            stream = self.overlap_stream if len(groups) > 1 else None
            if stream is not None:
                self.resolve(host, selected, decode_rows, q, attn_metadata.block_table, groups[0])
            for i, group in enumerate(groups):
                ready = None
                if stream is None:
                    self.resolve(host, selected, decode_rows, q, attn_metadata.block_table, group)
                elif i + 1 < len(groups):  # the next group's swap-in, after this group's, beside this group's attention
                    ready = torch.cuda.Event()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        self.resolve(host, selected, decode_rows, q, attn_metadata.block_table, groups[i + 1])
                        ready.record(stream)
                rows = (requests + torch.arange(group.start, group.stop, device=query.device)[None, :]).reshape(-1)
                out = output.index_select(0, rows)
                self.launch(layer, query.index_select(0, rows), host["device_kv"], self.attention.index_select(0, rows),
                            self.identity, self.zeros[:rows.shape[0]], use_prefill_config, out,
                            output_gate.index_select(0, rows), config_rows)
                output.index_copy_(0, rows, out)
                if ready is not None:
                    torch.cuda.current_stream().wait_event(ready)
        if decode_rows < num_tokens:
            self.stage_prefills(layer, host, query, selected, attn_metadata, side, output, token_to_req,
                                use_prefill_config, output_gate, num_tokens)

    def resolve(self, host: dict, selected, num_rows: int, rows_per_request: int, block_table, group=None) -> None:
        width, capacity = selected.shape[1] - 1, self.geo["capacity"]
        if width != self.geo["width"] or num_rows > self.geo["rows"]:
            raise ValueError(f"QSA host KV: {num_rows} rows of width {width}, buffers hold {self.geo['rows']} rows of "
                             f"width {self.geo['width']}")
        for first, rows in row_steps(num_rows, rows_per_request):
            if group is not None and first not in group:
                continue
            swap = slice(0, rows)
            self.ops.hisparse_resolve_residency(
                host["host"], host["device"], self.device_table, selected[first:num_rows:rows_per_request, :width],
                self.device_indices[:rows], host["keys"], host["lru"], self.states, capacity, None, self.stats,
                self.attention[first:num_rows:rows_per_request, :width], capacity, self.request_ids[:rows], block_table,
                self.geo["block_size"], None, None, self.swap_host[swap], self.swap_device[swap],
                self.swap_counts[swap], None, 0, 0)
            self.ops.hisparse_gather_compact(host["host"], host["device"], self.swap_host[swap], self.swap_device[swap],
                                             self.swap_counts[swap])
        if group is None:
            self.attention[:num_rows, width].copy_(selected[:num_rows, width])
        else:  # only this group's rows: another group may be attending beside it (ARC_VLLM_QSA_SWAP_OVERLAP)
            for first in group:
                self.attention[first:num_rows:rows_per_request, width].copy_(selected[first:num_rows:rows_per_request, width])

    def stage_prefills(self, layer, host: dict, query, selected, attn_metadata, side, output, token_to_req,
                       use_prefill_config, output_gate, num_tokens: int) -> None:
        import numpy as np
        import torch
        from vllm.utils.torch_utils import async_tensor_h2d

        starts = side.arc_query_start_loc_cpu.tolist()
        seq_lens = side.arc_seq_lens_cpu
        seq_lens = [attn_metadata.max_seq_len] * (len(starts) - 1) if seq_lens is None else seq_lens.tolist()
        requests = [r for r in range(side.num_decodes, len(starts) - 1) if starts[r + 1] > starts[r]]
        block_table = attn_metadata.block_table
        table = self.staging_table(block_table.shape[1])[:block_table.shape[0]]
        limit = block_table.shape[1] * self.geo["block_size"]  # a block table row covers at most this many tokens
        plan = staging_plan([min(seq_lens[r], limit) for r in requests], self.geo["block_size"], self.geo["staging"])
        for chunk in plan:
            members = [requests[i] for i, _, _ in chunk]
            pages = sum(blocks for _, _, blocks in chunk)
            ids = torch.cat([block_table[r, :blocks] for r, (_, _, blocks) in zip(members, chunk)]).long()
            torch.index_select(host["pages"], 0, ids, out=self.slab_pages[:pages])
            remap = np.full((len(members), block_table.shape[1]), -1, dtype=np.int32)
            for k, (_, first, blocks) in enumerate(chunk):
                remap[k, :blocks] = np.arange(first, first + blocks)
            table.fill_(-1)
            table[async_tensor_h2d(np.array(members), device=self.device, dtype=torch.int64)] = async_tensor_h2d(
                remap, device=self.device)
            rows = slice(starts[members[0]], starts[members[-1] + 1])
            self.launch(layer, query[rows], self.slab_kv, selected[rows], table, token_to_req[rows],
                        use_prefill_config, output[rows], output_gate[rows], num_tokens)

    def staging_table(self, width: int):
        if width > self.geo["staging"]:
            raise ValueError(f"QSA host KV: a block table of width {width} exceeds the staging slab "
                             f"({self.geo['staging']} blocks)")
        return self.staging_entries[:self.geo["slots"] * width].view(self.geo["slots"], width)

    @staticmethod
    def launch(layer, query, kv, indices, block_table, token_to_req, use_prefill_config, out, output_gate,
               config_rows) -> None:
        from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

        if config_rows is not None:
            _FORCED_ROWS.append(config_rows)
        try:
            qsa_sparse_paged_attention(query, kv[0], kv[1], indices, block_table, token_to_req, use_prefill_config,
                                       out=out, k_scale=layer._k_scale, v_scale=layer._v_scale,
                                       output_gate=output_gate)
        finally:
            if config_rows is not None:
                _FORCED_ROWS.pop()


def state() -> HostKV | None:
    return _WORKER.get("state")


def patch_attn_utils(module) -> None:
    allocate = module.allocate_kv_cache

    def allocate_kv_cache(kv_cache_config, device, layout, kernel_block_sizes=None):
        import copy

        host = [t for t in kv_cache_config.kv_cache_tensors if t.host_resident]
        if not host:
            return allocate(kv_cache_config, device, layout, kernel_block_sizes)
        device_config = copy.copy(kv_cache_config)
        device_config.kv_cache_tensors = [t for t in kv_cache_config.kv_cache_tensors if not t.host_resident]
        if state() is not None:  # a second host buffer registration would never be freed
            raise RuntimeError("QSA host KV: KV cache initialized twice in one worker (needs "
                               "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0)")
        caches = allocate(device_config, device, layout, kernel_block_sizes)
        host_kv = _WORKER["state"] = HostKV(kv_cache_config, host, device, _WORKER["vllm_config"])
        caches.update(host_kv.views)  # layers bind their UVA views: reshape_and_cache_flash writes to host memory
        return caches

    module.allocate_kv_cache = allocate_kv_cache


def patch_model_runner(module) -> None:
    cls = module.GPUModelRunner
    initialize = cls.initialize_kv_cache

    def initialize_kv_cache(self, kv_cache_config, *args, **kwargs):
        _WORKER.update(vllm_config=self.vllm_config, host_group_ids=kv_cache_config.host_group_ids)
        initialize(self, kv_cache_config, *args, **kwargs)
        host_kv = state()
        if host_kv is not None:  # generic copy-on-write must not see host views (separate block ids)
            host = {view.data_ptr() for view in host_kv.views.values()}
            self.kv_caches = [cache for cache in self.kv_caches if cache.data_ptr() not in host]

    class KVBlockZeroer(module.KVBlockZeroer):

        def __init__(self, *args, skip_kv_cache_group_ids=(), **kwargs):
            skip = (*skip_kv_cache_group_ids, *_WORKER.get("host_group_ids", ()))
            super().__init__(*args, skip_kv_cache_group_ids=skip, **kwargs)

    copy_blocks = module.copy_kv_cache_blocks_inplace

    def copy_kv_cache_blocks_inplace(kv_caches, num_blocks, kv_cache_block_copies):
        gpu, host = split_copies(kv_cache_block_copies or [])
        copy_blocks(kv_caches, num_blocks, gpu)
        if host:
            state().copy_host_blocks(host)

    cls.initialize_kv_cache = initialize_kv_cache
    module.KVBlockZeroer = KVBlockZeroer
    module.copy_kv_cache_blocks_inplace = copy_kv_cache_blocks_inplace


def patch_warmup(module) -> None:
    warmup = module._warmup_kernels

    def _warmup_kernels(model_runner, worker_execute_model, worker_sample_tokens):
        config = model_runner.kv_cache_config
        host = [i for i, group in enumerate(config.kv_cache_groups) if group.role == ROLE]
        packed = {i: group.slots_per_block for i, group in enumerate(config.kv_cache_groups)
                  if group.slots_per_block > 1}
        execute = worker_execute_model
        if host or packed:
            def worker_execute_model(output):
                return execute(fold_warmup_block_ids(output, host, config.hisparse_host_num_blocks, packed))
        return warmup(model_runner, worker_execute_model, worker_sample_tokens)

    module._warmup_kernels = _warmup_kernels


def patch_model_state(module) -> None:
    cls = module.Qwen4ExpModelState
    prepare, prepare_dummy, add = cls.prepare_inputs, cls.prepare_dummy_inputs, cls.add_request

    def prepare_inputs(self, input_batch, req_states):
        if state() is not None:  # the drafter's forwards of this step use the same batch order
            state().set_requests(input_batch.idx_mapping, input_batch.num_reqs)
        return prepare(self, input_batch, req_states)

    def prepare_dummy_inputs(self, num_reqs, num_tokens):
        if state() is not None:  # capture and dummy runs leave every device buffer alone
            state().set_requests(None, 0)
        return prepare_dummy(self, num_reqs, num_tokens)

    def add_request(self, req_index, new_req_data):
        add(self, req_index, new_req_data)
        if state() is not None:
            state().reset_slot(req_index)

    cls.prepare_inputs, cls.prepare_dummy_inputs, cls.add_request = prepare_inputs, prepare_dummy_inputs, add_request


def patch_qsa_cache(module) -> None:
    cls = module.QSAMetadataBuilder
    build = cls.build

    def build_with_cpu_lengths(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = build(self, common_prefix_len, common_attn_metadata, fast_build)
        metadata.arc_query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        metadata.arc_seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        return metadata

    cls.build = build_with_cpu_lengths


def patch_qsa(module) -> None:
    impl = module.Qwen4ExpQSAFlashAttentionImpl
    forward = impl.forward_qsa

    def forward_qsa(self, layer, query, key, value, kv_cache, attn_metadata, output, token_to_req, use_prefill_config,
                    output_gate, output_scale=None, output_block_scale=None):
        host_kv = state()
        host = None if host_kv is None else host_kv.layers.get(getattr(layer, "layer_name", None))
        if host is None or output_scale is not None or output_block_scale is not None:
            return forward(self, layer, query, key, value, kv_cache, attn_metadata, output, token_to_req,
                           use_prefill_config, output_gate, output_scale=output_scale,
                           output_block_scale=output_block_scale)
        from vllm.forward_context import get_forward_context

        output.zero_()
        if attn_metadata.num_actual_tokens == 0:
            return output
        metadata = get_forward_context().attn_metadata
        side = (metadata[0] if isinstance(metadata, list) else metadata)[layer.indexer.raw_key_cache.prefix]
        host_kv.attend(layer, host, query, attn_metadata, side, output, token_to_req, use_prefill_config, output_gate)
        return output

    impl.forward_qsa = forward_qsa


def patch_qsa_ops(module) -> None:
    select = module._select_config

    def _select_config(num_rows, num_kv_heads, use_prefill_config, num_columns):
        return select(_FORCED_ROWS[-1] if _FORCED_ROWS else num_rows, num_kv_heads, use_prefill_config, num_columns)

    module._select_config = _select_config


def patches() -> dict:
    return {
        "vllm.v1.core.kv_cache_utils": patch_kv_cache_utils,
        "vllm.v1.core.single_type_kv_cache_manager": patch_single_type_manager,
        "vllm.v1.core.kv_cache_coordinator": patch_coordinator,
        "vllm.v1.core.kv_cache_manager": patch_kv_cache_manager,
        "vllm.v1.worker.gpu.attn_utils": patch_attn_utils,
        "vllm.v1.worker.gpu.model_runner": patch_model_runner,
        "vllm.v1.worker.gpu.warmup": patch_warmup,
        "vllm.models.qwen4_exp.nvidia.model_state": patch_model_state,
        "vllm.models.qwen4_exp.common.qsa_cache": patch_qsa_cache,
        "vllm.models.qwen4_exp.nvidia.qsa": patch_qsa,
        "vllm.models.qwen4_exp.nvidia.ops.qsa": patch_qsa_ops,
    }

from arc_patches import chain
if os.environ.get("ARC_VLLM_QSA_HOST_KV") == "1":
    for _name, _hook in patches().items():
        chain(_name, _hook)
