import logging
import os
from collections import OrderedDict

logger = logging.getLogger("vllm.arc_gdn_park")

PARK_BASE = -(1 << 40)  # below every host id of the QSA host K/V patch (-1 - id)
STATS_EVERY = int(os.environ.get("ARC_VLLM_GDN_PARK_STATS", "2000"))


def park_slots() -> int:
    return int(os.environ.get("ARC_VLLM_GDN_PARK", "0") or 0)


GROUPS = 256  # a host id carries its slot and the cache group whose pages it holds: PARK_BASE - (slot x GROUPS + group)


def encode_park(gpu_block: int, slot: int, group: int) -> tuple[int, int]:
    return gpu_block, PARK_BASE - (slot * GROUPS + group)


def encode_restore(slot: int, group: int, gpu_block: int) -> tuple[int, int]:
    return PARK_BASE - (slot * GROUPS + group), gpu_block


def split_park_copies(pairs) -> tuple[list, list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    other, parks, restores = [], [], []
    for pair in pairs:
        src, dst = pair[0], pair[1]
        if dst <= PARK_BASE:
            parks.append((src, *divmod(PARK_BASE - dst, GROUPS)))
        elif src <= PARK_BASE:
            restores.append((*divmod(PARK_BASE - src, GROUPS), dst))
        else:
            other.append(pair)
    return other, parks, restores


class ParkStore:

    def __init__(self, slots: int):
        self.free = list(range(slots - 1, -1, -1))
        self.entries: OrderedDict = OrderedDict()  # token hash -> {group id: slot}, least recently used first
        self.pending: list[tuple[int, int]] = []
        self.counts = {"parks": 0, "restores": 0, "evictions": 0, "restore_blocks": 0}

    def _take_slot(self) -> int | None:
        while not self.free and self.entries:
            _, old = self.entries.popitem(last=False)
            self.free.extend(old.values())
            self.counts["evictions"] += 1
        return self.free.pop() if self.free else None

    def park(self, token_hash, group_id: int, gpu_block: int) -> bool:
        entry = self.entries.get(token_hash)
        if entry is not None and group_id in entry:  # the same position's state: same bytes, already parked
            self.entries.move_to_end(token_hash)
            return False
        slot = self._take_slot()
        if slot is None:
            return False
        self.entries.setdefault(token_hash, {})[group_id] = slot
        self.entries.move_to_end(token_hash)
        self.pending.append(encode_park(gpu_block, slot, group_id))
        self.counts["parks"] += 1
        self._maybe_log()
        return True

    def slots_for(self, token_hash, group_ids) -> list[int] | None:
        entry = self.entries.get(token_hash)
        if entry is None or any(g not in entry for g in group_ids):
            return None
        return [entry[g] for g in group_ids]

    def restore(self, block_pool, token_hash, group_ids, num_tokens: int, make_key) -> list | None:
        slots = self.slots_for(token_hash, group_ids)
        if slots is None or block_pool.get_num_free_blocks() < len(group_ids):
            return None
        blocks = block_pool.get_new_blocks(len(group_ids))
        for block, group_id, slot in zip(blocks, group_ids, slots):
            block_pool._insert_block_hash(make_key(token_hash, group_id), block, num_tokens)
            self.pending.append(encode_restore(slot, group_id, block.block_id))
        block_pool.free_blocks(blocks)  # hashed: to the free queue's end, evicted last
        self.entries.move_to_end(token_hash)
        self.counts["restores"] += 1
        self.counts["restore_blocks"] += len(blocks)
        self._maybe_log()
        return blocks

    def take(self) -> list[tuple[int, int]]:
        pending, self.pending = self.pending, []
        return pending

    def _maybe_log(self) -> None:
        if (self.counts["parks"] + self.counts["restores"]) % STATS_EVERY == 0:
            logger.info("GDN park: %s, %d host entries, %d free slots", self.counts, len(self.entries), len(self.free))


class ParkingHolds(dict):

    def __init__(self, manager, items=()):
        super().__init__()
        self.manager = manager
        for key, value in dict(items).items():
            self[key] = value

    def __setitem__(self, key, value) -> None:
        position, block = value
        manager = self.manager
        if not block.is_null:
            if block.block_hash is not None:
                manager._park.park(manager._park_hash(block.block_hash), manager._park_group(block.block_hash),
                                   block.block_id)
            manager.block_pool.free_blocks([block])
            block = manager.block_pool.null_block
        super().__setitem__(key, (position, block))


HOLD_MAPS = ("_retained_checkpoints", "_arc_previous", "_arc_prompt_state")  # the engine's own, then arc_last_state's


def patch_single_type_manager(module) -> None:
    from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id, make_block_hash_with_group_id

    manager = module.MambaManager
    init = manager.__init__
    find = manager.find_longest_cache_hit.__func__
    slots = park_slots()

    def __init__(self, kv_cache_spec, block_pool, **kwargs):
        init(self, kv_cache_spec, block_pool, **kwargs)  # runs arc_last_state's __init__ first, so its hold maps exist
        self._park = None
        if self.mamba_cache_mode == "align" and self.enable_caching and getattr(self, "_retain_checkpoint", False):
            store = getattr(block_pool, "gdn_park", None)
            if store is None:
                store = block_pool.gdn_park = ParkStore(slots)
                logger.info("GDN park: %d host slots per worker; Mamba holds are parked on the host (%s)", slots,
                            [name for name in HOLD_MAPS if hasattr(self, name)])
            self._park, self._park_hash, self._park_group = store, get_block_hash, get_group_id
            for name in HOLD_MAPS:
                if hasattr(self, name):
                    setattr(self, name, ParkingHolds(self, getattr(self, name)))

    def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids, block_pool, kv_cache_spec,
                               drop_eagle_block, alignment_tokens, dcp_world_size=1, pcp_world_size=1):
        computed, hit = find(cls, block_hashes, max_length, kv_cache_group_ids, block_pool, kv_cache_spec,
                             drop_eagle_block, alignment_tokens, dcp_world_size, pcp_world_size)
        store = getattr(block_pool, "gdn_park", None)
        block_size = kv_cache_spec.block_size
        if store is None or not store.entries or not (alignment_tokens < block_size and block_size % alignment_tokens == 0):
            return computed, hit  # the block-granular lookup path does not restore
        hashes = module.resolve_block_hashes(block_hashes, block_pool.hash_block_size, block_size,
                                             supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
                                             alignment_tokens=alignment_tokens)
        unit, scale = alignment_tokens, block_size // alignment_tokens
        for idx in range(min(max_length // unit, len(hashes)) - 1, -1, -1):
            num_tokens = (idx + 1) * unit
            if num_tokens <= hit:
                break
            blocks = store.restore(block_pool, hashes[idx], kv_cache_group_ids, num_tokens, make_block_hash_with_group_id)
            if blocks is not None:
                return tuple([block_pool.null_block] * (idx // scale) + [block] for block in blocks), num_tokens
        return computed, hit

    manager.__init__ = __init__
    manager.find_longest_cache_hit = classmethod(find_longest_cache_hit)


def patch_kv_cache_manager(module) -> None:
    take = module.KVCacheManager.take_kv_cache_block_copies

    def take_kv_cache_block_copies(self):
        copies, retained = take(self)
        store = getattr(self.block_pool, "gdn_park", None)
        pairs = store.take() if store is not None else []
        if not pairs:
            return copies, retained
        return [*copies, *(module.KVCacheBlockCopy(src, dst) for src, dst in pairs)], retained

    module.KVCacheManager.take_kv_cache_block_copies = take_kv_cache_block_copies


_WORKER: dict = {}  # "slab": HostSlab


def mamba_group_ranges(kv_cache_config) -> tuple[dict[int, list[tuple[int, int]]], int, int]:
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

    page, group_of = {}, {}
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        if getattr(group, "host_resident", False):
            continue
        spec = group.kv_cache_spec
        per = spec.kv_cache_specs if isinstance(spec, UniformTypeKVCacheSpecs) else {n: spec for n in group.layer_names}
        for name, layer_spec in per.items():
            if isinstance(layer_spec, MambaSpec) and name in group.layer_names:
                page[name], group_of[name] = layer_spec.page_size_bytes, gid
    ranges: dict[int, list[tuple[int, int]]] = {}
    strides, sizes = set(), set()
    for tensor in kv_cache_config.kv_cache_tensors:
        if getattr(tensor, "host_resident", False):
            continue
        for i, name in enumerate(tensor.layers):
            if name in page:
                strides.add(tensor.block_stride)
                sizes.add(tensor.size)
                ranges.setdefault(group_of[name], []).append((tensor.offset + i * tensor.layer_stride, page[name]))
    if not ranges or len(strides) != 1 or len(sizes) != 1:
        raise ValueError(f"GDN park: Mamba pages over {len(strides)} block strides / {len(sizes)} allocations")
    stride = strides.pop()
    merged = {}
    for gid, spans in ranges.items():
        out = []
        for start, length in sorted(spans):
            if start < 0 or start + length > stride:
                raise ValueError(f"GDN park needs a block-outermost layout: group {gid} bytes {start}+{length} > {stride}")
            if out and out[-1][0] + out[-1][1] == start:
                out[-1] = (out[-1][0], out[-1][1] + length)
            else:
                out.append((start, length))
        merged[gid] = out
    return merged, stride, sizes.pop()


class HostSlab:

    def __init__(self, kv_caches, num_blocks: int, slots: int, layout):
        import torch

        self.ranges, stride, size = layout
        self.view = None
        for cache in kv_caches:
            storage = cache.untyped_storage()
            if storage.nbytes() != size:
                continue
            blocks = torch.empty(0, dtype=torch.uint8, device=cache.device)
            blocks.set_(storage)
            self.view = blocks.view(num_blocks, -1)
            break
        if self.view is None or self.view.shape[1] != stride:
            raise ValueError(f"GDN park: no KV cache storage of {size} B with {stride} B blocks")
        width = max(sum(length for _, length in spans) for spans in self.ranges.values())
        self.slab = torch.empty((slots, width), dtype=torch.uint8)
        register_host_memory(self.slab)
        logger.info("GDN park: %d slots x %d B on the host (%.2f GB); group ranges %s", slots, width,
                    self.slab.numel() / 1e9, {g: len(r) for g, r in self.ranges.items()})

    def apply(self, parks, restores) -> None:
        # a group with no pages on this worker (pipeline parallel: a merged Mamba group's layers all on another stage) has nothing to copy here
        for slot, group, block in restores:
            at = 0
            for begin, length in self.ranges.get(group, ()):
                self.view[block, begin:begin + length].copy_(self.slab[slot, at:at + length], non_blocking=True)
                at += length
        for block, slot, group in parks:
            at = 0
            for begin, length in self.ranges.get(group, ()):
                self.slab[slot, at:at + length].copy_(self.view[block, begin:begin + length], non_blocking=True)
                at += length


def patch_model_runner(module) -> None:
    cls = module.GPUModelRunner
    initialize = cls.initialize_kv_cache
    copy_blocks = module.copy_kv_cache_blocks_inplace
    slots = park_slots()

    def initialize_kv_cache(self, kv_cache_config, *args, **kwargs):
        initialize(self, kv_cache_config, *args, **kwargs)
        # self.kv_caches holds no QSA host K/V views: that patch's initialize_kv_cache, called above, drops them
        _WORKER["slab"] = HostSlab(self.kv_caches, kv_cache_config.num_blocks, slots,
                                   mamba_group_ranges(kv_cache_config))

    def copy_kv_cache_blocks_inplace(kv_caches, num_blocks, kv_cache_block_copies):
        other, parks, restores = split_park_copies(kv_cache_block_copies or [])
        slab = _WORKER.get("slab")
        if (parks or restores) and slab is None:
            raise RuntimeError("GDN park: host copies without a parking slab")
        # Restores first (a same-step copy-on-write may read a restored block), parks last (a parked copy-on-write target is filled by this step's GPU copy)
        if restores:
            slab.apply([], restores)
        copy_blocks(kv_caches, num_blocks, other)
        if parks:
            slab.apply(parks, [])

    cls.initialize_kv_cache = initialize_kv_cache
    module.copy_kv_cache_blocks_inplace = copy_kv_cache_blocks_inplace


def patches() -> dict:
    return {
        "vllm.v1.core.single_type_kv_cache_manager": patch_single_type_manager,
        "vllm.v1.core.kv_cache_manager": patch_kv_cache_manager,
        "vllm.v1.worker.gpu.model_runner": patch_model_runner,
    }

from arc_patches import chain
from arc_qsa_host_kv import register_host_memory
if os.environ.get("ARC_VLLM_GDN_PARK", "0") not in ("", "0"):  # after the last-state hooks (its hold maps)
    for _name, _hook in patches().items():
        chain(_name, _hook)
