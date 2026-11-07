# Rewrites part of the Merlin engine's GDNRecoverSSMCommitContext.commit (Apache License 2.0) at run time; see THIRD_PARTY_NOTICES.md.
import logging
import os
import time

logger = logging.getLogger("vllm.arc_last_state")

RELEASE_UNITS = 2  # hash units past the newest decode state before the previous hold is released
PROTECT_SECONDS = float(os.environ.get("ARC_VLLM_RETAIN_LAST_STATE_PROTECT_SECONDS", "0"))


def committed_tokens(num_computed: int, num_in_flight: int, num_tokens: int) -> int:
    return max(0, min(num_computed - num_in_flight, num_tokens - 1))


def tail_key(committed: int, num_prompt_tokens: int, unit: int, block: int) -> int:
    state = (committed // unit * unit - unit) // block * block
    return state + unit if state > num_prompt_tokens else 0


def decode_block_ends(num_prompt_tokens: int, num_tokens: int, block: int) -> range:
    return range((num_prompt_tokens // block + 1) * block, num_tokens + 1, block)


def states_to_unhash(positions: list[int], floor: int, reach: int, unit: int) -> list[int]:
    decode = [p for p in positions if p > floor]
    keep = max((p for p in decode if p <= reach - unit), default=None)
    return [p for p in decode if p != keep]


def unhash(pool, blocks, floor: int, shared_too: bool = False) -> None:
    for block in blocks:
        if (block.block_hash is not None and block.block_hash_num_tokens > floor
                and (shared_too or block.ref_cnt == 1)):
            pool._maybe_evict_cached_block(block)


def patch_single_type_manager(module) -> None:
    cls = module.MambaManager
    init, cache, retire = cls.__init__, cls.cache_blocks, cls._retire_state_blocks
    skipped, pop_retained, pop_blocks = cls.remove_skipped_blocks, cls.pop_retained_checkpoints, cls.pop_blocks_for_free
    add_hit = cls.add_local_computed_blocks

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if self.mamba_cache_mode == "align" and self.enable_caching and not self._retain_checkpoint:
            raise ValueError("ARC_VLLM_RETAIN_LAST_STATE needs VLLM_E2_MAMBA_RETAIN_CHECKPOINT=1")
        self._arc_last_state = self._retain_checkpoint  # align mode with prefix caching
        self._arc_prompt: dict[str, int] = {}  # request -> num_prompt_tokens
        self._arc_previous: dict[str, tuple] = {}  # request -> (position, block) held until the newest is usable
        self._arc_reach: dict[str, int] = {}  # request -> T (committed tokens rounded down to 64), set just before its free
        self._arc_hit: dict[str, int] = {}  # request -> prefix-cache hit at admission
        self._arc_prompt_state: dict[str, tuple] = {}  # request -> (position, block): its prompt state, held until free

    def release(self, request_id: str, blocks) -> None:
        unhash(self.block_pool, blocks, self._arc_prompt[request_id])
        self.block_pool.free_blocks(blocks)

    def release_previous(self, request_id: str) -> None:
        _, block = self._arc_previous.pop(request_id)
        unhash(self.block_pool, [block], 0)
        self.block_pool.free_blocks([block])

    def add_local_computed_blocks(self, request_id, new_computed_blocks, num_local_computed_tokens, *args, **kwargs):
        add_hit(self, request_id, new_computed_blocks, num_local_computed_tokens, *args, **kwargs)
        if self._arc_last_state:
            self._arc_hit[request_id] = num_local_computed_tokens

    def cache_blocks(self, request, num_tokens, retention_interval=None, *, replay_boundaries):
        if not self._arc_last_state:
            return cache(self, request, num_tokens, retention_interval, replay_boundaries=replay_boundaries)
        if request.request_id not in self._arc_prompt and self._arc_hit.get(request.request_id, 0) > request.num_prompt_tokens:
            hold_prompt_tail(self, request)
        prompt = self._arc_prompt[request.request_id] = request.num_prompt_tokens
        # The mask keeps every listed boundary's state: the prompt's own plus every block end past the prompt
        ends = decode_block_ends(prompt, num_tokens, self.block_size)
        return cache(self, request, num_tokens, 0, replay_boundaries=(*replay_boundaries, *ends))

    def hold_prompt_tail(self, request) -> None:
        unit = self.block_pool.hash_block_size
        tail = request.num_prompt_tokens // unit * unit - unit
        cached = tail > 0 and self.block_pool.get_cached_block(request.block_hashes[tail // unit - 1], [self.kv_cache_group_id])
        if cached:
            self.block_pool.touch(cached)
            self._arc_prompt_state[request.request_id] = (tail, cached[0])

    def _retire_state_blocks(self, request_id, retired):
        if not self._arc_last_state or request_id not in self._arc_prompt:
            return retire(self, request_id, retired)
        newest = max(((b.block_hash_num_tokens, b) for _, b in retired if b.block_hash is not None),
                     key=lambda entry: entry[0], default=None)
        held = self._retained_checkpoints.get(request_id)
        if newest is None or (held is not None and newest[0] <= held[0]):
            return release(self, request_id, [b for _, b in retired])
        freed = [b for _, b in retired if b is not newest[1]]
        self._retained_checkpoints[request_id] = newest  # the table's ref becomes the hold's
        prompt = self._arc_prompt[request_id]
        if held is not None:
            if newest[0] <= prompt or (held[0] <= prompt and request_id in self._arc_prompt_state):
                freed.append(held[1])  # superseded inside the prompt, as in stock
            elif held[0] <= prompt:  # the first decode state: the prompt state stays held until free, as in stock
                self._arc_prompt_state[request_id] = held
            else:  # a decode state: unusable until the attention key passes it
                if request_id in self._arc_previous:
                    release_previous(self, request_id)
                self._arc_previous[request_id] = held
        release(self, request_id, freed)

    def remove_skipped_blocks(self, request_id, processed_computed_tokens, num_prompt_tokens=None):
        skipped(self, request_id, processed_computed_tokens, num_prompt_tokens)
        previous = self._arc_previous.get(request_id) if self._arc_last_state else None
        held = self._retained_checkpoints.get(request_id)
        unit = self.block_pool.hash_block_size
        if previous is not None and (held is None or held[0] + RELEASE_UNITS * unit <= processed_computed_tokens):
            release_previous(self, request_id)

    def pop_retained_checkpoints(self, request_id):
        held = self._retained_checkpoints.get(request_id) if self.mamba_cache_mode == "align" else None
        blocks = pop_retained(self, request_id)
        if not self._arc_last_state:
            return blocks
        previous = self._arc_previous.pop(request_id, None)
        fallback = self._arc_prompt_state.pop(request_id, None)
        if previous is None and held is None and fallback is None:
            return blocks
        holds = [h for h in (previous, held) if h is not None]
        # States at or below the request's own prefix hit were complete before it was admitted and keep their hash
        floor = max(self._arc_prompt.get(request_id, 0), self._arc_hit.get(request_id, 0))
        drop = states_to_unhash([p for p, _ in holds], floor, self._arc_reach.get(request_id, 0),
                                self.block_pool.hash_block_size)
        unhash(self.block_pool, [b for p, b in holds if p in drop], floor, shared_too=True)
        # Freed, and so evicted, in this order: the prompt-tail state last, because the LRU evicts a request's decode-tail block first and the next turn must still resume at the prompt tail then
        return [b for _, b in holds] + ([fallback[1]] if fallback else [])

    def pop_blocks_for_free(self, request_id):
        blocks = pop_blocks(self, request_id)
        prompt, reach = self._arc_prompt.pop(request_id, None), self._arc_reach.pop(request_id, 0)
        hit = self._arc_hit.pop(request_id, 0)
        if prompt is not None:  # table states above T - 64: under pipeline parallelism their stage-0 commit may not have run
            unit = self.block_pool.hash_block_size
            unhash(self.block_pool, [b for b in blocks if b.block_hash is not None
                                     and b.block_hash_num_tokens > reach - unit], max(prompt, hit), shared_too=True)
        return blocks

    cls.__init__, cls.cache_blocks, cls._retire_state_blocks = __init__, cache_blocks, _retire_state_blocks
    cls.remove_skipped_blocks, cls.pop_retained_checkpoints = remove_skipped_blocks, pop_retained_checkpoints
    cls.pop_blocks_for_free, cls.add_local_computed_blocks = pop_blocks_for_free, add_local_computed_blocks


def patch_kv_cache_manager(module) -> None:
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager

    cls = module.KVCacheManager
    init, free, pop, pin = cls.__init__, cls.free, cls.pop_blocks_for_free, cls.pin_waiting_hit
    yield_pins = cls.yield_pinned_hits

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._arc_last_state = self.enable_caching
        self._arc_protected: dict[str, float] = {}  # finished request -> deadline of its grace pin's protection
        if self._arc_last_state:
            logger.info("ARC_VLLM_RETAIN_LAST_STATE: each request keeps its newest usable decode GDN state; protection "
                        "%.0f s", PROTECT_SECONDS)

    def gdn_managers(self) -> list:
        return [m for m in self.coordinator.single_type_managers if isinstance(m, MambaManager) and m._arc_last_state]

    def retained(self, request_id: str) -> list:
        out = []
        for manager in gdn_managers(self):
            floor = max(manager._arc_prompt.get(request_id, 0), manager._arc_hit.get(request_id, 0))
            reach = manager._arc_reach.get(request_id, 0) - manager.block_pool.hash_block_size
            holds = [h for h in (manager._retained_checkpoints.get(request_id), manager._arc_previous.get(request_id))
                     if h is not None and floor < h[0] <= reach]
            out.append(max(holds, key=lambda h: h[0], default=None))
        return out

    def attention_managers(self) -> list:
        return [m for m in self.coordinator.single_type_managers if isinstance(m, FullAttentionManager)
                and m.enable_caching and m.kv_cache_spec.prefix_cacheable]

    def publish_tail(self, request) -> None:
        unit = self.block_pool.hash_block_size
        committed = committed_tokens(request.num_computed_tokens, request.num_in_flight_tokens, request.num_tokens)
        for manager in self.coordinator.single_type_managers:
            if isinstance(manager, MambaManager) and manager._arc_last_state:
                manager._arc_reach[request.request_id] = committed // unit * unit
        for manager in attention_managers(self):
            key = tail_key(committed, request.num_prompt_tokens, unit, manager.block_size)
            blocks = manager.req_to_blocks.get(request.request_id, ())
            if not key or key // unit > len(request.block_hashes) or key // manager.block_size >= len(blocks):
                continue
            block = blocks[key // manager.block_size]
            if not block.is_null:  # an extra key: the block keeps its prompt-tail key, its K/V is append-only
                manager.block_pool._insert_block_hash(
                    make_block_hash_with_group_id(request.block_hashes[key // unit - 1], manager.kv_cache_group_id),
                    block, num_tokens=key)

    def after_free(self, request, kept: list) -> None:
        request_id = request.request_id
        protect = bool(kept) and all(kept) and len({p for p, _ in kept}) == 1 and request_id in self._grace_pins
        if protect and PROTECT_SECONDS > 0:
            self._arc_protected[request_id] = time.monotonic() + PROTECT_SECONDS

    def yield_pinned_hits(self, keep=None, limit=None):
        if not self._arc_last_state:
            return yield_pins(self, keep, limit)
        now = time.monotonic()
        for request_id in [r for r, deadline in self._arc_protected.items() if deadline < now or r not in self._grace_pins]:
            del self._arc_protected[request_id]
        hidden = {r: self._grace_pins.pop(r) for r in list(self._grace_pins) if r in self._arc_protected}
        try:
            released = yield_pins(self, keep, limit)
        finally:
            self._grace_pins.update(hidden)
        if not released or limit is None:
            for request_id in list(self._arc_protected)[:limit]:  # oldest first
                released += self.release_grace_pin(request_id)
                del self._arc_protected[request_id]
        return released

    def pin_waiting_hit(self, request) -> int:
        pinned = pin(self, request)
        if not (pinned and self._arc_last_state):
            return pinned
        _, hit, _ = self.get_computed_blocks(request)  # the stock pin's lookup again, once per waiter
        unit, held = self.block_pool.hash_block_size, self._pinned_hits[request.request_id]
        for manager in attention_managers(self):
            for key in range(min(hit + manager.block_size, len(request.block_hashes) * unit), hit, -unit):
                cached = manager.block_pool.get_cached_block(request.block_hashes[key // unit - 1],
                                                             [manager.kv_cache_group_id])
                if cached:
                    if all(block is not cached[0] for block in held):
                        manager.block_pool.touch(cached)  # its own pool: the host K/V group's is private
                        held.append(cached[0])
                        pinned += 1
                    break
        for manager in self.coordinator.single_type_managers:
            if not (isinstance(manager, MambaManager) and manager._arc_last_state) or hit % manager.block_size:
                continue
            for key in range(hit - unit, 0, -unit):  # the attention blocks below the hit are pinned already
                cached = manager.block_pool.get_cached_block(request.block_hashes[key // unit - 1],
                                                             [manager.kv_cache_group_id])
                if cached:
                    if all(block is not cached[0] for block in held):
                        manager.block_pool.touch(cached)
                        held.insert(0, cached[0])  # release_pinned_hit frees in reverse: this one last
                        pinned += 1
                    break
        return pinned

    def free_with_tail(self, request):
        if not self._arc_last_state:
            return free(self, request)
        publish_tail(self, request)
        kept = retained(self, request.request_id)
        free(self, request)
        after_free(self, request, kept)

    def pop_with_tail(self, request):
        if not self._arc_last_state:
            return pop(self, request)
        publish_tail(self, request)
        kept = retained(self, request.request_id)
        blocks = pop(self, request)
        after_free(self, request, kept)
        return blocks

    cls.__init__, cls.free, cls.pop_blocks_for_free = __init__, free_with_tail, pop_with_tail
    cls.pin_waiting_hit, cls.yield_pinned_hits = pin_waiting_hit, yield_pinned_hits


BOUNDARY_ORDER = "for boundary in (False, True) if align else (False,):"


def patch_recoverssm_commit(module) -> None:
    import inspect
    import textwrap

    cls = module.GDNRecoverSSMCommitContext
    source = textwrap.dedent(inspect.getsource(cls.commit))
    if source.count(BOUNDARY_ORDER) != 1:
        raise RuntimeError("GDNRecoverSSMCommitContext.commit has changed; the boundary-first launch order cannot be applied")
    namespace: dict = {}
    exec(compile(source.replace(BOUNDARY_ORDER, "for boundary in (True, False) if align else (False,):"),
                 inspect.getsourcefile(cls.commit), "exec"), module.__dict__, namespace)
    cls.commit = namespace["commit"]
    logger.info("ARC_VLLM_RETAIN_LAST_STATE: GDN RecoverSSM commit launches boundary states first")


def patches() -> dict:
    return {
        "vllm.v1.core.single_type_kv_cache_manager": patch_single_type_manager,
        "vllm.v1.core.kv_cache_manager": patch_kv_cache_manager,
        "vllm.model_executor.layers.mamba.gdn.recoverssm_gdn": patch_recoverssm_commit,
    }

from arc_patches import chain
if os.environ.get("ARC_VLLM_RETAIN_LAST_STATE") == "1":  # after the host-KV hooks on the same modules
    for _name, _hook in patches().items():
        chain(_name, _hook)
