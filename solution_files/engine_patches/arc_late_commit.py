import logging
import os

logger = logging.getLogger("vllm.arc_late_commit")


def patch_model_runner(module) -> None:
    cls = module.GPUModelRunner
    sample = cls.sample_tokens

    def sample_tokens(self, grammar_output):
        if not (getattr(self, "is_last_pp_rank", False) and getattr(self, "pp_handler", None) is not None
                and getattr(self, "speculator", None) is not None):
            return sample(self, grammar_output)
        state = self.model_state
        own = state.__dict__.get("postprocess_state")
        commit, deferred = state.postprocess_state, []
        state.postprocess_state = lambda *args, **kwargs: deferred.append((args, kwargs))
        try:
            return sample(self, grammar_output)
        finally:
            if own is None:
                del state.postprocess_state
            else:
                state.postprocess_state = own
            for args, kwargs in deferred:
                commit(*args, **kwargs)

    cls.sample_tokens = sample_tokens
    logger.info("ARC_VLLM_LATE_COMMIT: the last pipeline stage commits RecoverSSM states after the draft broadcast")


def patches() -> dict:
    return {"vllm.v1.worker.gpu.model_runner": patch_model_runner}

from arc_patches import chain
if os.environ.get("ARC_VLLM_LATE_COMMIT") == "1":  # after the host-KV runner hook
    for _name, _hook in patches().items():
        chain(_name, _hook)
