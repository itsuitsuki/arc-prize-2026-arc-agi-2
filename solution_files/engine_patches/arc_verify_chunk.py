import os

from arc_patches import chain
if os.environ.get("ARC_VLLM_VERIFY_CHUNK_SIZE"):
    chain("vllm.v1.worker.gpu.spec_decode.rejection_sampler",
          lambda module, size=int(os.environ["ARC_VLLM_VERIFY_CHUNK_SIZE"]): setattr(module, "MAX_CHUNK_BYTES", size))
