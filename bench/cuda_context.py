"""A CUDA context held for a whole measured run (measurement gate decision D2).

Without a CUDA context in the measuring process, CodeCarbon's NVML energy
counter stalls for about half a second whenever the GPU has been idle, which
inflates the gross energy and duration of a pipeline that never touches the GPU
itself (the baseline, whose LLM runs in a separate Ollama process). Both
runners therefore hold a context for the entire run so the comparison is fair.
"""

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

CUDA_DEVICE = "cuda"
CONTEXT_PROBE_SIZE = 1


class CudaContextHolder(Protocol):
    """Keeps a CUDA context alive between `acquire()` and `release()`."""

    def acquire(self) -> bool:
        """Create the context; return whether one is now held."""
        ...

    def release(self) -> None:
        """Drop whatever `acquire` allocated."""
        ...


class NoOpCudaContextHolder:
    """Holds nothing; used in tests and on machines without CUDA."""

    def acquire(self) -> bool:
        """Report that no context is held."""
        return False

    def release(self) -> None:
        """Nothing to release."""


class TorchCudaContextHolder:
    """Creates the context by allocating one tiny tensor on the GPU.

    The context itself lives as long as the process; `release()` only drops the
    tensor. torch is imported lazily so this module stays cheap to import.
    """

    def __init__(self) -> None:
        self._probe: object | None = None

    def acquire(self) -> bool:
        """Allocate the probe tensor; False when CUDA is not available."""
        import torch

        if not torch.cuda.is_available():
            logger.warning("CUDA is not available; no context is held")
            return False
        self._probe = torch.zeros(CONTEXT_PROBE_SIZE, device=CUDA_DEVICE)
        torch.cuda.synchronize()
        return True

    def release(self) -> None:
        """Drop the probe tensor."""
        self._probe = None
