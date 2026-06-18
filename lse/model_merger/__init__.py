"""Model loading / conversion helpers.

This package exists to keep LSE runnable without depending on the external `verl`
package. We vendor only the small subset of logic we need (e.g., converting
FSDP-sharded checkpoints to HuggingFace format for inference).
"""

from .fsdp_to_hf import resolve_model_path

__all__ = ["resolve_model_path"]

