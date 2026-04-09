"""
V_chat_sft: drop-in replacement for scripts.chat_sft that uses V_gpt
(safe-mean NaN fix) instead of gpt.

Patches sys.modules so that checkpoint_manager and all downstream imports
pick up nanochat.V_gpt as nanochat.gpt before any model code loads.

Run with:
    torchrun --nproc_per_node=2 -m scripts.V_chat_sft -- <same args as chat_sft>
"""
import sys
import nanochat.V_gpt as _v_gpt

# Inject V_gpt before checkpoint_manager (or anything else) imports nanochat.gpt
sys.modules["nanochat.gpt"] = _v_gpt

# Running chat_sft as a module executes all top-level code (training loop included).
# All CLI args in sys.argv are forwarded transparently via argparse inside chat_sft.
import scripts.chat_sft  # noqa: F401, E402
