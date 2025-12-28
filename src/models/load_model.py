"""Model loading utilities for causal language models used in CIS experiments."""

import os
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging as hf_logging


def _resolve_device(device_override: Optional[str] = None) -> str:
    """Choose GPU when available to mirror deployment-time inference."""
    if device_override:
        if device_override.startswith("cuda") and not torch.cuda.is_available():
            hf_logging.get_logger(__name__).warning(
                "CUDA requested but not available; falling back to CPU."
            )
            return "cpu"
        return device_override
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_bitsandbytes_available() -> None:
    """Validate bitsandbytes is importable for 4-bit quantization."""
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "bitsandbytes is required for 4-bit loading. Install it with "
            "`pip install bitsandbytes`. On Windows you may need a CUDA-matched wheel."
        ) from exc


def load_model_and_tokenizer(model_config: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any]:
    """Load a frozen causal LM and tokenizer for inference-only factual probes.

    Supports both full-precision (FP16/BF16) and 4-bit quantized loading.
    4-bit quantization can be enabled via use_4bit=true in config.
    """
    model_config = model_config or {}
    model_name = model_config.get("model_name", "mistralai/Mistral-7B-v0.1")
    device = _resolve_device(model_config.get("device"))
    revision = model_config.get("revision")
    cache_dir = model_config.get("cache_dir")
    trust_remote_code = bool(model_config.get("trust_remote_code", False))
    tokenizer_padding_side = model_config.get("tokenizer_padding_side", "left")
    local_files_only = bool(model_config.get("local_files_only", False))
    device_map_cfg = model_config.get("device_map", "auto")
    offload_folder = model_config.get("offload_folder")
    max_memory = model_config.get("max_memory")
    use_4bit = bool(model_config.get("use_4bit", False))
    use_flash_attention = bool(model_config.get("use_flash_attention", False))

    # Auth token is required for gated models like LLaMA-2.
    auth_token = (
        model_config.get("auth_token")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )

    # Prefer configured dtype on GPU; fall back to fp32 on CPU to avoid unsupported dtype errors.
    dtype_str = model_config.get("dtype", "float16")
    torch_dtype = getattr(torch, dtype_str, torch.float16) if device.startswith("cuda") else torch.float32

    logger = hf_logging.get_logger(__name__)

    if not torch.cuda.is_available() and use_4bit:
        logger.warning("CUDA is not available; 4-bit quantization may fail on CPU.")

    try:
        quantization_config = None

        # Only configure 4-bit quantization if explicitly requested
        if use_4bit:
            _ensure_bitsandbytes_available()
            quant_type = model_config.get("bnb_4bit_quant_type", "nf4")
            use_double_quant = bool(model_config.get("bnb_4bit_use_double_quant", True))
            bnb_compute_dtype = model_config.get("bnb_4bit_compute_dtype", dtype_str)
            bnb_torch_dtype = getattr(torch, bnb_compute_dtype, torch_dtype)

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=bnb_torch_dtype,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_use_double_quant=use_double_quant,
            )
            logger.info(f"Loading model with 4-bit quantization (quant_type={quant_type})")
        else:
            logger.info(f"Loading model in {dtype_str} precision (no quantization)")

        # Tokenizer setup ensures consistent tokenization for locating subject tokens.
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            padding_side=tokenizer_padding_side,
            trust_remote_code=trust_remote_code,
            use_fast=False,
            token=auth_token,
            local_files_only=local_files_only,
        )
        if tokenizer.pad_token is None:
            # Stable padding token prevents accidental truncation when batching later.
            tokenizer.pad_token = tokenizer.eos_token

        # Model loading kwargs
        model_kwargs = {
            "revision": revision,
            "cache_dir": cache_dir,
            "device_map": device_map_cfg if device_map_cfg is not None else "auto",
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": trust_remote_code,
            "token": auth_token,
            "local_files_only": local_files_only,
        }

        # Add optional parameters only if they're set
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if offload_folder is not None:
            model_kwargs["offload_folder"] = offload_folder
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory

        # Enable Flash Attention 2 if requested and available
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Attempting to load model with Flash Attention 2")

        # Try loading with Flash Attention 2 first, fallback to default if it fails
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except (ImportError, ValueError) as e:
            if use_flash_attention and "flash_attn" in str(e).lower():
                logger.warning(
                    "Flash Attention 2 is not available (flash-attn not installed). "
                    "Falling back to default attention implementation."
                )
                # Remove flash attention requirement and retry
                model_kwargs.pop("attn_implementation", None)
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            else:
                raise
        model.eval()

        # Freeze all parameters for inference-only use
        for param in model.parameters():
            param.requires_grad_(False)

        logger.info(f"Model loaded successfully on device: {device}")

    except Exception as exc:
        logger.error(
            "Failed to load model %s. Ensure credentials are set if required "
            "(export HUGGINGFACE_HUB_TOKEN or set auth_token in config) or set "
            "local_files_only=true if weights are cached locally.",
            model_name,
        )
        raise exc

    return model, tokenizer


def load_causal_lm(model_config: Dict[str, Any]) -> Tuple[Any, Any]:
    """Backward-compatible alias for loading the model and tokenizer."""
    return load_model_and_tokenizer(model_config)


def prepare_model_for_hooks(model: Any, layers: Optional[Tuple[int, ...]] = None) -> Any:
    """Placeholder for future activation hook preparation."""
    raise NotImplementedError
