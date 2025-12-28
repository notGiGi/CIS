"""Utilities for attaching and managing residual stream hooks in transformer models.

This module provides functions to:
1. Capture activations from the residual stream at specific layers
2. Inject perturbations (deltas) into the residual stream
3. Manage and remove hooks cleanly

Scientific intent: Enable causal intervention on internal representations to test
how factual predictions depend on specific activation patterns.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _get_transformer_layer(model: Any, layer_idx: int) -> nn.Module:
    """Access the transformer layer module at the specified index.

    Works with Mistral, LLaMA, and similar architectures where layers are in model.model.layers.
    """
    try:
        # Try Mistral/LLaMA architecture first
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers[layer_idx]
        # Fallback for other architectures
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer.h[layer_idx]
        else:
            raise ValueError(
                f"Unknown model architecture. Expected 'model.model.layers' or 'transformer.h', "
                f"but got: {type(model)}"
            )
    except IndexError:
        total_layers = len(model.model.layers) if hasattr(model, 'model') else 'unknown'
        raise ValueError(
            f"Layer index {layer_idx} out of range. Model has {total_layers} layers."
        )


def add_residual_perturbation_hook(
    model: Any,
    layer_idx: int,
    delta_vector: torch.Tensor,
    token_position: int = -1,
) -> Tuple[Any, Callable]:
    """Attach a hook that adds a perturbation delta to the residual stream.

    Args:
        model: The transformer model (e.g., MistralForCausalLM)
        layer_idx: Which transformer layer to intervene on (0-indexed)
        delta_vector: Perturbation to add to hidden states [hidden_dim]
        token_position: Which token position to perturb (-1 = last token)

    Returns:
        handle: Hook handle that can be used to remove the hook
        hook_fn: The hook function itself (for inspection/debugging)

    Example:
        >>> handle, _ = add_residual_perturbation_hook(model, layer_idx=16, delta_vector=delta)
        >>> logits = model(input_ids)  # Model runs with intervention
        >>> handle.remove()  # Clean up
    """
    layer = _get_transformer_layer(model, layer_idx)

    # Ensure delta is on the same device as the model
    if delta_vector.device != next(model.parameters()).device:
        delta_vector = delta_vector.to(next(model.parameters()).device)

    def hook_fn(module: nn.Module, input: Tuple[torch.Tensor], output: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Hook function that intercepts and modifies the residual stream output.

        The output of a transformer layer is typically (hidden_states, ) or (hidden_states, attention_weights).
        We modify hidden_states at the specified token position.
        """
        # Output is typically a tuple; first element is hidden states
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Clone to avoid in-place modification issues
        modified_hidden = hidden_states.clone()

        # Apply perturbation to specified token position
        # hidden_states shape: [batch_size, seq_len, hidden_dim]
        if token_position == -1:
            # Last token position
            modified_hidden[:, -1, :] = modified_hidden[:, -1, :] + delta_vector
        else:
            # Specific token position
            modified_hidden[:, token_position, :] = modified_hidden[:, token_position, :] + delta_vector

        # Return modified output in same format as input
        if isinstance(output, tuple):
            return (modified_hidden,) + output[1:]
        else:
            return modified_hidden

    # Register the hook
    handle = layer.register_forward_hook(hook_fn)

    return handle, hook_fn


def register_residual_capture(
    model: Any,
    layer_idx: int,
    token_position: int = -1,
) -> Dict[str, Any]:
    """Register forward hooks to capture residual activations at a target layer and token.

    This is useful for:
    1. Inspecting what activations look like at a specific layer
    2. Setting up optimization targets for CIS

    Args:
        model: The transformer model
        layer_idx: Which layer to capture from
        token_position: Which token position to capture (-1 = last token)

    Returns:
        Dictionary containing:
            - 'handle': Hook handle for cleanup
            - 'activations': List that will be populated with captured tensors

    Example:
        >>> capture = register_residual_capture(model, layer_idx=16)
        >>> _ = model(input_ids)
        >>> captured = capture['activations'][0]  # [batch, seq, hidden]
        >>> capture['handle'].remove()
    """
    layer = _get_transformer_layer(model, layer_idx)

    # Storage for captured activations
    captured_activations: List[torch.Tensor] = []

    def capture_hook(module: nn.Module, input: Tuple[torch.Tensor], output: Tuple[torch.Tensor]) -> None:
        """Hook that captures activations without modifying them."""
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Capture the specified token position
        if token_position == -1:
            captured = hidden_states[:, -1, :].detach().clone()
        else:
            captured = hidden_states[:, token_position, :].detach().clone()

        captured_activations.append(captured)

    handle = layer.register_forward_hook(capture_hook)

    return {
        'handle': handle,
        'activations': captured_activations,
        'layer_idx': layer_idx,
        'token_position': token_position,
    }


def apply_residual_injection(
    model: Any,
    layer: int,
    token_position: int,
    delta: torch.Tensor,
) -> Any:
    """Attach a hook that adds a perturbation delta to the residual stream.

    This is an alias for add_residual_perturbation_hook for backward compatibility.

    Returns:
        handle: The hook handle (callable that can be removed)
    """
    handle, _ = add_residual_perturbation_hook(model, layer, delta, token_position)
    return handle


def clear_hooks(handles: Optional[List[Any]]) -> None:
    """Remove all provided hooks to avoid persistent graph modifications.

    Args:
        handles: List of hook handles or dict with 'handle' key, or single handle

    Example:
        >>> handle1 = add_residual_perturbation_hook(...)
        >>> handle2 = register_residual_capture(...)
        >>> clear_hooks([handle1, handle2['handle']])
    """
    if handles is None:
        return

    # Handle different input types
    if isinstance(handles, dict):
        # Single dict with 'handle' key
        if 'handle' in handles:
            handles['handle'].remove()
    elif isinstance(handles, list):
        # List of handles or dicts
        for h in handles:
            if isinstance(h, dict) and 'handle' in h:
                h['handle'].remove()
            elif hasattr(h, 'remove'):
                h.remove()
    elif hasattr(handles, 'remove'):
        # Single handle
        handles.remove()


def get_model_num_layers(model: Any) -> int:
    """Get the total number of transformer layers in the model.

    Useful for choosing intervention layers (e.g., middle layer).
    """
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return len(model.model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return len(model.transformer.h)
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")


def get_hidden_size(model: Any) -> int:
    """Get the hidden dimension size of the model.

    Useful for creating appropriately-sized delta vectors.
    """
    if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
        return model.config.hidden_size
    else:
        # Try to infer from first layer
        try:
            layer = _get_transformer_layer(model, 0)
            # Most layers have a self_attn or input_layernorm with hidden_size
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'hidden_size'):
                return layer.self_attn.hidden_size
            # Fallback: check model parameters
            for param in model.parameters():
                if len(param.shape) >= 2:
                    # Assume hidden_size is one of the dimensions
                    return param.shape[-1]
        except Exception:
            pass

    raise ValueError(f"Could not determine hidden size for model type: {type(model)}")
