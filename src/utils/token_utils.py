"""Token-level utilities for encoding prompts and inspecting next-token predictions."""

from typing import Any, Iterable, List, Optional, Sequence, Tuple

import torch


def encode_prompt(prompt: str, tokenizer: Any, device: str) -> dict:
    """Tokenize a prompt and move tensors to the target device for inference."""
    inputs = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def _find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> int:
    """Return the start index of a subsequence within a sequence or -1 if absent."""
    for idx in range(len(sequence) - len(subsequence) + 1):
        if list(sequence[idx : idx + len(subsequence)]) == list(subsequence):
            return idx
    return -1


def find_subject_span(prompt: str, subject: str, tokenizer: Any) -> Tuple[int, int]:
    """Return the token span corresponding to the subject mention within the prompt.

    Scientific intent: knowing the exact token span lets us map activations to the
    subject representation we intend to perturb or measure.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    subject_ids = tokenizer.encode(subject, add_special_tokens=False)
    start = _find_subsequence(prompt_ids, subject_ids)
    if start == -1:
        raise ValueError("Subject tokens not found in prompt tokens.")
    end = start + len(subject_ids)
    return start, end


def subject_last_token_index(prompt: str, subject: str, tokenizer: Any) -> int:
    """Return the index of the last subject token within the tokenized prompt."""
    _, end = find_subject_span(prompt, subject, tokenizer)
    return end - 1


def locate_intervention_token(token_ids: List[int], token_index: int) -> int:
    """Resolve an intervention position given a relative index (e.g., -1 for last)."""
    return token_index if token_index >= 0 else len(token_ids) + token_index


def decode_tokens(tokenizer: Any, token_ids: Iterable[int]) -> str:
    """Convert token IDs back to text for debugging interventions."""
    return tokenizer.decode(list(token_ids))


def get_next_token_logits(model: Any, tokenizer: Any, prompt: str, device: str) -> torch.Tensor:
    """Compute logits for the next token following the prompt."""
    inputs = encode_prompt(prompt, tokenizer, device=device)
    with torch.no_grad():
        outputs = model(**inputs)
    # The final position logits correspond to the next-token distribution we probe.
    return outputs.logits[:, -1, :]


def decode_topk_predictions(tokenizer: Any, logits: torch.Tensor, k: int = 5) -> List[dict]:
    """Decode the top-k tokens and probabilities from next-token logits."""
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k)
    top_probs = top_probs.squeeze(0).tolist()
    top_ids = top_ids.squeeze(0).tolist()
    decoded = []
    for token_id, prob in zip(top_ids, top_probs):
        # Using tokenizer.decode preserves leading spaces (e.g., \" Paris\").
        decoded.append({"token_id": token_id, "token_str": tokenizer.decode([token_id]), "prob": float(prob)})
    return decoded


def print_topk_predictions(
    tokenizer: Any,
    logits: torch.Tensor,
    k: int = 5,
    expected: Optional[str] = None,
) -> List[dict]:
    """Decode and print the top-k next-token predictions with probabilities.

    Scientific intent: the ranked distribution is the baseline for later CIS
    interventions, so we log the exact next-token probabilities.
    """
    predictions = decode_topk_predictions(tokenizer, logits, k=k)
    for rank, pred in enumerate(predictions, start=1):
        token_str = pred["token_str"]
        marker = ""
        if expected is not None and token_str.strip() == expected.strip():
            marker = " <-- expected factual object"
        print(f"{rank}. {token_str!r}\tprob={pred['prob']:.4f}{marker}")
    return predictions
