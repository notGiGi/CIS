"""Prompt templates for eliciting factual completions."""

from typing import Dict, List


def make_factual_prompt(subject: str, relation: str) -> str:
    """Construct a minimal, declarative factual prompt.

    Scientific intent: keep wording neutral so the next-token distribution reflects
    factual recall rather than stylistic framing.
    """
    subject = subject.strip()
    relation = relation.strip()
    return f"The {subject} is {relation}"


def build_factual_prompt(subject: str, relation: str, template_id: str = "factual_prompt_v1") -> str:
    """Alias to construct the default factual prompt template."""
    if template_id != "factual_prompt_v1":
        raise ValueError(f"Unknown template_id: {template_id}")
    return make_factual_prompt(subject, relation)


def available_templates() -> List[str]:
    """Return the list of registered prompt template identifiers."""
    return ["factual_prompt_v1"]


def load_prompt_bank(path: str) -> Dict[str, str]:
    """Placeholder for loading a prompt bank; unused in the baseline sanity check."""
    raise NotImplementedError
