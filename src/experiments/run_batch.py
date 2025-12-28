"""Batch runner for CIS experiments over multiple facts."""

import argparse
from typing import Any, Dict, Iterable


def load_batch(path: str) -> Iterable[Dict[str, Any]]:
    """Load a collection of factual instances for batch processing."""
    raise NotImplementedError


def run_batch(config_path: str) -> None:
    """Run CIS search across a dataset and aggregate metrics."""
    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch execution."""
    parser = argparse.ArgumentParser(description="Run batch CIS experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    return parser.parse_args()


def main() -> None:
    """CLI wrapper for executing batch CIS experiments."""
    args = parse_args()
    run_batch(args.config)


if __name__ == "__main__":
    main()
