# cis_factual_llm

Counterfactual Internal States (CIS) study the minimal activation perturbations that flip a model's factual prediction while keeping the underlying network weights frozen. This repository provides a research scaffold for probing factual knowledge representations in large language models using activation hooks rather than fine-tuning.

## What is a Counterfactual Internal State?
A Counterfactual Internal State is an activation pattern injected into a chosen layer and token position that steers the model from a ground-truth factual completion to a specified counterfactual alternative. The intervention is localized (layer, token position) and optimized to be as small as possible under an L2 metric.

## Geometric cost
The geometric cost of a CIS is defined as the minimal L2 norm of an additive activation perturbation that changes the model's predicted token from the factual answer to the counterfactual target. This scalar quantifies how resistant the model's internal representation is to counterfactual steering; lower cost indicates a more pliable factual representation.

## Factual pilot experiment
The pilot experiment targets a single fact such as "The Eiffel Tower is located in" using a frozen LLaMA-2-7B. We inject an activation vector into a mid-layer residual stream at the subject token position and optimize this vector to increase the logit of a counterfactual completion (e.g., " Rome") relative to the factual completion (e.g., " Paris"). Primary readouts include success rate of flipping the top-1 prediction, geometric cost of the perturbation, and collateral effects on nearby tokens.

## Repository layout
- `config/`: YAML configs for model loading and experiment hyperparameters.
- `data/`: Small CounterFact-style subset for quick smoke tests.
- `src/`: Modular code for loading models, constructing prompts, attaching hooks, searching for CIS, computing metrics, and orchestrating experiments.
- `notebooks/`: Lightweight notebooks for exploratory analyses and visualization.

## Notes
- All experiments assume access to a GPU and do not perform any parameter updates.
- Activation interventions are implemented via hooks; no training or fine-tuning routines are included.
