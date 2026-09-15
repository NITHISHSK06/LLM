# Phase 10 Report: Coding Domain Specialization

## Objective

Specialize the existing small instruction-tuned Transformer on manually authored programming examples while preserving the original checkpoints and measuring general-capability retention.

## Dataset Description

The manually authored coding dataset is stored in `data/raw/coding_instruction.jsonl`. It covers Python, Java, C/C++, data structures, algorithms, OOP, SQL, debugging, code explanation, code completion, complexity, programming concepts, code improvement, and error diagnosis.

## Current Data Quality

The quality audit is saved in `evaluation/coding_data_quality_report.json`. It currently reports 34 valid examples, no malformed records, no duplicate prompts, and no duplicate prompt-response pairs. Some code examples trigger whitespace/repeated-character heuristics because code formatting is intentionally present. The prepared split contains 31 training examples and 3 validation examples.

## Training Configuration

The reusable trainer is `training/domain_train.py`. It preserves the model architecture, uses AdamW, gradient clipping, CUDA mixed precision when available, checkpoint resume, early stopping, and configurable `general_data_ratio` with a default of 0.2. It prefers `checkpoints/instruct_v2/best_model.pt` and falls back to `checkpoints/instruct/best_model.pt`.

## Training Results

No real Phase 10 training result is reported yet. The current workspace does not contain the required source checkpoint files. Temporary smoke tests verified the training step and safe checkpoint loading only.

## Coding Evaluation

The 50-prompt coding suite is in `evaluation/coding_test_prompts.json`. The evaluator is `evaluation/domain_evaluate.py`. Real coding metrics remain unavailable until the three checkpoints exist.

## General Capability Retention

The separate 14-prompt retention suite is in `evaluation/general_retention_prompts.json`. No retention claim is made until the instruction-tuned and coding-specialized checkpoints are evaluated on exactly the same prompts.

## Comparison and Human Evaluation

Machine-readable comparison output is `evaluation/coding_model_comparison.json`. Manual coding review belongs in `evaluation/human_coding_evaluation.md`. Generated code must be checked by a human or an appropriate local test before correctness is claimed.

## Catastrophic Forgetting Analysis

The configurable general-data mixture is intended to provide a controlled comparison at ratios such as 0.0, 0.1, 0.2, and 0.3. No conclusion is possible before those experiments and retention results are run.

## Best Configuration

Not established. The best configuration must be selected using coding relevance, instruction following, grammar, coherence, repetition, response length, validation loss, and retention results together.

## Limitations

This is a small research and educational language model, not a production coding assistant. A lower validation loss does not prove better code, factuality, safety, or conversational behavior. The current dataset is intentionally small and manually authored.

## Recommended Next Experiment

Provide the existing base and instruction-tuned checkpoints, then run the coding specialization with `--general_data_ratio 0.0`, `0.2`, and `0.3`, keeping the seed, prompt sets, and evaluation procedure fixed.
