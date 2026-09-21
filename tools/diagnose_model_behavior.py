"""Read-only evidence collection for model, tokenizer, and SFT behavior."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.inference.generate import generate_tokens
from model.config import ModelConfig
from model.model import DecoderLanguageModel
from tokenizer.tokenizer import SimpleBPETokenizer
from training.instruct_train import InstructionExample, encode_example

TOKENIZER_DIR = PROJECT_ROOT / "tokenizer_exp003"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "diagnostics"
TEXT_OUTPUT = OUTPUT_DIR / "model_behavior_diagnostic.txt"
JSON_OUTPUT = OUTPUT_DIR / "model_behavior_diagnostic.json"

MODEL_PATHS = {
    "A Base pretrained": PROJECT_ROOT / "checkpoints/exp012_capacity_25m/best_model.pt",
    "B Exp012/Exp011 SFT": PROJECT_ROOT / "checkpoints/exp012_capacity_25m_sft/best_model.pt",
    "C Exp014 v2 SFT": PROJECT_ROOT / "checkpoints/exp014_general_chat_v2_25m/best_model.pt",
}

PROMPTS = [
    "hi", "hello", "hey", "good morning", "good night", "bye", "thank you",
    "can you help me?", "what is ai?", "what are you doing?", "I had a bad day",
    "I am happy", "I am sad", "I like music", "I do not like music", "How are you?",
]

CONTEXT_PAIRS = [
    ("I am happy", "I am sad"),
    ("I like music", "I do not like music"),
    ("good morning", "good night"),
    ("thank you", "can you help me?"),
]
CONDITIONING_PROMPTS = [
    "How are you?", "What is your favorite color?", "What is your favorite food?",
    "Can you help me?", "Tell me about music.", "Tell me about programming.",
]
ATTRACTOR_PHRASES = [
    "I", "I am", "I am glad", "I hope", "It is nice", "I understand", "I have",
    "I have been", "you", "the", "of", "and",
]
CANDIDATE_LABELS = [
    "I", "You", "Hello", "Hi", "Hey", "Good", "Thank", "I\u2019m", "I am",
    "Sorry", "Yes", "No", "It", "The", "A",
]


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def token_text(tokenizer: SimpleBPETokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)


def tokenization_record(tokenizer: SimpleBPETokenizer, prompt: str) -> dict[str, Any]:
    token_ids = tokenizer.encode(prompt, add_bos=True)
    decoded = [token_text(tokenizer, token_id) for token_id in token_ids]
    return {
        "prompt": prompt,
        "token_ids": token_ids,
        "decoded_tokens": decoded,
        "prompt_tokens": len(token_ids),
        "unk_present": tokenizer.unk_token_id in token_ids,
    }


def load_model(path: Path, device: torch.device) -> tuple[DecoderLanguageModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    checkpoint_config = checkpoint.get("model_config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"Checkpoint does not contain model_config: {path}")
    model = DecoderLanguageModel(ModelConfig(**checkpoint_config)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def final_hidden_states(model: DecoderLanguageModel, input_ids: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    hidden = model.token_embedding(input_ids) + model.position_embedding(positions)
    for block in model.blocks:
        hidden = block(hidden)
    return model.final_norm(hidden)


@torch.inference_mode()
def next_token_analysis(
    model: DecoderLanguageModel,
    tokenizer: SimpleBPETokenizer,
    prompt: str,
    device: torch.device,
    top_n: int = 20,
) -> dict[str, Any]:
    token_ids = tokenizer.encode(prompt, add_bos=True)
    context_ids = token_ids[-model.config.context_length:]
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
    logits, _ = model(input_ids)
    final_logits = logits[0, -1]
    probabilities = torch.softmax(final_logits, dim=-1)
    top_probabilities, top_ids = torch.topk(probabilities, min(top_n, probabilities.shape[0]))
    ranks = torch.argsort(final_logits, descending=True)
    rank_by_id = torch.empty_like(ranks)
    rank_by_id[ranks] = torch.arange(1, len(ranks) + 1, device=device)
    top = []
    for rank, (token_id, probability) in enumerate(zip(top_ids.tolist(), top_probabilities.tolist()), 1):
        top.append({
            "token_id": int(token_id),
            "token": token_text(tokenizer, int(token_id)),
            "probability": float(probability),
            "rank": rank,
        })
    candidates = {}
    for label in CANDIDATE_LABELS:
        candidate_ids = tokenizer.encode(label, add_bos=False)
        if not candidate_ids:
            candidates[label] = {"token_id": None, "probability": None, "rank": None}
            continue
        candidate_id = candidate_ids[0]
        candidates[label] = {
            "token_id": int(candidate_id),
            "token": token_text(tokenizer, candidate_id),
            "probability": float(probabilities[candidate_id]),
            "rank": int(rank_by_id[candidate_id]),
        }
    return {
        "prompt": prompt,
        "input_token_ids": context_ids,
        "top20": top,
        "candidates": candidates,
        "probabilities": probabilities.detach().cpu().tolist(),
        "final_hidden": final_hidden_states(model, input_ids)[0, -1].detach().cpu().tolist(),
        "logits_finite": bool(torch.isfinite(final_logits).all().item()),
        "hidden_finite": bool(torch.isfinite(final_hidden_states(model, input_ids)).all().item()),
    }


def distribution_metrics(first: dict[str, Any], second: dict[str, Any], tokenizer: SimpleBPETokenizer) -> dict[str, Any]:
    first_probs = torch.tensor(first["probabilities"], dtype=torch.float64)
    second_probs = torch.tensor(second["probabilities"], dtype=torch.float64)
    epsilon = 1e-12
    p = first_probs.clamp_min(epsilon)
    q = second_probs.clamp_min(epsilon)
    midpoint = ((p + q) / 2).clamp_min(epsilon)
    kl_pq = float((p * (p.log() - q.log())).sum())
    kl_qp = float((q * (q.log() - p.log())).sum())
    js = float(0.5 * (p * (p.log() - midpoint.log())).sum() + 0.5 * (q * (q.log() - midpoint.log())).sum())
    first_top = {item["token_id"] for item in first["top20"]}
    second_top = {item["token_id"] for item in second["top20"]}
    changes = torch.abs(first_probs - second_probs)
    changed_ids = torch.topk(changes, min(10, len(changes))).indices.tolist()
    changed_tokens = [
        {
            "token_id": int(token_id),
            "token": token_text(tokenizer, int(token_id)),
            "first_probability": float(first_probs[token_id]),
            "second_probability": float(second_probs[token_id]),
            "absolute_change": float(changes[token_id]),
        }
        for token_id in changed_ids
    ]
    return {
        "top20_overlap_count": len(first_top & second_top),
        "top20_overlap_fraction": len(first_top & second_top) / 20.0,
        "kl_first_to_second": kl_pq,
        "kl_second_to_first": kl_qp,
        "js_divergence": js,
        "largest_probability_changes": changed_tokens,
        "meaningfully_changed": bool(js >= 0.01 or len(first_top & second_top) <= 15),
    }


def generate_record(
    model: DecoderLanguageModel,
    tokenizer: SimpleBPETokenizer,
    prompt: str,
    device: torch.device,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    all_ids = generate_tokens(
        model, tokenizer, prompt, 30, 1.0, 0, 1.0, device, greedy=True,
    )
    new_ids = all_ids[len(prompt_ids):]
    return {
        "prompt": prompt,
        "generated_token_ids": new_ids,
        "output": tokenizer.decode(new_ids),
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if isinstance(record, dict) and isinstance(record.get("prompt"), str) and isinstance(record.get("response"), str):
                records.append(record)
    return records


def raw_sft_measurements(
    record: dict[str, Any], tokenizer: SimpleBPETokenizer, context_length: int,
) -> dict[str, Any]:
    example = InstructionExample(record["prompt"].strip(), record["response"].strip())
    prefix = f"User: {example.prompt}\nAssistant: "
    prefix_ids = tokenizer.encode(prefix, add_bos=True)
    response_ids = tokenizer.encode(example.response, add_eos=True)
    all_ids = prefix_ids + response_ids
    max_sequence_length = context_length + 1
    truncated = len(all_ids) > max_sequence_length
    if truncated:
        all_ids = all_ids[-max_sequence_length:]
    response_start = len(prefix_ids)
    if truncated:
        response_start -= len(prefix_ids + response_ids) - max_sequence_length
    response_target_positions = [
        index - 1 for index in range(max(response_start, 1), len(all_ids))
        if index < len(all_ids) and index >= response_start
    ]
    zero_targets = not any(position >= 0 and position < len(all_ids) - 1 for position in response_target_positions)
    encoded = None
    if not zero_targets:
        try:
            encoded = encode_example(example, tokenizer, context_length)
        except ValueError:
            zero_targets = True
    return {
        "example": example,
        "encoded": encoded,
        "prompt_tokens": len(prefix_ids),
        "response_tokens": len(response_ids),
        "truncated": truncated,
        "zero_targets": zero_targets,
    }


@torch.inference_mode()
def evaluate_sft_records(
    model: DecoderLanguageModel,
    tokenizer: SimpleBPETokenizer,
    records: list[dict[str, Any]],
    device: torch.device,
    seed: int,
    include_generations: bool,
) -> dict[str, Any]:
    rng = random.Random(seed)
    selected = records.copy()
    rng.shuffle(selected)
    selected = selected[: min(100, len(selected))]
    measurements = [raw_sft_measurements(item, tokenizer, model.config.context_length) for item in selected]
    valid = [item for item in measurements if item["encoded"] is not None]
    losses = []
    ranks = []
    details = []
    for item in valid:
        encoded = item["encoded"]
        input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=device)
        target_ids = torch.tensor([encoded.target_ids], dtype=torch.long, device=device)
        logits, loss = model(input_ids, target_ids)
        assert loss is not None
        first_position = next(index for index, target in enumerate(encoded.target_ids) if target != -100)
        first_target = encoded.target_ids[first_position]
        rank = int((logits[0, first_position] > logits[0, first_position, first_target]).sum().item()) + 1
        losses.append(float(loss.item()))
        ranks.append(rank)
        detail = {
            "prompt": item["example"].prompt,
            "response": item["example"].response,
            "loss": float(loss.item()),
            "first_response_token_id": int(first_target),
            "first_response_token": token_text(tokenizer, first_target),
            "first_response_token_rank": rank,
            "prompt_tokens": item["prompt_tokens"],
            "response_tokens": item["response_tokens"],
            "truncated": item["truncated"],
            "zero_targets": item["zero_targets"],
        }
        if include_generations:
            generated = generate_tokens(model, tokenizer, f"User: {item['example'].prompt}\nAssistant: ", 30, 1.0, 0, 1.0, device, greedy=True)
            formatted_prompt_ids = tokenizer.encode(f"User: {item['example'].prompt}\nAssistant: ", add_bos=True)
            detail["greedy_response"] = tokenizer.decode(generated[len(formatted_prompt_ids):])
        details.append(detail)
    if not losses:
        raise ValueError("No valid SFT examples were available for evaluation.")
    rank_counts = {"top1": 0, "top5": 0, "top10": 0, "top50": 0, "over50": 0}
    for rank in ranks:
        if rank == 1:
            rank_counts["top1"] += 1
        if rank <= 5:
            rank_counts["top5"] += 1
        if rank <= 10:
            rank_counts["top10"] += 1
        if rank <= 50:
            rank_counts["top50"] += 1
        if rank > 50:
            rank_counts["over50"] += 1
    sample_count = len(measurements)
    return {
        "sample_count": sample_count,
        "valid_model_examples": len(valid),
        "teacher_forced_loss": sum(losses) / len(losses),
        "perplexity": math.exp(min(sum(losses) / len(losses), 20.0)),
        "mean_prompt_tokens": sum(item["prompt_tokens"] for item in measurements) / sample_count,
        "mean_response_tokens": sum(item["response_tokens"] for item in measurements) / sample_count,
        "context_truncation_percent": 100.0 * sum(item["truncated"] for item in measurements) / sample_count,
        "zero_valid_response_target_percent": 100.0 * sum(item["zero_targets"] for item in measurements) / sample_count,
        "first_response_rank_percent": {key: 100.0 * value / len(ranks) for key, value in rank_counts.items()},
        "details": details,
    }


def attractor_analysis(
    model: DecoderLanguageModel,
    tokenizer: SimpleBPETokenizer,
    next_records: list[dict[str, Any]],
) -> dict[str, Any]:
    output = {}
    for phrase in ATTRACTOR_PHRASES:
        candidate_ids = tokenizer.encode(phrase, add_bos=False)
        if not candidate_ids:
            continue
        token_id = candidate_ids[0]
        probabilities = [record["probabilities"][token_id] for record in next_records]
        ranks = []
        for record in next_records:
            sorted_ids = sorted(range(len(record["probabilities"])), key=lambda index: record["probabilities"][index], reverse=True)
            ranks.append(sorted_ids.index(token_id) + 1)
        output[phrase] = {
            "token_id": token_id,
            "mean_probability": sum(probabilities) / len(probabilities),
            "max_probability": max(probabilities),
            "mean_rank": sum(ranks) / len(ranks),
            "top20_count": sum(rank <= 20 for rank in ranks),
        }
    return output


def conditioning_analysis(
    model: DecoderLanguageModel,
    tokenizer: SimpleBPETokenizer,
    device: torch.device,
) -> dict[str, Any]:
    records = [next_token_analysis(model, tokenizer, prompt, device) for prompt in CONDITIONING_PROMPTS]
    pairwise = []
    for first_index in range(len(records)):
        for second_index in range(first_index + 1, len(records)):
            first_hidden = F.normalize(torch.tensor(records[first_index]["final_hidden"], dtype=torch.float64), dim=0)
            second_hidden = F.normalize(torch.tensor(records[second_index]["final_hidden"], dtype=torch.float64), dim=0)
            metrics = distribution_metrics(records[first_index], records[second_index], tokenizer)
            metrics.update({
                "first_prompt": CONDITIONING_PROMPTS[first_index],
                "second_prompt": CONDITIONING_PROMPTS[second_index],
                "hidden_cosine_similarity": float(torch.dot(first_hidden, second_hidden)),
            })
            pairwise.append(metrics)
    return {
        "prompts": CONDITIONING_PROMPTS,
        "pairwise": pairwise,
        "mean_hidden_cosine": sum(item["hidden_cosine_similarity"] for item in pairwise) / len(pairwise),
        "mean_js_divergence": sum(item["js_divergence"] for item in pairwise) / len(pairwise),
        "mean_top20_overlap": sum(item["top20_overlap_fraction"] for item in pairwise) / len(pairwise),
    }


def sanity_check(model: DecoderLanguageModel, tokenizer: SimpleBPETokenizer, device: torch.device) -> dict[str, Any]:
    config = model.config
    parameter_values = [parameter.detach() for parameter in model.parameters()]
    smoke_ids = torch.tensor([tokenizer.encode("hi", add_bos=True)], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits, _ = model(smoke_ids)
        hidden = final_hidden_states(model, smoke_ids)
    return {
        "parameter_count": model.parameter_count(),
        "embedding_dim": config.embedding_dim,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "context_length": config.context_length,
        "model_vocab_size": config.vocab_size,
        "tokenizer_vocab_size": len(tokenizer.vocab),
        "tokenizer_vocab_matches_model": len(tokenizer.vocab) == config.vocab_size,
        "checkpoint_parameters_finite": all(torch.isfinite(value).all().item() for value in parameter_values),
        "forward_logits_finite": bool(torch.isfinite(logits).all().item()),
        "forward_hidden_finite": bool(torch.isfinite(hidden).all().item()),
    }


def smoke_test(device: torch.device) -> None:
    tokenizer = SimpleBPETokenizer.load(TOKENIZER_DIR)
    model, _ = load_model(MODEL_PATHS["A Base pretrained"], device)
    token_record = tokenization_record(tokenizer, "hi")
    next_record = next_token_analysis(model, tokenizer, "hi", device)
    generation = generate_record(model, tokenizer, "hi", device)
    sanity = sanity_check(model, tokenizer, device)
    print("SMOKE TEST PASSED")
    print(json.dumps({"tokenization": token_record, "top20": next_record["top20"], "generation": generation, "sanity": sanity}, indent=2))


def model_status(model_result: dict[str, Any]) -> dict[str, str]:
    sanity = model_result["sanity"]
    if not sanity["tokenizer_vocab_matches_model"] or not sanity["checkpoint_parameters_finite"]:
        tokenizer_status = "FAIL"
    elif any(item["unk_present"] for item in model_result["tokenization"]):
        tokenizer_status = "SUSPICIOUS"
    else:
        tokenizer_status = "PASS"
    if sanity["forward_logits_finite"] and sanity["forward_hidden_finite"]:
        forward_status = "PASS"
    else:
        forward_status = "FAIL"
    conditioning = model_result["conditioning"]
    conditioning_status = "SUSPICIOUS" if conditioning["mean_top20_overlap"] > 0.9 and conditioning["mean_js_divergence"] < 0.01 else "PASS"
    return {"tokenizer": tokenizer_status, "forward": forward_status, "conditioning": conditioning_status}


def final_report(results: dict[str, Any]) -> dict[str, Any]:
    tokenizer_unk = sum(item["unk_present"] for result in results["models"].values() for item in result["tokenization"])
    tokenizer_total = sum(len(result["tokenization"]) for result in results["models"].values())
    forward_ok = all(result["sanity"]["forward_logits_finite"] and result["sanity"]["forward_hidden_finite"] for result in results["models"].values())
    conditioning_values = [result["conditioning"]["mean_js_divergence"] for result in results["models"].values()]
    mean_conditioning_js = sum(conditioning_values) / len(conditioning_values)
    sft_results = results["sft_evaluation"]
    train_valid_gap = sft_results["B Exp012/Exp011 SFT"]["train"]["teacher_forced_loss"] - sft_results["B Exp012/Exp011 SFT"]["validation"]["teacher_forced_loss"]
    attractor = results["models"]["B Exp012/Exp011 SFT"]["attractors"]
    generic_top20 = sorted(attractor.items(), key=lambda item: item[1]["top20_count"], reverse=True)[:3]
    candidate_evidence = {
        "1. tokenizer/data representation": {
            "evidence_for": f"{tokenizer_unk}/{tokenizer_total} diagnostic prompts contain UNK; model/tokenizer vocab matches are reported per checkpoint.",
            "evidence_against": "No tokenizer mismatch is inferred unless the explicit vocabulary check fails.",
            "confidence": "LOW" if tokenizer_unk == 0 else "MEDIUM",
        },
        "2. model forward/causal attention": {
            "evidence_for": "Non-finite parameters, logits, or hidden states would support this location.",
            "evidence_against": f"All finite checks passed: {forward_ok}.",
            "confidence": "LOW",
        },
        "3. pretraining optimization": {
            "evidence_for": "Base-model next-token and generation behavior are recorded for direct inspection.",
            "evidence_against": "This diagnostic does not establish optimization causality from behavior alone.",
            "confidence": "LOW",
        },
        "4. insufficient pretraining/generalization": {
            "evidence_for": "Base generation and SFT train/validation metrics are included; Exp011 loss gap is %.6f." % train_valid_gap,
            "evidence_against": "A loss gap alone is not semantic accuracy and does not isolate pretraining from SFT effects.",
            "confidence": "MEDIUM" if train_valid_gap > 0.25 else "LOW",
        },
        "5. SFT data/conditioning": {
            "evidence_for": "SFT validation metrics, exact-format samples, and prompt-conditioned distributions are included.",
            "evidence_against": "The report does not infer a causal SFT defect from generic responses alone.",
            "confidence": "LOW",
        },
        "6. inference/decoding": {
            "evidence_for": "Greedy decoding outputs are reported separately from logits and teacher-forced metrics.",
            "evidence_against": "Greedy output alone cannot explain a mismatch when next-token probabilities also show it.",
            "confidence": "LOW",
        },
        "7. experiment-control/configuration": {
            "evidence_for": "Checkpoint paths, model dimensions, tokenizer dimensions, and all loaded artifacts are recorded.",
            "evidence_against": "This script does not inspect historical commands or prove every prior run used its intended arguments.",
            "confidence": "MEDIUM",
        },
        "8. insufficient evidence": {
            "evidence_for": "Several observations are correlational and the diagnostic does not test causal interventions.",
            "evidence_against": "The report provides direct numerical evidence for tokenization, forward pass, conditioning, generation, and SFT fit.",
            "confidence": "HIGH" if mean_conditioning_js < 0.01 else "MEDIUM",
        },
    }
    likely = "8. insufficient evidence"
    if not forward_ok:
        likely = "2. model forward/causal attention"
    elif tokenizer_unk > tokenizer_total * 0.1:
        likely = "1. tokenizer/data representation"
    elif train_valid_gap > 0.25:
        likely = "4. insufficient pretraining/generalization"
    results["final_report"] = {
        "A. TOKENIZER": {"evidence": f"{tokenizer_unk}/{tokenizer_total} prompts contain an UNK token.", "status": "SUSPICIOUS" if tokenizer_unk else "PASS"},
        "B. MODEL FORWARD PASS": {"evidence": f"All finite checks passed: {forward_ok}.", "status": "PASS" if forward_ok else "FAIL"},
        "C. PROMPT CONDITIONING": {"evidence": f"Mean JS divergence across conditioning prompt pairs: {mean_conditioning_js:.6f}.", "status": "SUSPICIOUS" if mean_conditioning_js < 0.01 else "PASS"},
        "D. BASE MODEL GENERATION": {"evidence": "Thirty-token greedy outputs for all prompts are included in the model sections."},
        "E. SFT EFFECT": {"evidence": "Exp012 and Exp014 teacher-forced validation metrics and greedy outputs are included."},
        "F. TRAIN VS VALIDATION": {"evidence": f"Exp011 SFT train minus validation loss: {train_valid_gap:.6f}."},
        "G. GENERIC ATTRACTOR BEHAVIOR": {"evidence": "Top attractor token probability/rank statistics are included; most frequent top-20 tokens: " + ", ".join(item[0] for item in generic_top20)},
        "H. MOST LIKELY FAILURE LOCATION": {"choice": likely, "candidates": candidate_evidence},
    }
    return results


def format_text(results: dict[str, Any]) -> str:
    lines = ["MODEL BEHAVIOR DIAGNOSTIC REPORT", "=" * 80, "", "Read-only diagnostic. No training, checkpoint, or tokenizer changes were performed.", ""]
    lines.append("MODEL SANITY")
    lines.append("-" * 80)
    for name, result in results["models"].items():
        lines.append(f"{name}: {result['checkpoint']}")
        lines.append(json.dumps(result["sanity"], indent=2))
    lines.append("")
    for name, result in results["models"].items():
        lines.extend([name.upper(), "=" * 80, "", "PART 1 - TOKENIZATION"])
        for item in result["tokenization"]:
            lines.append(json.dumps(item, ensure_ascii=False))
        lines.append("\nPART 2 - NEXT-TOKEN LOGIT ANALYSIS")
        for item in result["next_token"]:
            lines.append(f"PROMPT: {item['prompt']}")
            lines.append("TOP 20:")
            for candidate in item["top20"]:
                lines.append(json.dumps(candidate, ensure_ascii=False))
            lines.append("CANDIDATES:")
            lines.append(json.dumps(item["candidates"], ensure_ascii=False))
        lines.append("\nPART 3 - CONTEXT SENSITIVITY")
        for item in result["context_sensitivity"]:
            lines.append(json.dumps(item, ensure_ascii=False))
        lines.append("\nPART 4 - GENERIC ATTRACTOR ANALYSIS")
        lines.append(json.dumps(result["attractors"], ensure_ascii=False, indent=2))
        lines.append("\nPART 5 - SHORT GREEDY GENERATION")
        for item in result["generation"]:
            lines.append(f"PROMPT: {item['prompt']}\nOUTPUT: {item['output']}")
        lines.append("\nPART 8 - PROMPT CONDITIONING TEST")
        lines.append(json.dumps(result["conditioning"], ensure_ascii=False, indent=2))
        lines.append("")
    lines.extend(["PART 6 - TEACHER-FORCED SFT EVALUATION", "=" * 80])
    for name, data in results["sft_evaluation"].items():
        lines.append(f"{name}")
        for split, metrics in data.items():
            lines.append(f"{split.upper()} SUMMARY")
            summary = {key: value for key, value in metrics.items() if key != "details"}
            lines.append(json.dumps(summary, indent=2))
    lines.extend(["\nPART 7 - MEMORIZATION / GENERALIZATION CHECK", "=" * 80])
    for split, metrics in results["sft_evaluation"]["B Exp012/Exp011 SFT"].items():
        lines.append(f"{split.upper()} DETAILS")
        for detail in metrics["details"]:
            lines.append(json.dumps(detail, ensure_ascii=False))
    lines.extend(["\nPART 10 - FINAL DIAGNOSTIC REPORT", "=" * 80])
    for section, value in results["final_report"].items():
        lines.append(section)
        lines.append(json.dumps(value, ensure_ascii=False, indent=2))
    return "\n".join(lines) + "\n"


def run_full(device: torch.device) -> None:
    tokenizer = SimpleBPETokenizer.load(TOKENIZER_DIR)
    results: dict[str, Any] = {
        "configuration": {
            "tokenizer": TOKENIZER_DIR,
            "models": MODEL_PATHS,
            "prompts": PROMPTS,
            "seed": 42,
            "device": str(device),
        },
        "models": {},
        "sft_evaluation": {},
    }
    for name, checkpoint_path in MODEL_PATHS.items():
        print(f"Loading {name}...")
        model, checkpoint = load_model(checkpoint_path, device)
        tokenization = [tokenization_record(tokenizer, prompt) for prompt in PROMPTS]
        next_token = [next_token_analysis(model, tokenizer, prompt, device) for prompt in PROMPTS]
        context_sensitivity = []
        for first_prompt, second_prompt in CONTEXT_PAIRS:
            first = next_token_analysis(model, tokenizer, first_prompt, device)
            second = next_token_analysis(model, tokenizer, second_prompt, device)
            context_sensitivity.append({"first_prompt": first_prompt, "second_prompt": second_prompt, **distribution_metrics(first, second, tokenizer)})
        generation = [generate_record(model, tokenizer, prompt, device) for prompt in PROMPTS]
        conditioning = conditioning_analysis(model, tokenizer, device)
        sanity = sanity_check(model, tokenizer, device)
        sanity["checkpoint_loaded"] = True
        results["models"][name] = {
            "checkpoint": checkpoint_path,
            "checkpoint_keys": sorted(checkpoint.keys()),
            "tokenization": tokenization,
            "next_token": next_token,
            "context_sensitivity": context_sensitivity,
            "attractors": attractor_analysis(model, tokenizer, next_token),
            "generation": generation,
            "conditioning": conditioning,
            "sanity": sanity,
            "status": model_status({"sanity": sanity, "tokenization": tokenization, "conditioning": conditioning}),
        }
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    exp011_train = load_records(PROJECT_ROOT / "data/processed/exp011_general_chat/instruction_train.jsonl")
    exp011_val = load_records(PROJECT_ROOT / "data/processed/exp011_general_chat/instruction_val.jsonl")
    exp012_val = exp011_val
    exp014_val = load_records(PROJECT_ROOT / "data/processed/exp014_general_chat_v2/instruction_val.jsonl")
    for name, dataset in (("B Exp012/Exp011 SFT", exp012_val), ("C Exp014 v2 SFT", exp014_val)):
        model, _ = load_model(MODEL_PATHS[name], device)
        results["sft_evaluation"][name] = {
            "validation": evaluate_sft_records(model, tokenizer, dataset, device, 42, False),
        }
        if name == "B Exp012/Exp011 SFT":
            results["sft_evaluation"][name]["train"] = evaluate_sft_records(model, tokenizer, exp011_train, device, 42, True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    results = final_report(results)
    TEXT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEXT_OUTPUT.write_text(format_text(results), encoding="utf-8")
    JSON_OUTPUT.write_text(json.dumps(json_value(results), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FULL DIAGNOSTIC COMPLETE")
    print(f"Text report: {TEXT_OUTPUT}")
    print(f"JSON report: {JSON_OUTPUT}")
    print(f"Most likely failure location: {results['final_report']['H. MOST LIKELY FAILURE LOCATION']['choice']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only model behavior diagnostics.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one model and one prompt only.")
    parser.add_argument("--full", action="store_true", help="Run all diagnostics and write reports.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if args.smoke_test:
        smoke_test(device)
        return
    if not args.full:
        raise SystemExit("Choose --smoke-test or --full.")
    run_full(device)


if __name__ == "__main__":
    main()
