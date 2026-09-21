from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from threading import Lock

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from experiments.inference.generate import generate_response, load_model_and_tokenizer

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "Exp012"


def _path_from_env(name: str, default: Path) -> Path:
	value = os.getenv(name)
	return Path(value) if value else default


class ModelService:
	def __init__(self) -> None:
		self.model = None
		self.tokenizer = None
		self.device: torch.device | None = None
		self.parameter_count = 0
		self._generation_lock = Lock()

	def load(self) -> None:
		if self.model is not None:
			return

		checkpoint_path = _path_from_env(
			"EXP012_CHECKPOINT",
			PROJECT_ROOT / "checkpoints" / "exp012_capacity_25m_sft" / "best_model.pt",
		)
		tokenizer_dir = _path_from_env(
			"EXP012_TOKENIZER",
			PROJECT_ROOT / "tokenizer_exp003",
		)
		LOGGER.info("Loading tokenizer...")
		LOGGER.info("Loading Exp012 checkpoint...")
		model, tokenizer, device = load_model_and_tokenizer(
			checkpoint_path=checkpoint_path,
			tokenizer_dir=tokenizer_dir,
		)
		self.model = model
		self.tokenizer = tokenizer
		self.device = device
		self.parameter_count = sum(parameter.numel() for parameter in model.parameters())
		LOGGER.info("Device: %s", device.type.upper())
		LOGGER.info("Parameters: %.2fM", self.parameter_count / 1_000_000)
		LOGGER.info("Model ready.")

	def health(self) -> dict[str, str | int]:
		if self.model is None or self.device is None:
			raise RuntimeError("Model is not loaded")
		return {
			"status": "ok",
			"model": MODEL_NAME,
			"parameters": self.parameter_count,
			"device": self.device.type,
		}

	def generate(
		self,
		message: str,
		temperature: float,
		top_k: int,
		top_p: float,
		max_new_tokens: int,
	) -> str:
		if self.model is None or self.tokenizer is None or self.device is None:
			raise RuntimeError("Model is not loaded")
		with self._generation_lock:
			return generate_response(
				self.model,
				self.tokenizer,
				message.strip(),
				max_new_tokens,
				temperature,
				top_k,
				top_p,
				self.device,
			)