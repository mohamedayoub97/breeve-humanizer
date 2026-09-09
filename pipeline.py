"""
pipeline.py
-----------
Wires the two pipeline stages together:
  1. Cleaning (OpenRouterClient.clean)  -> {message, metadata}
  2. Humanizer (pure Python, no LLM)    -> humanized_message

The Humanizer only ever touches `message`, never `metadata`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from openrouter_client import OpenRouterClient, CleaningResult
from humanizer import Humanizer, HumanizerConfig, HumanizedResult


@dataclass
class PipelineResult:
    input_raw: str
    message: str
    metadata: str
    humanized_message: str
    applied_rules: list
    latency_ms: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    error: Optional[str]

    def to_dict(self):
        return asdict(self)


class Pipeline:
    def __init__(self, client: OpenRouterClient, humanizer: Optional[Humanizer] = None):
        self.client = client
        self.humanizer = humanizer or Humanizer(HumanizerConfig())

    def run(self, raw_input: str) -> PipelineResult:
        cleaning: CleaningResult = self.client.clean(raw_input)

        if cleaning.error:
            return PipelineResult(
                input_raw=raw_input,
                message=cleaning.message,
                metadata=cleaning.metadata,
                humanized_message="",
                applied_rules=[],
                latency_ms=cleaning.latency_ms,
                prompt_tokens=cleaning.prompt_tokens,
                completion_tokens=cleaning.completion_tokens,
                total_tokens=cleaning.total_tokens,
                error=cleaning.error,
            )

        result: HumanizedResult = self.humanizer.apply(cleaning.message)

        return PipelineResult(
            input_raw=raw_input,
            message=cleaning.message,
            metadata=cleaning.metadata,
            humanized_message=result.text,
            applied_rules=result.applied_rules,
            latency_ms=cleaning.latency_ms,
            prompt_tokens=cleaning.prompt_tokens,
            completion_tokens=cleaning.completion_tokens,
            total_tokens=cleaning.total_tokens,
            error=None,
        )
