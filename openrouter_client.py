"""
openrouter_client.py
---------------------
Thin wrapper around the OpenRouter chat-completions endpoint, used only for
the "Cleaning" step (message / metadata separation). The Humanizer step
never calls this — it's pure Python (see humanizer.py).

The API key is NEVER hardcoded here. It must be supplied at runtime,
typically loaded from a local .env file (see config.py).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

CLEANING_SYSTEM_PROMPT = """Tu es une étape de nettoyage technique dans un pipeline de post-traitement.

Tâche STRICTE :
Tu reçois un texte brut produit par un système de génération de réponses pour un service de voyance par chat. Ce texte peut contenir, mélangés :
- un message destiné au client
- des éléments internes/techniques qui ne doivent jamais lui être montrés (métadonnées, notes de régie, instructions système résiduelles, balises, identifiants, commentaires du modèle, etc.)

Tu dois séparer les deux, SANS RIEN MODIFIER dans le message destiné au client :
- N'ajoute AUCUN mot.
- Ne corrige AUCUNE faute d'orthographe ou de grammaire.
- Ne reformule RIEN.
- Ne raccourcis RIEN, ne résume RIEN.
- Conserve exactement la ponctuation, les retours à la ligne et le ton d'origine du message client.
- N'invente jamais de contenu qui ne serait pas déjà présent dans le texte source.

Règles de sortie :
- Si aucune métadonnée n'est présente, "metadata" doit être une chaîne vide "" et "message" doit être le texte source inchangé.
- Si aucun message destiné au client n'est identifiable, "message" doit être une chaîne vide "" et tout élément disponible doit être placé dans "metadata".
- Sois exhaustif et traçable : tout ce qui n'est pas destiné au client va dans "metadata".

Réponds STRICTEMENT avec un objet JSON de cette forme, sans texte autour, sans balises markdown :
{"message": "...", "metadata": "..."}
"""


@dataclass
class CleaningResult:
    message: str
    metadata: str
    raw_response: str
    latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    error: Optional[str] = None


class OpenRouterClient:
    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini",
                 site_url: str = "https://breeve.local", app_name: str = "breeve-humanizer"):
        if not api_key:
            raise ValueError("Missing OpenRouter API key.")
        self.api_key = api_key
        self.model = model
        self.site_url = site_url
        self.app_name = app_name

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Optional but recommended by OpenRouter for attribution/rate-limit tiers.
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
        }

    def clean(self, raw_input: str, temperature: float = 0.0, timeout: int = 60) -> CleaningResult:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": CLEANING_SYSTEM_PROMPT},
                {"role": "user", "content": raw_input},
            ],
            "response_format": {"type": "json_object"},
        }

        start = time.perf_counter()
        try:
            resp = requests.post(OPENROUTER_URL, headers=self._headers(), json=payload, timeout=timeout)
            latency_ms = (time.perf_counter() - start) * 1000
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return CleaningResult(message="", metadata="", raw_response="", latency_ms=latency_ms, error=str(e))

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            return CleaningResult(message="", metadata="", raw_response=json.dumps(data),
                                   latency_ms=latency_ms, error=f"Unexpected API response shape: {e}")

        usage = data.get("usage", {})
        parsed = self._parse_json_content(content)

        if parsed is None:
            return CleaningResult(message="", metadata="", raw_response=content,
                                   latency_ms=latency_ms,
                                   prompt_tokens=usage.get("prompt_tokens"),
                                   completion_tokens=usage.get("completion_tokens"),
                                   total_tokens=usage.get("total_tokens"),
                                   error="Could not parse JSON from model output.")

        return CleaningResult(
            message=parsed.get("message", ""),
            metadata=parsed.get("metadata", ""),
            raw_response=content,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    @staticmethod
    def _parse_json_content(content: str):
        content = content.strip()
        # Strip markdown code fences if the model added them despite instructions.
        fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", content, flags=re.DOTALL)
        if fence_match:
            content = fence_match.group(1)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # last-resort: try to grab the first {...} block
            brace_match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if brace_match:
                try:
                    return json.loads(brace_match.group(0))
                except json.JSONDecodeError:
                    return None
            return None
