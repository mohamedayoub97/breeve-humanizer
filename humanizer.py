"""
humanizer.py
------------
Custom "Humanizer" component for the Breeve post-processing pipeline.

Design goals (see README.md for the full rationale):
- Pure Python, deterministic-if-seeded, NO second LLM call.
- Operates ONLY on the `message` field, never on `metadata`.
- Implements exactly the 5 rules from regles_humanizer.csv (R-01..R-05).
- Frequency / probability / ordering of each rule is configurable, per the
  brief ("Vous êtes libre de définir la fréquence, les probabilités et
  l'ordre d'application des règles").
- Hard constraint respected unconditionally: R-03 and R-04 never both fire
  on the same message.
- Each application is capped at "at most one occurrence" per the rule text
  (e.g. R-01 replaces at most one word, R-05 removes at most one sign).
- Returns a log of which rules actually fired, for traceability /
  measurability (the brief asks for a "pipeline mesurable et compréhensible").
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_ABBREVIATIONS = {
    "quelque chose": "qqch",   # multi-word phrase, must be checked before single words
    "maintenant": "mtn",
    "beaucoup": "bcp",
    "pour": "pr",
    "vous": "vs",
}

# French accented -> unaccented letter map (upper + lower)
ACCENT_MAP = {
    "à": "a", "â": "a", "ä": "a",
    "é": "e", "è": "e", "ê": "e", "ë": "e",
    "î": "i", "ï": "i",
    "ô": "o", "ö": "o",
    "ù": "u", "û": "u", "ü": "u",
    "ç": "c",
    "ÿ": "y",
}
ACCENT_MAP.update({k.upper(): v.upper() for k, v in ACCENT_MAP.items()})

# R-05 is only ever allowed to remove one of these signs.
REMOVABLE_PUNCTUATION = [",", ";", ":", "!"]

# Hard guard: these characters must NEVER be removed by R-05, no matter what.
# Kept as an explicit set (rather than relying on '?' simply being absent
# from REMOVABLE_PUNCTUATION above) so the guarantee survives future edits
# to that list and is enforced defensively at the point of removal.
NEVER_REMOVE_PUNCTUATION = {"?", "？"}  # '?' (ASCII) and '？' (full-width)


@dataclass
class HumanizerConfig:
    """All knobs are intentionally exposed here so the component stays
    configurable from the GUI without touching the rule logic."""

    # Probability [0-1] that a rule is *attempted* on a given message,
    # provided a valid candidate exists in the text. If no candidate
    # exists (e.g. no word >= 5 letters for R-03), the rule silently
    # does not fire regardless of probability.
    prob_r01_abbreviation: float = 0.5
    prob_r02_accent: float = 0.5
    prob_r05_punctuation: float = 0.5

    # R-03 / R-04 are mutually exclusive on a given message. We first
    # decide (weighted coin) which of the two we are even allowed to
    # attempt this round, then apply that single rule's own probability.
    prob_r03_letter_omission: float = 0.5
    prob_r04_word_fusion: float = 0.5
    # Relative weight used to choose between R-03 and R-04 when both are
    # enabled (weight_r03 : weight_r04). Does not need to sum to 1.
    weight_r03: float = 0.5
    weight_r04: float = 0.5

    # Order in which independent rule "slots" are attempted. The R-03/R-04
    # pair is always treated as a single slot internally.
    # Allowed tokens: "R-01", "R-02", "R-03/R-04", "R-05"
    order: tuple = ("R-01", "R-02", "R-03/R-04", "R-05")

    # Master toggles, useful for isolating a rule during testing.
    enable_r01: bool = True
    enable_r02: bool = True
    enable_r03: bool = True
    enable_r04: bool = True
    enable_r05: bool = True

    abbreviations: dict = field(default_factory=lambda: dict(DEFAULT_ABBREVIATIONS))

    # If set, every call to apply() with the same message text is
    # reproducible. If None, a fresh random seed derived from the message
    # is used each time (still deterministic per-message, but two
    # different messages won't correlate).
    rng_seed: Optional[int] = None


@dataclass
class HumanizedResult:
    text: str
    applied_rules: list  # e.g. ["R-01", "R-05"]
    seed_used: int


# ---------------------------------------------------------------------------
# Humanizer
# ---------------------------------------------------------------------------

class Humanizer:
    """Stateless-per-call component. Instantiate once, call .apply(message)
    for every message. Thread-safe as long as each call uses its own RNG
    (it does — see _make_rng)."""

    def __init__(self, config: Optional[HumanizerConfig] = None):
        self.config = config or HumanizerConfig()

    # -- public API ---------------------------------------------------

    def apply(self, message: str) -> HumanizedResult:
        if not message:
            return HumanizedResult(text=message, applied_rules=[], seed_used=0)

        rng, seed_used = self._make_rng(message)
        text = message
        applied: list[str] = []

        for slot in self.config.order:
            if slot == "R-01":
                text, ok = self._rule_r01(text, rng)
            elif slot == "R-02":
                text, ok = self._rule_r02(text, rng)
            elif slot == "R-03/R-04":
                text, ok = self._rule_r03_or_r04(text, rng)
                if ok:
                    applied.append(ok)  # ok holds "R-03" or "R-04" string
                    continue
                ok = None
            elif slot == "R-05":
                text, ok = self._rule_r05(text, rng)
            else:
                continue
            if ok:
                applied.append(slot)

        return HumanizedResult(text=text, applied_rules=applied, seed_used=seed_used)

    # -- RNG ------------------------------------------------------------

    def _make_rng(self, message: str) -> tuple[random.Random, int]:
        if self.config.rng_seed is not None:
            seed = self.config.rng_seed
        else:
            seed = abs(hash(message)) % (2**32)
        return random.Random(seed), seed

    # -- R-01: abbreviation ---------------------------------------------

    def _rule_r01(self, text: str, rng: random.Random):
        if not self.config.enable_r01 or rng.random() > self.config.prob_r01_abbreviation:
            return text, False

        # Sort keys longest-first so multi-word phrases are matched before
        # their sub-words (e.g. "quelque chose" before a lone "chose").
        keys = sorted(self.config.abbreviations.keys(), key=len, reverse=True)
        candidates = []
        for key in keys:
            for m in re.finditer(r"(?<!\w)" + re.escape(key) + r"(?!\w)", text, flags=re.IGNORECASE):
                candidates.append((m.start(), m.end(), key))

        if not candidates:
            return text, False

        start, end, key = rng.choice(candidates)
        replacement = self.config.abbreviations[key]
        # Preserve capitalisation of the first letter if the source was capitalised.
        if text[start:end][0].isupper():
            replacement = replacement[0].upper() + replacement[1:]
        new_text = text[:start] + replacement + text[end:]
        return new_text, True

    # -- R-02: remove one accent -----------------------------------------

    def _rule_r02(self, text: str, rng: random.Random):
        if not self.config.enable_r02 or rng.random() > self.config.prob_r02_accent:
            return text, False

        positions = [i for i, ch in enumerate(text) if ch in ACCENT_MAP]
        if not positions:
            return text, False

        idx = rng.choice(positions)
        new_text = text[:idx] + ACCENT_MAP[text[idx]] + text[idx + 1:]
        return new_text, True

    # -- R-03 / R-04 (mutually exclusive) --------------------------------

    def _rule_r03_or_r04(self, text: str, rng: random.Random):
        r03_available = self.config.enable_r03 and self._r03_candidates(text)
        r04_available = self.config.enable_r04 and self._r04_candidates(text)

        if not r03_available and not r04_available:
            return text, None

        if r03_available and r04_available:
            w03, w04 = self.config.weight_r03, self.config.weight_r04
            choice = rng.choices(["R-03", "R-04"], weights=[w03, w04], k=1)[0]
        elif r03_available:
            choice = "R-03"
        else:
            choice = "R-04"

        if choice == "R-03":
            if rng.random() > self.config.prob_r03_letter_omission:
                return text, None
            new_text = self._apply_r03(text, rng)
            return new_text, "R-03"
        else:
            if rng.random() > self.config.prob_r04_word_fusion:
                return text, None
            new_text = self._apply_r04(text, rng)
            return new_text, "R-04"

    @staticmethod
    def _r03_candidates(text: str):
        return [m for m in re.finditer(r"\b[^\W\d_]{5,}\b", text, flags=re.UNICODE)]

    def _apply_r03(self, text: str, rng: random.Random) -> str:
        candidates = self._r03_candidates(text)
        m = rng.choice(candidates)
        word = m.group(0)
        # remove an interior letter (not first, not last)
        pos_in_word = rng.randint(1, len(word) - 2)
        new_word = word[:pos_in_word] + word[pos_in_word + 1:]
        start, end = m.span()
        return text[:start] + new_word + text[end:]

    @staticmethod
    def _r04_candidates(text: str):
        # a single space between two "word" tokens (letters/digits)
        return [m for m in re.finditer(r"(?<=[^\W_])( )(?=[^\W_])", text, flags=re.UNICODE)]

    def _apply_r04(self, text: str, rng: random.Random) -> str:
        candidates = self._r04_candidates(text)
        m = rng.choice(candidates)
        start, end = m.span()
        return text[:start] + text[end:]

    # -- R-05: punctuation -------------------------------------------------

    def _rule_r05(self, text: str, rng: random.Random):
        if not self.config.enable_r05 or rng.random() > self.config.prob_r05_punctuation:
            return text, False

        positions = [
            i for i, ch in enumerate(text)
            if ch in REMOVABLE_PUNCTUATION and ch not in NEVER_REMOVE_PUNCTUATION
        ]
        if not positions:
            return text, False

        idx = rng.choice(positions)
        removed_char = text[idx]
        assert removed_char not in NEVER_REMOVE_PUNCTUATION, (
            f"Punctuation guard violated: attempted to remove {removed_char!r}"
        )
        new_text = text[:idx] + text[idx + 1:]
        return new_text, True


# ---------------------------------------------------------------------------
# Quick self-test when run directly (no display / no network needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "Bonjour, je vois beaucoup d'énergie autour de vous en ce moment ! Voulez-vous que je continue ?",
        "Maintenant je perçois quelque chose d'important pour votre avenir professionnel.",
        "Merci pour votre question ; la réponse est claire, précise et rassurante.",
    ]

    cfg = HumanizerConfig(rng_seed=42)
    h = Humanizer(cfg)
    for s in samples:
        res = h.apply(s)
        print("SRC:", s)
        print("OUT:", res.text)
        print("RULES:", res.applied_rules, "SEED:", res.seed_used)
        print("-" * 60)
