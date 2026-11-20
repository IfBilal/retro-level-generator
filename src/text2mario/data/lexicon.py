"""Loader for data/prompts/lexicon.yaml (doc 12 §6): noun variants with singular/plural forms and train/heldout halves.

Explain-me:
  What: turns the compact YAML entries ("goomba", "spiny|spinies", "gold|=") into (singular, plural) pairs.
  In/Out: yaml -> Lexicon object. `train_nouns(topic)` is what caption augmentation may use;
          `heldout_nouns(topic)` is reserved for the normalizer and robustness tests.
  Try: add a train synonym for "pipe" and watch augmented captions start to use it (drill 4).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
LEXICON_PATH = ROOT / "data" / "prompts" / "lexicon.yaml"


def parse_entry(entry: str) -> tuple[str, str]:
    """'goomba' -> ('goomba','goombas'); 'spiny|spinies' explicit; 'gold|=' mass; trailing y -> ies."""
    if "|" in entry:
        sg, pl = entry.split("|", 1)
        return sg, sg if pl == "=" else pl
    return entry, (entry[:-1] + "ies" if entry.endswith("y") and not entry.endswith(("ay", "ey", "oy")) else entry + "s")


@dataclass(frozen=True)
class Lexicon:
    raw: dict

    def train_nouns(self, topic: str) -> list[tuple[str, str]]:
        return [parse_entry(e) for e in self.raw["topics"][topic]["train"]]

    def heldout_nouns(self, topic: str) -> list[tuple[str, str]]:
        return [parse_entry(e) for e in self.raw["topics"][topic]["heldout"]]

    def floor(self, key: str, half: str = "train") -> list[tuple[str, str]]:
        node = self.raw["floor"][key]
        return [parse_entry(e) for e in (node[half] if isinstance(node, dict) else node)]

    def ceiling(self, key: str, half: str = "train") -> list[tuple[str, str]]:
        return [parse_entry(e) for e in self.raw["ceiling"][key][half]]

    def casual_quantities(self, bucket: str) -> list[str]:
        return [str(q) for q in self.raw["quantity_casual"][bucket]]

    def all_heldout_words(self) -> set[str]:
        """Every word that occurs only in held-out variants (used by the 'augmentation never leaks heldout' test)."""
        held, train = set(), set()
        for node in list(self.raw["topics"].values()) + list(self.raw["floor"].values()) + list(self.raw["ceiling"].values()):
            if isinstance(node, dict) and "train" in node:
                for e in node["train"]:
                    train.update(w for form in parse_entry(e) for w in form.split())
                for e in node["heldout"]:
                    held.update(w for form in parse_entry(e) for w in form.split())
        return held - train


def load(path: Path = LEXICON_PATH) -> Lexicon:
    return Lexicon(yaml.safe_load(path.read_text(encoding="utf-8")))
