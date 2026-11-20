"""Measure WordPiece lengths of captions to confirm max_len = 48 (doc 11 §6, ledger M7).

Explain-me:
  What: tokenises all 7,687 stored captions and 10,000 augmented captions (incl. casual paraphrases) with the real
        MiniLM tokenizer, reports min/mean/p99/max and which words split, and asserts max <= text.max_len.
  Out: reports/eda/tokens.json.  If the assertion fails: set text.max_len = 64 in configs/base.yaml and log it.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.data import lexicon
from text2mario.data.augment import AugConfig, augment_caption
from text2mario.utils.config import load_config


def stats(tok, texts):
    lens = np.array([len(x) for x in tok(texts, add_special_tokens=True, truncation=False)["input_ids"]])
    return {"n": len(texts), "min": int(lens.min()), "mean": round(float(lens.mean()), 2), "p99": int(np.percentile(lens, 99)), "max": int(lens.max())}


def main():
    cfg = load_config()
    tok = AutoTokenizer.from_pretrained(ROOT / "models" / "minilm", local_files_only=True)
    caps = pd.read_parquet(ROOT / "data/processed/index.parquet").caption.tolist()
    lex = lexicon.load()
    rng = np.random.default_rng(0)
    aug = [augment_caption(caps[int(i)], 20261117, s, 0, AugConfig(), lex)[0] for s, i in enumerate(rng.integers(0, len(caps), 10000))]
    out = {"stored": stats(tok, caps), "augmented": stats(tok, [a for a in aug if a]), "null_tokens": len(tok([""])["input_ids"][0])}
    split_words = Counter()
    for w in {w for c in caps for w in c.replace(".", " ").split()}:
        n = len(tok.tokenize(w))
        if n > 1:
            split_words[w] = n
    out["multi_piece_words"] = dict(split_words)
    out["max_len"] = cfg["data"]["max_len"]
    out["pass"] = max(out["stored"]["max"], out["augmented"]["max"]) <= out["max_len"] and out["null_tokens"] == 2
    (ROOT / "reports/eda/tokens.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
