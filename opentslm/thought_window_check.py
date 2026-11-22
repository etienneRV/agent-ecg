"""Which way of cutting the agent's thoughts keeps the most signal? Train split to fit, validation to score.

Each variant cuts every checkpoint's thoughts exactly as the OpenTSLM export would (the same tokenizer and,
for the head + tail variants, ``opentslm_data.Trimmer.thoughts`` itself). A word-count model (the
thoughts baseline: hashed word and word-pair counts, TF-IDF, logistic regression with a fixed C) is fitted
on a fixed sample of training runs and scored on every validation checkpoint. The test split is never read.

    uv run python opentslm/thought_window_check.py [--train-runs 8000]
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tokenizers import Tokenizer

from agent_ecg import data
from agent_ecg.baselines import _SEED, _weights
from agent_ecg.opentslm_data import PER_THOUGHT_TOKENS, TOKENIZER, Trimmer

RECENT, OLDER, CAP = 10, 32, 2048  # the tiered design first proposed for ecg-full v2


def tiered(trim: Trimmer, ids: list[list[int]], k: int) -> str:
    """The latest ``RECENT`` steps in full, older ones cut to ``OLDER`` tokens, newest first within ``CAP``."""
    lines, used = [], 0
    for step in range(k - 1, -1, -1):
        piece = trim.head(ids[step], PER_THOUGHT_TOKENS if step >= k - RECENT else OLDER)
        if used + piece.tokens > CAP and lines:
            lines.append(f"(Thoughts of steps 1-{step + 1} are left out to fit the length limit.)")
            break
        lines.append(f"Step {step + 1}: {piece.text}")
        used += piece.tokens
    return "\n".join(lines[::-1])


def variants(trim: Trimmer) -> dict[str, Callable[[list[str], list[list[int]], int], str]]:
    def head_tail(head: int, tail: int):
        return lambda raw, ids, k: trim.thoughts(ids, k, head, tail)[0]

    return {
        "latest 1,024 (ecg-thought, ecg-full)": head_tail(0, 1024),
        "latest 2,048": head_tail(0, 2048),
        "first 512 + latest 1,024": head_tail(512, 1024),
        "first 512 + latest 1,536 (ecg-full v2)": head_tail(512, 1536),
        "tiered: latest 10 full + older 32 tokens": lambda raw, ids, k: tiered(trim, ids, k),
        "all thoughts, uncut": lambda raw, ids, k: "\n".join(raw[:k]),
    }


def texts(table: pd.DataFrame, thoughts: dict[str, list[str]], trim: Trimmer, cut) -> list[str]:
    """Each checkpoint's thought text under one variant, run by run, tokenizing each run once."""
    out = [""] * len(table)
    for run_id, rows in table.groupby("run_id").indices.items():
        raw = thoughts[run_id]
        ids = trim.ids(raw)
        for i in rows:
            out[i] = cut(raw, ids, int(table["k"].iat[i]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--train-runs", type=int, default=8000, help="training runs sampled (all their checkpoints)")
    parser.add_argument("--out", type=Path, default=Path("results/thought_window_check.json"))
    args = parser.parse_args()
    started = time.perf_counter()

    table = data.load_checkpoints(args.data)
    table = table[table["split"] != "test"]  # never read
    train_runs = table.loc[table["split"] == "train", "run_id"].unique()
    keep = np.random.default_rng(_SEED).choice(train_runs, size=min(args.train_runs, len(train_runs)), replace=False)
    train = table[table["run_id"].isin(keep)].reset_index(drop=True)
    valid = table[table["split"] == "validation"].reset_index(drop=True)
    thoughts = data.load_thoughts(args.data)
    trim = Trimmer(Tokenizer.from_file(hf_hub_download(*TOKENIZER)))
    y_train, w_train = train["resolved"].to_numpy(bool), _weights(train)
    print(f"{len(train):,} train checkpoints ({len(keep):,} runs), {len(valid):,} validation checkpoints", flush=True)

    hasher = HashingVectorizer(n_features=2**20, ngram_range=(1, 2), alternate_sign=False, norm=None, dtype=np.float32)
    results = {}
    for name, cut in variants(trim).items():
        tfidf = TfidfTransformer(sublinear_tf=True)
        x_train = tfidf.fit_transform(hasher.transform(texts(train, thoughts, trim, cut)))
        x_valid = tfidf.transform(hasher.transform(texts(valid, thoughts, trim, cut)))
        model = LogisticRegression(C=0.3, max_iter=2000, solver="liblinear", random_state=_SEED)
        model.fit(sparse.csr_matrix(x_train), y_train, sample_weight=w_train)
        p = model.predict_proba(x_valid)[:, 1]
        per_k = {int(k): float(roc_auc_score(g["resolved"], p[g.index])) for k, g in valid.groupby("k")}
        results[name] = per_k
        print(
            f"{name:42} "
            + " / ".join(f"{per_k[k]:.3f}" for k in sorted(per_k))
            + f"   ({(time.perf_counter() - started) / 60:.1f} min)",
            flush=True,
        )
        del x_train, x_valid

    args.out.write_text(json.dumps({"train_runs": len(keep), "C": 0.3, "auroc_validation": results}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
