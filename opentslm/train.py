"""Fine-tune OpenTSLM-SP on the agent-ecg checkpoints.

    python opentslm/train.py --smoke                      # 20 steps: does everything run? prints s/step and memory
    python opentslm/train.py --name llama1b-multi         # all three tasks, defaults below
    python opentslm/train.py --name llama1b-outcome --tasks outcome   # pass/fail only
    python opentslm/train.py --name gemma-v0 --llm google/gemma-3-270m

Trains the time-series encoder, the projector and LoRA adapters on the language model (OpenTSLM's
own split of trainable parts), on all three tasks at once. Each epoch draws ``--samples-per-epoch``
items with replacement: each checkpoint weighted by 1 / (checkpoints of its run), and resolved runs
reweighted to ``--resolved-share`` of the weight.

The loss averages each answer's tokens first, then the answers (weighted by ``--task-weights``), so a
3-token "unresolved" counts as much as a 110-token status report. Every ``--eval-every`` steps the
script measures the validation loss and the pass/fail AUROC (mean over the 5/10/20/30-step
checkpoints, from the likelihood of "resolved" vs "unresolved"); the best AUROC is saved to
``runs/<name>/best_model.pt`` with ``config.json`` (``--select loss`` uses the loss instead).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_model, loss, p_resolved, pick_device

from agent_ecg import opentslm_samples as samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/opentslm"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--name", default="opentslm-sp")
    parser.add_argument("--llm", default="meta-llama/Llama-3.2-1B", help="base language model (Hugging Face id)")
    parser.add_argument(
        "--init-from", help="start from a pretrained OpenTSLM-SP on the Hub, e.g. OpenTSLM/llama-3.2-1b-tsqa-sp"
    )
    parser.add_argument("--no-lora", action="store_true", help="train only the encoder and projector")
    parser.add_argument("--tasks", default="outcome,ending,status", help="e.g. `outcome` for a pass/fail-only model")
    parser.add_argument("--task-weights", default="", help="e.g. `outcome=2,status=0.5`; unnamed tasks weigh 1")
    parser.add_argument(
        "--select", default="auroc", choices=("auroc", "loss"), help="how the best checkpoint is chosen"
    )
    parser.add_argument("--one-per-run", action="store_true", help="train on one random checkpoint per run")
    parser.add_argument("--max-train", type=int, help="cap on training checkpoints")
    parser.add_argument("--val-checkpoints", type=int, default=2000, help="validation checkpoints for the AUROC")
    parser.add_argument("--val-loss-checkpoints", type=int, default=300, help="of those, used for the loss")
    parser.add_argument(
        "--resolved-share", type=float, default=0.3, help="share of sampling weight on resolved runs; 0 = natural"
    )
    parser.add_argument("--samples-per-epoch", type=int, default=60_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr-encoder", type=float, default=2e-4)
    parser.add_argument("--lr-projector", type=float, default=1e-4)
    parser.add_argument("--lr-lora", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup", type=float, default=0.03)
    parser.add_argument("--eval-every", type=int, default=500, help="optimizer steps between validation runs")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="20 optimizer steps on a small sample, then stop")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.name, args.max_train, args.val_checkpoints, args.val_loss_checkpoints = f"{args.name}-smoke", 400, 32, 16
        args.samples_per_epoch, args.epochs, args.eval_every = 20 * args.batch_size * args.grad_accum, 1, 10
    torch.manual_seed(args.seed)
    run = args.runs / args.name
    run.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    tasks = [t for t in args.tasks.split(",") if t]
    prompts = samples.post_prompts(args.data)

    train = samples.load(args.data, "train", one_per_run=args.one_per_run, cap=args.max_train, seed=args.seed)
    valid = samples.load(args.data, "validation", cap=args.val_checkpoints, seed=args.seed)
    train_items, valid_items = samples.items(train, tasks), samples.items(valid, tasks)
    w = samples.weights(train, train_items, args.resolved_share or None)
    train_rows, valid_rows = train.to_dict("records"), valid.to_dict("records")
    print(
        f"device {device}; {len(train)} train checkpoints ({len(train_items)} items), {len(valid_items)} validation items"
    )

    model = load_model(args.llm, device, init_from=args.init_from, lora=not args.no_lora)
    if args.gradient_checkpointing:
        model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.llm.config.use_cache = False
    groups = [
        {"params": list(model.encoder.parameters()), "lr": args.lr_encoder},
        {"params": list(model.projector.parameters()), "lr": args.lr_projector},
    ]
    if not args.no_lora:
        groups.append({"params": model.get_lora_parameters(), "lr": args.lr_lora})
    trainable = sum(p.numel() for g in groups for p in g["params"])
    print(f"trainable parameters: {trainable / 1e6:.1f} M")
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(args.samples_per_epoch / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * total_steps), total_steps)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(torch.as_tensor(w), args.samples_per_epoch, replacement=True, generator=generator)
    loader = DataLoader(range(len(train_items)), batch_size=args.batch_size, sampler=sampler, collate_fn=list)

    config = vars(args) | {"device": device, "trainable_parameters": trainable, "total_steps": total_steps}
    (run / "config.json").write_text(json.dumps(config, indent=2, default=str))

    task_weights = samples.parse_weights(args.task_weights)
    loss_items = [item for item in valid_items if item[0] < args.val_loss_checkpoints]
    valid_resolved, valid_k = valid["resolved"].to_numpy(bool), valid["k"].to_numpy()

    def validate() -> tuple[float, float]:
        model.eval()
        losses = []
        with torch.no_grad():
            for start in range(0, len(loss_items), args.batch_size):
                chunk = loss_items[start : start + args.batch_size]
                rows, chunk_tasks = [valid_rows[r] for r, _ in chunk], [t for _, t in chunk]
                losses.append(loss(model, rows, chunk_tasks, prompts, task_weights).item())
            auroc = samples.mean_auroc(valid_resolved, valid_k, np.array(p_resolved(model, valid_rows, prompts)))
        model.train()
        return float(np.mean(losses)), auroc

    best, step, history, started = -math.inf, 0, [], time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        for micro, batch in enumerate(loader):
            chosen = [train_items[i] for i in batch]
            rows, chosen_tasks = [train_rows[r] for r, _ in chosen], [t for _, t in chosen]
            value = loss(model, rows, chosen_tasks, prompts, task_weights)
            (value / args.grad_accum).backward()
            if (micro + 1) % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            elapsed = time.perf_counter() - started
            if step % 10 == 0 or args.smoke:
                memory = (
                    f", peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if device == "cuda" else ""
                )
                print(
                    f"epoch {epoch} step {step}/{total_steps} loss {value.item():.4f} ({elapsed / step:.2f} s/step{memory})",
                    flush=True,
                )
            if step % args.eval_every == 0 or step == total_steps:
                val_loss, val_auroc = validate()
                score = val_auroc if args.select == "auroc" else -val_loss
                history.append(
                    {
                        "step": step,
                        "train_loss": value.item(),
                        "val_loss": val_loss,
                        "val_auroc": val_auroc,
                        "elapsed_s": elapsed,
                    }
                )
                (run / "history.json").write_text(json.dumps(history, indent=2))
                # AUROC is undefined when a small validation sample has no resolved run (e.g. --smoke):
                # keep the first checkpoint so prediction still has one, and wait for a defined score.
                defined = not math.isnan(score)
                saved = score > best if defined else not (run / "best_model.pt").exists()
                print(
                    f"  validation loss {val_loss:.4f}, AUROC {val_auroc:.3f}" + ("  (best, saved)" if saved else ""),
                    flush=True,
                )
                if saved:
                    best = score if defined else best
                    model.store_to_file(str(run / "best_model.pt"))
    hours = (time.perf_counter() - started) / 3600
    print(f"done: best validation {args.select} {abs(best):.4f}, {hours:.2f} h; checkpoint {run / 'best_model.pt'}")
    if args.smoke:
        full = math.ceil(60_000 / (args.batch_size * args.grad_accum)) * 2
        print(f"estimate for the default full run ({full} steps): {full * hours / step:.1f} h")


if __name__ == "__main__":
    main()
