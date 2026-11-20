"""Model loading and input building shared by train.py and predict.py.

OpenTSLM-SP's own ``pad_and_apply_batch`` embeds the prompt: text segments as token embeddings, each
time series as projected patch embeddings. Two of its batch habits are avoided here:

- it pads the series of a batch to the longest one and keeps the padding patches as tokens, so a
  5-step checkpoint batched with a 30-step one would see 22 extra zero patches; and
- ``compute_loss`` appends the answer after the right-padded prompt, leaving a gap of padding
  between prompt and answer.

So each sample's prompt is embedded on its own, then the batch is left-padded with explicit
position ids, the same way for training, likelihood scoring and generation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP

from agent_ecg.opentslm_samples import answer

LORA = {"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.0}  # OpenTSLM's curriculum settings


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(llm_id: str, device: str, *, init_from: str | None = None, lora: bool = True) -> OpenTSLMSP:
    """A fresh OpenTSLM-SP on ``llm_id``, or a pretrained one from the Hub (``init_from``), with LoRA."""
    if init_from:
        model = OpenTSLM.load_pretrained(init_from, device=device, enable_lora=lora)
        if not isinstance(model, OpenTSLMSP):
            raise ValueError(f"{init_from} is not an OpenTSLM-SP model (its id should end in -sp)")
    else:
        model = OpenTSLMSP(llm_id=llm_id, device=device)
        if lora:
            model.enable_lora(**LORA)
    return model


def model_sample(row: dict, task: str, post_prompts: dict[str, str]) -> dict:
    """One checkpoint row as the dict OpenTSLM-SP embeds."""
    return {
        "pre_prompt": row["pre_prompt"],
        "time_series_text": list(row["time_series_text"]),
        "time_series": torch.tensor([list(v) for v in row["time_series"]], dtype=torch.float32),
        "post_prompt": post_prompts[task],
    }


def embed_prompts(model: OpenTSLMSP, samples: list[dict]) -> list[torch.Tensor]:
    """Each sample's prompt embeddings, [length, hidden], embedded one at a time (no padding)."""
    return [model.pad_and_apply_batch([sample])[0][0] for sample in samples]


def answer_ids(model: OpenTSLMSP, texts: list[str]) -> list[list[int]]:
    """Token ids of each answer followed by the end-of-sequence token, without a leading BOS."""
    eos = model.get_eos_token()
    return [model.tokenizer(text + eos, add_special_tokens=False).input_ids for text in texts]


def assemble(model: OpenTSLMSP, prompts: list[torch.Tensor], answers: list[list[int]] | None = None):
    """Left-pad prompt (+ answer) embeddings into one batch.

    Returns:
        inputs_embeds [B, L, H], attention_mask [B, L], position_ids [B, L], and labels [B, L] with -100
        everywhere but the answer tokens (``None`` without answers).
    """
    embedding = model.llm.get_input_embeddings()
    device = prompts[0].device
    pieces = []
    for i, prompt in enumerate(prompts):
        if answers is None:
            pieces.append(prompt)
        else:
            ids = torch.tensor(answers[i], device=device)
            pieces.append(torch.cat([prompt, embedding(ids).to(prompt.dtype)], dim=0))
    length = max(piece.size(0) for piece in pieces)
    batch, hidden = len(pieces), pieces[0].size(1)
    inputs = torch.zeros(batch, length, hidden, dtype=pieces[0].dtype, device=device)
    mask = torch.zeros(batch, length, dtype=torch.long, device=device)
    labels = torch.full((batch, length), -100, dtype=torch.long, device=device) if answers is not None else None
    for i, piece in enumerate(pieces):
        inputs[i, length - piece.size(0) :] = piece
        mask[i, length - piece.size(0) :] = 1
        if labels is not None:
            labels[i, length - len(answers[i]) :] = torch.tensor(answers[i], device=device)
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    return inputs, mask, positions, labels


def loss(model: OpenTSLMSP, rows: list[dict], tasks: list[str], post_prompts: dict[str, str]) -> torch.Tensor:
    """Mean cross-entropy over the answer tokens of a batch of (checkpoint, task) items."""
    prompts = embed_prompts(model, [model_sample(r, t, post_prompts) for r, t in zip(rows, tasks, strict=True)])
    answers = answer_ids(model, [answer(r, t) for r, t in zip(rows, tasks, strict=True)])
    inputs, mask, positions, labels = assemble(model, prompts, answers)
    return model.llm(inputs_embeds=inputs, attention_mask=mask, position_ids=positions, labels=labels).loss


@torch.no_grad()
def candidate_logprobs(model: OpenTSLMSP, prompt: torch.Tensor, candidates: list[str]) -> torch.Tensor:
    """Summed log-probability of each candidate answer (plus end of sequence) after one prompt."""
    answers = answer_ids(model, candidates)
    inputs, mask, positions, labels = assemble(model, [prompt] * len(candidates), answers)
    logits = model.llm(inputs_embeds=inputs, attention_mask=mask, position_ids=positions).logits.float()
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    targets = labels[:, 1:]
    chosen = logp.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (chosen * (targets != -100)).sum(-1)


@torch.no_grad()
def generate(model: OpenTSLMSP, prompts: list[torch.Tensor], max_new_tokens: int) -> list[str]:
    """Greedy continuation of each prompt."""
    inputs, mask, _, _ = assemble(model, prompts)
    ids = model.llm.generate(
        inputs_embeds=inputs,
        attention_mask=mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=model.tokenizer.pad_token_id,
    )
    return [text.strip() for text in model.tokenizer.batch_decode(ids, skip_special_tokens=True)]
