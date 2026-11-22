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

from agent_ecg.opentslm_samples import OUTCOMES, answer

LORA = {"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.0}  # OpenTSLM's curriculum settings
ATTENTION = "sdpa"  # replaces OpenTSLM's "eager"; same math, far less memory on long prompts


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    llm_id: str, device: str, *, init_from: str | None = None, lora: bool = True, lora_dropout: float = 0.0
) -> OpenTSLMSP:
    """An OpenTSLM-SP on ``llm_id`` with LoRA, fresh or starting from a pretrained one on the Hub.

    ``init_from`` (e.g. ``OpenTSLM/llama-3.2-1b-ecg-sp``) loads that checkpoint's encoder, projector
    and LoRA weights, as ``OpenTSLM.load_pretrained`` does, but onto ``llm_id``. OpenTSLM's loader always
    uses the gated ``meta-llama/...`` id; ``unsloth/Llama-3.2-1B`` holds the same weights without the
    gate, so ``--llm unsloth/Llama-3.2-1B --init-from OpenTSLM/llama-3.2-1b-ecg-sp`` needs no approval.

    ``lora_dropout`` (default 0, OpenTSLM's setting) drops LoRA inputs during training only, against
    memorising the training text (ecg-full v2); prediction runs in eval mode, where it has no effect.

    Raises:
        ValueError: If ``init_from`` is not an SP checkpoint or was trained on another base model.
    """
    if init_from:
        if not init_from.endswith("-sp"):
            raise ValueError(f"{init_from} is not an OpenTSLM-SP model (its id should end in -sp)")
        base = OpenTSLM._get_base_llm_id(init_from)
        if base.split("/")[-1].lower() != llm_id.split("/")[-1].lower():
            raise ValueError(f"{init_from} was trained on {base}; pass --llm {base} or an ungated copy of it")
    model = OpenTSLMSP(llm_id=llm_id, device=device)
    # OpenTSLM loads Llama with eager attention, which keeps a [batch, heads, length, length] matrix per
    # layer for the backward pass: the 1,400-2,300-token prompts with issue and thoughts run out of
    # memory on a 46 GB GPU. PyTorch's SDPA computes the same attention without that matrix.
    model.llm.set_attn_implementation(ATTENTION)
    if lora:
        model.enable_lora(**(LORA | {"lora_dropout": lora_dropout}))
    if init_from:
        from huggingface_hub import hf_hub_download

        model.load_from_file(hf_hub_download(init_from, "model_checkpoint.pt"))
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


def _answer_logprobs(
    model: OpenTSLMSP, prompts: list[torch.Tensor], answers: list[list[int]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probability of each answer token, and which positions are answer tokens.

    Left padding puts every answer at the end of its row, so only the last (longest answer + 1)
    positions need logits; with Llama's 128k vocabulary that saves most of the memory.

    Returns:
        token log-probabilities [B, A] (0 off the answers) and the answer mask [B, A].
    """
    inputs, mask, positions, labels = assemble(model, prompts, answers)
    keep = max(len(ids) for ids in answers) + 1
    logits = model.llm(inputs_embeds=inputs, attention_mask=mask, position_ids=positions, logits_to_keep=keep).logits
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = labels[:, -keep:][:, 1:]
    valid = targets != -100
    chosen = logp.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return chosen * valid, valid


def loss(
    model: OpenTSLMSP,
    rows: list[dict],
    tasks: list[str],
    post_prompts: dict[str, str],
    task_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Weighted mean over items of each item's mean token cross-entropy.

    Averaging per item first makes a 3-token "unresolved" count as much as a 110-token status report;
    a plain token average (Hugging Face's default) would give the status reports ~97 % of the signal.
    """
    prompts = embed_prompts(model, [model_sample(r, t, post_prompts) for r, t in zip(rows, tasks, strict=True)])
    answers = answer_ids(model, [answer(r, t) for r, t in zip(rows, tasks, strict=True)])
    token_logp, valid = _answer_logprobs(model, prompts, answers)
    per_item = -token_logp.sum(-1) / valid.sum(-1)
    weights = torch.tensor([(task_weights or {}).get(t, 1.0) for t in tasks], device=per_item.device)
    return (per_item * weights).sum() / weights.sum()


@torch.no_grad()
def candidate_logprobs(model: OpenTSLMSP, prompt: torch.Tensor, candidates: list[str]) -> torch.Tensor:
    """Summed log-probability of each candidate answer (plus end of sequence) after one prompt."""
    token_logp, _ = _answer_logprobs(model, [prompt] * len(candidates), answer_ids(model, candidates))
    return token_logp.sum(-1)


@torch.no_grad()
def p_resolved(model: OpenTSLMSP, rows: list[dict], post_prompts: dict[str, str]) -> list[float]:
    """Probability of "resolved" over "unresolved" for each checkpoint, from the answers' likelihoods."""
    out = []
    for row in rows:
        (prompt,) = embed_prompts(model, [model_sample(row, "outcome", post_prompts)])
        out.append(torch.softmax(candidate_logprobs(model, prompt, list(OUTCOMES)), 0)[0].item())
    return out


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
