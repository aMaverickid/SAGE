"""Local HookedSAETransformer + SAE helpers ported from
``feature_descriptions_pipeline.ipynb``.

Two reasons we duplicate (rather than import) ``core.system``:
    1. The notebook treats the model + SAE as plain objects with documented
       methods (``run_with_saes``, ``run_with_hooks_with_saes``,
       ``sae.encode``/``decode``). We want eval_metrics callable with any
       HookedSAETransformer + SAELens SAE, not just a SAGE-loaded ``System``.
    2. ``System`` is built around per-feature configuration; here we need
       to evaluate many features and want to keep the model + SAE alive
       across the whole sweep.

Callers that already have a ``core.system.System`` instance can pass
``system.model`` and ``system.sae["__sae_lens_obj__"]`` directly.
"""
from __future__ import annotations

import functools
from typing import List, Tuple

GEN_PROMPTS_DEFAULT = ("The explanation is simple:", "I think", "We")
KL_DIV_VALUES_DEFAULT = (0.25, 0.5, -0.25, -0.5)


def _require_torch():
    """Lazy-import torch so this module is importable without torch installed."""
    import torch
    return torch


def _sae_hook_name(sae) -> str:
    """Return the SAE's residual hook name across sae-lens versions.

    Older sae-lens (≤5.x) exposed ``sae.cfg.hook_name`` directly. sae-lens
    6.x moved it under ``sae.cfg.metadata.hook_name`` (SAEMetadata). We
    try the legacy path first, then fall back to the metadata namespace.
    """
    direct = getattr(sae.cfg, "hook_name", None)
    if direct is not None:
        return str(direct)
    metadata = getattr(sae.cfg, "metadata", None)
    if metadata is not None:
        name = getattr(metadata, "hook_name", None)
        if name is not None:
            return str(name)
    raise AttributeError(
        "SAE config exposes no hook_name (checked sae.cfg.hook_name and "
        "sae.cfg.metadata.hook_name); is this an unsupported sae-lens version?"
    )


def _set_feature_act_hook(act, hook, feature: int, value: float):  # noqa: ARG001
    """Clamp the SAE feature dim to ``value`` in-place (KL-measurement hook).

    ``hook`` is the HookPoint TransformerLens passes by keyword — the
    parameter name MUST be ``hook`` (TL calls it as ``fn(act, hook=...)``);
    we just don't use it."""
    act[:, :, feature] = value


def _gen_hook(clean_act, hook, sae, feature: int, value: float):  # noqa: ARG001
    """Steering hook used during generation: re-encode through the SAE,
    clamp the chosen feature, then decode back into the residual stream
    while preserving the SAE reconstruction error."""
    if sae is None:
        clean_act[:, :, feature] = value
        return clean_act
    encoded = sae.encode(clean_act)
    dirty = sae.decode(encoded)
    error_term = clean_act - dirty
    encoded[:, :, feature] = value
    steered = sae.decode(encoded) + error_term
    return steered.to(dtype=clean_act.dtype)


def _kl_div(p, q, eps: float = 1e-10):
    torch = _require_torch()
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)


def get_kl_div(
    model, sae, prompts: List[str], layer: int, feature: int, value: float,
) -> float:
    """Mean KL divergence (per non-pad token) between clean and feature-clamped
    output probabilities at the SAE hook point.

    ``sae=None`` falls back to the MLP-post path the notebook used for
    Transluce-style features without an SAE."""
    toks = model.to_tokens(prompts)
    fwd_hooks = _kl_hooks(sae, layer, feature, value)
    if sae is None:
        clean_logits = model(toks)
        hooked_logits = model.run_with_hooks(toks, fwd_hooks=fwd_hooks)
    else:
        clean_logits = model.run_with_saes(toks, saes=[sae])
        hooked_logits = model.run_with_hooks_with_saes(toks, saes=[sae], fwd_hooks=fwd_hooks)
    clean_probs = clean_logits.float().softmax(dim=-1)
    hooked_probs = hooked_logits.float().softmax(dim=-1)
    clean_probs[toks == 0] = 0
    hooked_probs[toks == 0] = 0
    kl = _kl_div(clean_probs, hooked_probs)
    row_means: List[float] = []
    for row in kl:
        nz = row[row != 0]
        if nz.numel() > 0:
            row_means.append(nz.mean().item())
    return float(sum(row_means) / len(row_means)) if row_means else 0.0


def _kl_hooks(sae, layer: int, feature: int, value: float):
    """Build the forward-hook list for KL measurement (SAE vs raw MLP)."""
    if sae is None:
        hook = f"blocks.{layer}.mlp.hook_pre"
    else:
        hook = f"{_sae_hook_name(sae)}.hook_sae_acts_post"
    return [(hook, functools.partial(_set_feature_act_hook, feature=feature, value=value))]


def get_activation_for_kl(
    model, sae, prompts: List[str], layer: int, feature: int,
    target_kl: float, high_thresh: float = 0.1, neg: bool = False,
    verbose: bool = False,
) -> float:
    """Binary search for the activation magnitude that yields ``target_kl``.

    Mirrors the notebook: positive search range [1, 1000]; negative search
    range [-1000, -1]. ``high_thresh`` is the slack above ``target_kl`` we
    accept before terminating."""
    low, high = (-1000, -1) if neg else (1, 1000)
    kl = -1.0
    mid = 0
    while (low + 1 < high) and (kl < target_kl or kl > target_kl + high_thresh):
        mid = (low + high) // 2
        kl = get_kl_div(model, sae, prompts, layer, feature, mid)
        if (neg and kl < target_kl) or (not neg and kl > target_kl):
            high = mid
        else:
            low = mid
        if verbose:
            print(f"  low={low} high={high} mid={mid} kl={kl:.3f} target={target_kl}")
    return float(mid)


def hooked_gen(
    prompt, model, sae, layer: int, feature: int,
    value: float, n_new: int = 25, temperature: float = 0.75,
) -> List[str]:
    """Generate ``n_new`` tokens with the feature clamped to ``value``.

    ``prompt`` may be a string, list of strings, or pre-tokenised tensor;
    the model's ``to_tokens`` is responsible for normalising it."""
    model.reset_hooks()
    hook_name = f"blocks.{layer}.mlp.hook_pre" if sae is None else _sae_hook_name(sae)
    model.add_hook(
        hook_name,
        functools.partial(_gen_hook, sae=sae, feature=feature, value=value),
    )
    try:
        toks = model.to_tokens(prompt)
        output = model.generate(toks, max_new_tokens=n_new, verbose=False, temperature=temperature)
    finally:
        model.reset_hooks()
    decoded = model.to_string(output)
    return [x[len(model.to_string(toks[i])):] for i, x in enumerate(decoded)]


def get_completions_for_kl_val(
    model, sae, prompts: List[str], layer: int, feature: int,
    target_kl: float, neg: bool = False, n_new: int = 25,
) -> List[str]:
    """Tune activation for ``target_kl`` then generate completions on
    ``prompts``. Newlines are escaped so completions are safe to embed
    in the LLM-picker prompt."""
    act = get_activation_for_kl(
        model, sae, prompts, layer, feature, abs(target_kl), neg=neg,
    )
    completions = hooked_gen(prompts, model, sae, layer, feature, value=act, n_new=n_new)
    return [c.replace("\n", "\\n").replace("\r", "\\r") for c in completions]


def get_pos_neg_acts(
    model, sae, pos: List[str], neg: List[str],
    layer: int, feature: int, pre_relu: bool = False,
) -> Tuple[float, float, float, float]:
    """Run pos + neg sentence sets through the model and read per-sentence
    feature activations from the SAE hook cache.

    Returns ``(pos_max_all, neg_max_all, pos_max_toks, neg_max_toks)``:
        - ``*_max_all`` — single max across (sentences × tokens)
        - ``*_max_toks`` — per-sentence max, then averaged across sentences
          (the metric's success criterion)."""
    if sae is None:
        pos_cache = model.run_with_cache(pos, return_type=None)[1]
        neg_cache = model.run_with_cache(neg, return_type=None)[1]
        block = f"blocks.{layer}.mlp.hook_post"
    else:
        pos_cache = model.run_with_cache_with_saes(pos, saes=[sae], return_type=None)[1]
        neg_cache = model.run_with_cache_with_saes(neg, saes=[sae], return_type=None)[1]
        relu = "pre" if pre_relu else "post"
        block = f"{_sae_hook_name(sae)}.hook_sae_acts_{relu}"

    pos_acts = pos_cache[block][:, :, feature]
    neg_acts = neg_cache[block][:, :, feature]
    return (
        float(pos_acts.max().item()),
        float(neg_acts.max().item()),
        float(pos_acts.max(dim=-1).values.mean().item()),
        float(neg_acts.max(dim=-1).values.mean().item()),
    )


def get_feature_activation_local(
    model, sae, text: str, layer: int, feature: int, pre_relu: bool = False,
) -> float:
    """Convenience wrapper: max activation of one feature on one text input."""
    pos_max, _, _, _ = get_pos_neg_acts(
        model, sae, [text], [text], layer, feature, pre_relu=pre_relu,
    )
    return pos_max


__all__ = [
    "GEN_PROMPTS_DEFAULT",
    "KL_DIV_VALUES_DEFAULT",
    "get_activation_for_kl",
    "get_completions_for_kl_val",
    "get_feature_activation_local",
    "get_kl_div",
    "get_pos_neg_acts",
    "hooked_gen",
]
