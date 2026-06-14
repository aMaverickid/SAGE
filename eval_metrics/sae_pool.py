"""(model, source) → (SAE, layer) cache for multi-layer eval runs.

The Input/Output evaluation metrics need:

    1. A single HookedSAETransformer for the target LLM (layer-agnostic).
    2. A specific SAE per feature, since SAGE results under ``results/``
       can span multiple layers (e.g., layer 0/3/7/11/23 for
       ``gemma-2-2b``). Using a single SAE for features that live on
       other layers silently mis-evaluates them — the steering hook
       fires at the wrong block index and the activations measured for
       the Input metric come from a mismatched feature dictionary.

``SAEPool`` loads the LLM once and lazily loads each SAE on first
lookup, templating a ``sae-lens://`` URI with the feature's layer
(parsed from the Neuronpedia source string — e.g.
``11-gemmascope-mlp-16k`` → layer 11). Subsequent lookups of the same
``(model, source)`` pair hit the cache.

Typical use::

    pool = SAEPool(
        target_llm="google/gemma-2-2b",
        sae_path_template=(
            "sae-lens://release=gemma-scope-2b-pt-mlp-canonical;"
            "sae_id=layer_{layer}/width_16k/canonical"
        ),
        device="cuda",
    )
    sae, layer = pool.get_for("gemma-2-2b", "11-gemmascope-mlp-16k")
    model = pool.model  # shared HookedSAETransformer
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

AUTO_SENTINEL = "auto"

_LAYER_FROM_SOURCE = re.compile(r"^(\d+)")


def layer_from_source(source: str) -> int:
    """Parse the leading layer index from a Neuronpedia source string.

    Examples
    --------
    >>> layer_from_source("11-gemmascope-mlp-16k")
    11
    >>> layer_from_source("0-res-jb")
    0
    >>> layer_from_source("5-resid-post-aa")
    5
    """
    match = _LAYER_FROM_SOURCE.match((source or "").strip())
    if not match:
        raise ValueError(
            f"Cannot extract layer from source {source!r}; "
            "expected a leading digit prefix (e.g. '11-gemmascope-mlp-16k')."
        )
    return int(match.group(1))


def render_sae_path(template: str, model: str, source: str, layer: int) -> str:
    """Substitute ``{layer}``, ``{model}``, ``{source}`` placeholders.

    Non-template paths (no placeholders) are returned unchanged, so the
    helper is safe to call when the caller is in single-layer mode.
    """
    return (
        template
        .replace("{layer}", str(layer))
        .replace("{model}", model)
        .replace("{source}", source)
    )


def is_template(sae_path: str) -> bool:
    """True iff ``sae_path`` can be resolved per-feature.

    Either a placeholder template (``{layer}``/``{model}``/``{source}``)
    or the literal ``"auto"`` sentinel — both produce a layer-correct SAE
    per ``(model, source)`` lookup.
    """
    if (sae_path or "").strip().lower() == AUTO_SENTINEL:
        return True
    return any(token in sae_path for token in ("{layer}", "{model}", "{source}"))


@lru_cache(maxsize=1)
def _sae_lens_directory() -> Any:
    """Cache the sae-lens pretrained directory (it parses YAML on import)."""
    from sae_lens.loading.pretrained_saes_directory import (
        get_pretrained_saes_directory,
    )
    return get_pretrained_saes_directory()


def resolve_via_neuronpedia(model: str, source: str) -> Tuple[str, str]:
    """Reverse-lookup the sae-lens ``(release, sae_id)`` for one Neuronpedia source.

    Each sae-lens release exposes a ``neuronpedia_id`` map from local
    ``sae_id`` → ``"{model}/{source}"``. We scan every release for that
    target string and return the first hit. If multiple releases publish
    the same ``neuronpedia_id`` (rare but happens for variant ports), the
    one whose release name ends in ``-canonical`` is preferred — that
    matches the SAE Neuronpedia actually serves.

    Args:
        model: Neuronpedia model id, e.g. ``"gemma-2-2b"``.
        source: Neuronpedia source string, e.g. ``"11-gemmascope-mlp-16k"``.

    Returns:
        ``(release, sae_id)`` ready for ``SAELensSAE.from_pretrained``.

    Raises:
        ValueError: if no release in the sae-lens registry exposes that
            neuronpedia id (typo, wrong model, or unsupported SAE).
    """
    target = f"{model}/{source}"
    directory = _sae_lens_directory()
    hits: List[Tuple[str, str]] = []
    for release, info in directory.items():
        np_map = getattr(info, "neuronpedia_id", None) or {}
        for sae_id, np_id in np_map.items():
            if np_id == target:
                hits.append((release, sae_id))
    if not hits:
        raise ValueError(
            f"No sae-lens release exposes Neuronpedia id {target!r}. "
            "Check that the model + source in your results/structured_results.json "
            "match an SAE published on Neuronpedia, or pass an explicit "
            "--sae_path template."
        )
    hits.sort(key=lambda r: (not r[0].endswith("-canonical"), r[0]))
    return hits[0]


def parse_sae_lens_uri(uri: str) -> Tuple[str, str]:
    """Pull ``(release, sae_id)`` from a ``sae-lens://release=...;sae_id=...`` URI."""
    if not uri.startswith("sae-lens://"):
        raise ValueError(f"Expected sae-lens:// URI, got {uri!r}")
    spec = uri[len("sae-lens://"):]
    kv: Dict[str, str] = {}
    for part in (p.strip() for p in spec.split(";") if p.strip()):
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k.strip()] = v.strip()
    release = kv.get("release") or kv.get("repo") or kv.get("model")
    sae_id = kv.get("sae_id") or kv.get("path")
    if not release or not sae_id:
        raise ValueError(f"Invalid sae-lens URI: missing release/sae_id in {uri!r}")
    return release, sae_id


class SAEPool:
    """Lazy, per-``(model, source)`` SAE cache sharing one HookedSAETransformer.

    The first :meth:`get_for` call triggers the LLM + initial SAE load
    via :class:`core.system.System`. Subsequent calls reuse the LLM and
    load only the new SAE via :func:`sae_lens.SAE.from_pretrained`.

    SAEs are keyed on ``(neuronpedia_model_id, source)`` because that
    pair already identifies a unique SAE in this codebase (Neuronpedia
    sources encode the layer; the random-amps pool filename is keyed on
    the same pair). The layer is derived from the source string.
    """

    def __init__(
        self, target_llm: str, sae_path_template: str, device: str = "cuda",
        dtype: str = "float32",
    ) -> None:
        """``dtype`` defaults to ``float32`` because TransformerLens's
        attention path (and SAELens encode/decode round-trips inside
        steering hooks) cannot tolerate Half/Float mixing — loading the
        SAE in fp16 still upcasts internally (JumpReLU threshold compare,
        b_dec, etc.), then the residual stream re-enters fp32 attention
        and trips ``q_ @ k_``. fp32 throughout is the only consistent
        path; on an 80GB A800 a 2B model + 16k SAE fits easily.
        """
        self.target_llm = target_llm
        self.sae_path_template = sae_path_template
        self.device = device
        self.dtype: str = dtype
        self._model: Any = None
        self._cache: Dict[Tuple[str, str], Tuple[Any, int]] = {}

    @property
    def model(self) -> Any:
        """The HookedSAETransformer. Available only after the first ``get_for`` call."""
        if self._model is None:
            raise RuntimeError(
                "SAEPool.model accessed before any get_for() call; "
                "the LLM is loaded lazily on first SAE lookup."
            )
        return self._model

    @property
    def is_initialized(self) -> bool:
        return self._model is not None

    def get_for(self, neuronpedia_model_id: str, source: str) -> Tuple[Any, int]:
        """Return ``(sae_obj, layer)`` for ``(model, source)``, loading on miss."""
        key = (neuronpedia_model_id, source)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        layer = layer_from_source(source)
        release, sae_id = self._resolve_release_and_id(neuronpedia_model_id, source, layer)
        if self._model is None:
            self._load_model()
        sae_obj = self._load_sae(release, sae_id, layer)
        self._cache[key] = (sae_obj, layer)
        return sae_obj, layer

    def _resolve_release_and_id(
        self, neuronpedia_model_id: str, source: str, layer: int,
    ) -> Tuple[str, str]:
        """Return ``(release, sae_id)`` for the URI template or for ``"auto"``."""
        if self.sae_path_template.strip().lower() == AUTO_SENTINEL:
            return resolve_via_neuronpedia(neuronpedia_model_id, source)
        rendered = render_sae_path(
            self.sae_path_template, neuronpedia_model_id, source, layer,
        )
        return parse_sae_lens_uri(rendered)

    def _load_model(self) -> None:
        """Load the HookedSAETransformer once.

        We bypass :class:`core.system.System` here because ``System._load_sae``
        swallows ``SAELensSAE.from_pretrained`` exceptions (printing only a
        warning) — that hides the real reason an SAE fails to load. Loading
        the model + SAE independently lets the SAELens exception propagate
        with its actual message.
        """
        from sae_lens import HookedSAETransformer  # type: ignore
        print(f"⟳ Loading LLM: {self.target_llm} → {self.device} ({self.dtype})")
        self._model = HookedSAETransformer.from_pretrained(
            model_name=self.target_llm,
            device=str(self.device),
            dtype=self.dtype,
        )

    def _load_sae(self, release: str, sae_id: str, layer: int) -> Any:
        """Load one SAE via :func:`sae_lens.SAE.from_pretrained`.

        The SAE is loaded in the same dtype as the LLM so encode/decode
        round-trips inside steering hooks don't trip TransformerLens's
        attention path with mixed Float/Half tensors. Errors propagate
        verbatim — typically a bad ``release`` / ``sae_id`` combo (the
        canonical gemma-scope MLP 16k release is
        ``gemma-scope-2b-pt-mlp-canonical``, not ``gemma-scope-2b-pt-mlp``).
        """
        from sae_lens import SAE as SAELensSAE  # type: ignore
        print(
            f"⟳ Loading SAE (layer {layer}): "
            f"release={release} sae_id={sae_id} dtype={self.dtype}"
        )
        return SAELensSAE.from_pretrained(
            release=release, sae_id=sae_id,
            device=str(self.device), dtype=self.dtype,
        )


__all__ = [
    "AUTO_SENTINEL",
    "SAEPool",
    "is_template",
    "layer_from_source",
    "parse_sae_lens_uri",
    "render_sae_path",
    "resolve_via_neuronpedia",
]
