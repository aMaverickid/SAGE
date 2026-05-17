#!/usr/bin/env python3
"""Create reproducible feature-selection manifests for Agent4Interp experiments."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class FeatureSpec:
    model_name: str
    neuronpedia_model_id: str
    layer: str
    layer_index: int
    source: str
    feature_index: int
    split: str


MODEL_DEFAULTS: Dict[str, Dict[str, str]] = {
    "gemma-2-2b": {
        "target_llm": "google/gemma-2-2b",
        "source_template": "{layer}-gemmascope-mlp-16k",
    },
    "gpt2-small": {
        "target_llm": "gpt2",
        "source_template": "{layer}-res-jb",
    },
    "llama3.1-8b-it": {
        "target_llm": "meta-llama/Llama-3.1-8B-Instruct",
        "source_template": "{layer}-resid-post-aa",
    },
}


def parse_layers(raw_layers: str) -> List[int]:
    layers: List[int] = []
    for item in raw_layers.split(","):
        item = item.strip()
        if not item:
            continue
        layers.append(int(item))
    return layers


def sample_features_for_layer(
    model_id: str,
    layer_index: int,
    features_per_layer: int,
    feature_min: int,
    feature_max: int,
    rng: random.Random,
    split: str,
) -> List[FeatureSpec]:
    defaults = MODEL_DEFAULTS.get(model_id, MODEL_DEFAULTS["gemma-2-2b"])
    source = defaults["source_template"].format(layer=layer_index)
    layer_name = f"layer{layer_index}"
    sampled = rng.sample(range(feature_min, feature_max + 1), features_per_layer)
    return [
        FeatureSpec(
            model_name=defaults["target_llm"],
            neuronpedia_model_id=model_id,
            layer=layer_name,
            layer_index=layer_index,
            source=source,
            feature_index=feature,
            split=split,
        )
        for feature in sampled
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reproducible feature selection manifest.")
    parser.add_argument("--models", default="gemma-2-2b", help="Comma-separated Neuronpedia model IDs")
    parser.add_argument("--layers", default="0", help="Comma-separated numeric layer indices")
    parser.add_argument("--features_per_layer", type=int, default=50)
    parser.add_argument("--feature_min", type=int, default=0)
    parser.add_argument("--feature_max", type=int, default=15999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="pilot", choices=["pilot", "main", "polysemantic"])
    parser.add_argument("--output", default="experiment_manifests/feature_protocol.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    selected: List[FeatureSpec] = []
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    layers = parse_layers(args.layers)

    for model_id in models:
        for layer_index in layers:
            selected.extend(
                sample_features_for_layer(
                    model_id=model_id,
                    layer_index=layer_index,
                    features_per_layer=args.features_per_layer,
                    feature_min=args.feature_min,
                    feature_max=args.feature_max,
                    rng=rng,
                    split=args.split,
                )
            )

    output = {
        "protocol": {
            "seed": args.seed,
            "split": args.split,
            "models": models,
            "layers": layers,
            "features_per_layer": args.features_per_layer,
            "feature_range": [args.feature_min, args.feature_max],
            "selection_method": "uniform_without_replacement_per_model_layer",
        },
        "features": [asdict(item) for item in selected],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(selected)} feature specs to {out_path}")
    for item in selected[:10]:
        print(f"{item.neuronpedia_model_id} {item.source} feature={item.feature_index}")


if __name__ == "__main__":
    main()
