"""Description evaluation metrics for SAE feature descriptions.

This package implements the Input Metric and Output Metric described in
``feature_descriptions_pipeline.ipynb`` (Description Evaluation section),
adapted to SAGE's Neuronpedia-API-based pipeline.

Public entry points:
    - ``input_metric.compute_input_score``
    - ``output_metric.compute_output_score``
    - ``shared.discover_variant_features``
"""
from .input_metric import InputScore, compute_input_score, generate_pos_neg_sentences
from .output_metric import OutputScore, compute_output_score, build_steered_set
from .shared import (
    DEFAULT_LABEL_FILENAME,
    description_hash,
    discover_variant_features,
    load_description,
    load_feature_text,
    load_labels,
)

__all__ = [
    "DEFAULT_LABEL_FILENAME",
    "InputScore",
    "OutputScore",
    "build_steered_set",
    "compute_input_score",
    "compute_output_score",
    "description_hash",
    "discover_variant_features",
    "generate_pos_neg_sentences",
    "load_description",
    "load_feature_text",
    "load_labels",
]
