"""Description evaluation metrics for SAE feature descriptions.

This package implements the Input Metric and Output Metric described in
``feature_descriptions_pipeline.ipynb`` (Description Evaluation section),
adapted to SAGE's Neuronpedia-API-based pipeline.

Public entry points:
    - ``input_metric.compute_input_score``
    - ``output_metric.compute_output_score``
    - ``shared.discover_variant_features``
"""
from .input_metric import (
    DEFAULT_ACTIVATION_THRESHOLD,
    DEFAULT_MODERATE_THRESHOLD,
    DEFAULT_N_EXAMPLES,
    DEFAULT_PREDICTIVE_LLM_MODEL,
    InputScore,
    compute_input_score,
    generate_test_examples,
)
from .input_predictive import (
    compute_predictive_accuracy,
    predictive_cache_path,
)
from .output_metric import OutputScore, compute_output_score, build_steered_set
from .shared import (
    DEFAULT_DESCRIPTION_FILENAME,
    DEFAULT_LABEL_FILENAME,
    DEFAULT_TEXT_FILENAME,
    description_hash,
    discover_variant_features,
    load_description,
    load_feature_text,
    load_labels,
)

__all__ = [
    "DEFAULT_ACTIVATION_THRESHOLD",
    "DEFAULT_DESCRIPTION_FILENAME",
    "DEFAULT_LABEL_FILENAME",
    "DEFAULT_MODERATE_THRESHOLD",
    "DEFAULT_N_EXAMPLES",
    "DEFAULT_PREDICTIVE_LLM_MODEL",
    "DEFAULT_TEXT_FILENAME",
    "InputScore",
    "OutputScore",
    "build_steered_set",
    "compute_input_score",
    "compute_output_score",
    "compute_predictive_accuracy",
    "description_hash",
    "discover_variant_features",
    "generate_test_examples",
    "load_description",
    "load_feature_text",
    "load_labels",
    "predictive_cache_path",
]
