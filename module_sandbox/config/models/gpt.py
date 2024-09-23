"""GPT model configuration."""
from dataclasses import dataclass, field

from module_sandbox.config.models.base import FloatPrecision, ModelConfig


@dataclass(frozen=True)
class AttentionClassifierConfig(GPTConfig):
    """Attention Classifier Configuration."""

    model: str = 'AttentionClassifier'
    n_classes: int = field(
        default=2, metadata={'description': 'Number of classes for classification'}
    )


@dataclass(frozen=True)
class EmbeddingClassifierConfig(AttentionClassifierConfig):
    """Embedding Classifier Configuration."""

    model: str = 'EmbeddingClassifier'
