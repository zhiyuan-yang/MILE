"""Various Deep Learning Models and their configurations written in jax."""
from module_sandbox.models.images import LeNet, LeNetti
from module_sandbox.models.tabular import FCN, ResFCN
from module_sandbox.models.text import (
    GPT,
    AttentionClassifier,
    EmbeddingClassifier,
    ToySeqSeqModel,
)

__all__ = [
    'GPT',
    'AttentionClassifier',
    'ToySeqSeqModel',
    'EmbeddingClassifier',
    'LeNet',
    'LeNetti',
    'FCN',
    'ResFCN',
]
