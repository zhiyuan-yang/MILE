"""Basic fully connected neural networks."""
import flax.linen as nn
import jax.numpy as jnp

from module_sandbox.config.models.fcn import FCNConfig, ResFCNConfig
from module_sandbox.flax_building_blocks.basic import (
    FullyConnected,
    ResidualFullyConnected,
)


class FCN(nn.Module):
    """Fully connected neural network."""

    config: FCNConfig

    def setup(self):
        """Initialize the fully connected neural network."""
        self.fcn = FullyConnected(
            hidden_sizes=self.config.hidden_structure,
            activation=self.config.activation.flax_activation,
            use_bias=self.config.use_bias,
            last_layer_activation=None,
            blockid=None,
        )

    def __call__(self, x: jnp.ndarray):
        """Forward pass."""
        return self.fcn(x)
