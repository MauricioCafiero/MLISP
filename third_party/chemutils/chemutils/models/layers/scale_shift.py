
import haiku as hk

import e3nn_jax as e3nn

class ScaleShiftLayer(hk.Module):
    """Scales and shifts the input by learnable parameters."""

    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.scale = hk.get_parameter(
            "scale", shape=(), init=hk.initializers.Constant(scale))
        self.shift = hk.get_parameter(
            "shift", shape=(), init=hk.initializers.Constant(shift))

    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        return self.scale * x + self.shift

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(scale={self.scale:.6f}, shift={self.shift:.6f})"
        )
