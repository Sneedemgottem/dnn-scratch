import jax.numpy as jnp
from abc import ABC, abstractmethod

# activation functions

# can use jax.grad() for derivatives but this is a research project for me :)
class Activation(ABC):
    @abstractmethod
    def Function(self, n):
        pass

    @abstractmethod
    def Derivative(self, n):
        pass

    def __call__(self, n):
        return self.Function(n)

class Sigmoid(Activation):
    def Function(self, n: float) -> float:
        return 1 / (1 + jnp.exp(-n))
    
    def Derivative(self, n: float) -> float:
        return self.Function(n) * (1 - self.Function(n))

class Linear(Activation):
    def Function(self, n: float) -> float:
        return n
    
    def Derivative(self, n: float) -> float:
        return 1


# used to turn output layer into probabilities for classification networks
class SoftMax(Activation):
    def Function(self, logits: jnp.ndarray) -> jnp.ndarray:
        logits_1d = logits.flatten()
        exp_logits = jnp.exp(logits_1d - jnp.max(logits_1d))
        res = exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)
        return res.reshape(-1, 1)
    
    def Derivative(self, n):
        pass