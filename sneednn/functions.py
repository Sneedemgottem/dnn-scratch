import jax.numpy as jnp
from abc import ABC, abstractmethod

# activation functions

# can use jax.grad() for derivatives but this is a research project for me :)
class Activation(ABC):
    @abstractmethod
    def Function(self, n: float) -> float:
        pass

    @abstractmethod
    def Derivative(self, n: float) -> float:
        pass

    def __call__(self, n):
        return self.Function(n)

class Sigmoid(Activation):
    def Function(self, n):
        return 1 / (1 + jnp.exp(-n))
    
    def Derivative(self, n):
        return self.Function(n) * (1 - self.Function(n))

class Linear(Activation):
    def Function(self, n):
        return n
    
    def Derivative(self, n):
        return 1