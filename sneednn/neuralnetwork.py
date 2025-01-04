import jax.numpy as jnp


class FeedForwardDNN():
    def __init__(self, layers: list):
        self.layers_ = layers
        self._ConnectLayers()
    
    def _ConnectLayers(self):
        for i in range(len(self.layers_) - 1):
            self.layers_[i].SetNext(self.layers_[i + 1])
        
        for layer in self.layers_:
            layer.InitializeWeightsAndBiases()