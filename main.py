from sneednn import FeedForwardDNN
from sneednn.functions import Sigmoid, Linear, SoftMax
from sneednn.layers import Dense
import jax.numpy as jnp


# nn = FeedForwardDNN([
#     Dense(num_neurons=2, activation=Linear),
#     Dense(num_neurons=3, activation=Sigmoid),
#     Dense(num_neurons=2, activation=Sigmoid),
# ])

layer1 = Dense(num_neurons=2, activation=Linear)
layer2 = Dense(num_neurons=3, activation=Sigmoid)
inputs = jnp.array([[1], [0]])
layer1.RecieveInput(inputs)
layer1.SetNext(layer2)
layer1.InitializeWeightsAndBiases()
layer2.RecieveInput(layer1.ComputeOutput())
print(layer2.ComputeOutput())