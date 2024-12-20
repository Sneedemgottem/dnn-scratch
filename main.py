from sneednn import FeedForwardDNN
from sneednn.functions import Sigmoid, Linear
from sneednn.layers import Dense


nn = FeedForwardDNN([
    Dense(num_neurons=2, activation=Linear),
    Dense(num_neurons=3, activation=Sigmoid),
    Dense(num_neurons=2, activation=Sigmoid),
])