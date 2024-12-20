import jax
from sneednn import functions

MINMAXVAL = 10

class Dense():
    def __init__(self, num_neurons: int, activation: functions.Activation, next=None):
        self.num_neurons_ = num_neurons
        self.activation_ = activation()
        self.next_: Dense = next
        self.weights_ = None
        self.biases_ = None
        self.input_ = None
        self.key_ = jax.random.PRNGKey(1)
    
    def SetNext(self, layer):
        self.next_ = layer
    
    def GetNumNeurons(self):
        return self.num_neurons_
    
    def RecieveInput(self, inputs): # TODO: make sure the shape matches the num_neurons
        self.input_ = inputs
    
    def InitializeWeightsAndBiases(self): # TODO: deal with it when there's no next (last layer)
        self.weights_ = jax.random.uniform(self.key_, shape=(self.next_.GetNumNeurons(), self.num_neurons_), minval=-MINMAXVAL, maxval=MINMAXVAL)
        self.biases_ = jax.random.uniform(self.key_, shape=(self.next_.GetNumNeurons(), 1), minval=-MINMAXVAL, maxval=MINMAXVAL)
    
    def Describe(self):
        print("These are the weights:")
        print(self.weights_)
        print("These are the biases:")
        print(self.biases_)