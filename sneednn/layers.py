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
        self.z_output = None # This is used for gradient descent. Dot product without activation
        self.key_ = jax.random.PRNGKey(1)
    
    def SetNext(self, layer):
        self.next_ = layer
    
    def GetNumNeurons(self):
        return self.num_neurons_
    
    def RecieveInput(self, inputs): # TODO: make sure the shape matches the num_neurons
        self.input_ = inputs
    
    def InitializeWeightsAndBiases(self): # TODO: deal with it when there's no next (last layer)
        """
        The weights matrix only represents a connection between layers.
        Therefore if there's no layer after this there's nothing to be initialized.
        Only the activation function should be applied across the inputs
        """
        if self.next_ == None:
            return

        self.weights_ = jax.random.uniform(self.key_, shape=(self.next_.GetNumNeurons(), self.num_neurons_), minval=-MINMAXVAL, maxval=MINMAXVAL)
        self.biases_ = jax.random.uniform(self.key_, shape=(self.next_.GetNumNeurons(), 1), minval=-MINMAXVAL, maxval=MINMAXVAL)
    
    def ComputeZ(self):
        if self.next_ == None:
            return

        self.z_output = self.weights_.dot(self.input_) + self.biases_
    
    def ComputeOutput(self) -> jax.Array:
        activation_vmap = jax.vmap(self.activation_)
        if self.next_ == None:
            return activation_vmap(self.input_)

        self.ComputeZ()
        return activation_vmap(self.z_output)
    
    def Describe(self):
        print("These are the weights:")
        print(self.weights_)
        print("These are the biases:")
        print(self.biases_)