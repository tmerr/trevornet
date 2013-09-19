from neurons import InputNeuron, HiddenNeuron, OutputNeuron

class FeedForwardNet(object):
    """A backpropagating feed forward neural network."""

    def __init__(self, *layers):
        """Parameters are the neuron count for each layer. For example to make a
        network with 2 input neurons, 2 hidden neurons, and 1 output neuron
        you would pass (2, 2, 1). A four layer network could be constructed with
        (1, 3, 3, 7)."""
        if len(layers) < 2:
            raise ValueError("Need at least two layers")

        #create layers of neurons (which can also be accessed via properties)
        self._layers = [
            [InputNeuron() for i in range(layers[0])],
            [[HiddenNeuron() for j in range(layers[k])]
            for k in range(1, len(layers)-1)],
            [OutputNeuron() for l in range(layers[-1])]
        ]

        #connect layers
        for i in range(len(self._layers)-1):
            for pair in itertools.product(self._layers[i], self._layers[i+1]):
                pair[0].connect_forward(pair[1], 1)

    def train(self, data, targets):
        """Train the neural net with the given input data and targets.

        Params:
            data: Some sequence of input data to map to input neurons
            targets: Some sequence of output data to compare against output neurons
        """

        if len(data) != len(self.inputlayer):
            raise ValueError("Data must have same number of elements as input layer")
        elif len(targets) != len(self.outputlayer):
            raise ValueError("Targets must have same number of elements as output layer")

        #propagate
        self._propagate(data)
        
        #backpropagate
        for idx, val in enumerate(targets):
            self.outputlayer[idx].backpropagate1(val)
        for layer in self.hiddenlayers[::-1]:
            for neuron in layer:
                neuron.backpropagate()

    def predict(self, data):
        """Predict the target for the data.
        
        Params:
            data: Some sequence of input data to map to input neurons

        Return:
            a sequence of numerical signals from the output neurons.
        """
        if len(data) != len(self.inputlayer):
            raise ValueError("Data must have same number of elements as input layer")

        self._propagate(data)
        out = [neuron.signal for neuron in self.outputlayer]
        return out

    def _propagate(self, data):
        for idx, val in enumerate(data):
            self.inputlayer[idx] = val
        for layer in self._layers[:-1]:
            for neuron in layer:
                neuron.propagate()

    @property
    def inputlayer(self):
        return self._layers[0]

    @property
    def hiddenlayers(self):
        return self._layers[1:-1]

    @property
    def outputlayer(self):
        return self._layers[-1]

