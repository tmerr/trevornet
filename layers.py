import neurons

class InputLayer:
    def __init__(self, num_neurons):
        self._layer = Layer(num_neurons, neurons.InputNeuron)

    @property
    def neurons(self):
        return self._layer.neurons
    
    def connect_layer(self, other):
        self._layer.connect_layer(other)

    def setInputs(self, inputs):
        '''
        inputs: An ordered collection of floats that is matched to each neuron
        '''
        if len(inputs) != len(self._layer.neurons):
            raise ValueError(("inputs must have as many elements are there"
                              "are neurons"))
        for idx, n in enumerate(self._layer.neurons):
            n.signal = inputs[idx]

class OutputLayer:
    def __init__(self, num_neurons):
        self._layer = Layer(num_neurons, neurons.OutputNeuron)

    @property
    def neurons(self):
        return self._layer._neurons

    def connect_layer(self, other):
        self._layer.connect_layer(other)

    def propagate(self):
        self._layer.propagate()

    def backpropagate1(self):
        self._layer.backpropagate1()

    def backpropagate2(self):
        self._layer.backpropagate2()

class Layer:
    def __init__(self, num_neurons, neuron_type=neurons.Neuron):
        self._neurons = [neuron_type for x in range(num_neurons)]

    @property
    def neurons(self):
        return self._neurons

    def connect_layer(self, other):
        for n in self._neurons:
            for m in other.neurons:
                self._connect_neurons(n, m)

    def _connect_neurons(self, neuron1, neuron2, weight=1):
        connection = neurons.Connection(neuron1, neuron2, weight)
        neuron1.attach_forward(connection)
        neuron2.attach_back(connection)

    def propagate(self):
        for n in self._neurons:
            n.propagate()

    def backpropagate1(self):
        for n in self._neurons:
            n.backpropagate1()

    def backpropagate2(self):
        for n in self._neurons:
            n.backpropagate2()

