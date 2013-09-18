import neurons

class InputLayer(object):
    def __init__(self, num_neurons):
        self._layer = HiddenLayer(num_neurons, neurons.InputNeuron)

    @property
    def neurons(self):
        return self._layer.neurons
    
    def connect_layer(self, other):
        self._layer.connect_layer(other)

    @property
    def inputs(self):
        return [n.signal for n in self._layer.neurons]

    @inputs.setter
    def inputs(self, value):
        '''
        value: A sequence of floats that is mapped to each neuron
        '''
        if len(value) != len(self._layer.neurons):
            raise ValueError(("num floats must be same as num neurons"))
        for idx, n in enumerate(self._layer.neurons):
            n.signal = value[idx]

class OutputLayer(object):
    def __init__(self, num_neurons):
        self._layer = HiddenLayer(num_neurons, neurons.OutputNeuron)

    @property
    def outputs(self):
        return [n.signal for n in self._layer.neurons]

    @property
    def neurons(self):
        return self._layer._neurons

    def connect_layer(self, other):
        self._layer.connect_layer(other)

    def propagate(self):
        self._layer.propagate()

    def backpropagate1(self, targetlabels):
        for idx, n in enumerate(self._layer._neurons):
            n.backpropagate1(targetlabels[idx])

    def backpropagate2(self):
        self._layer.backpropagate2()

class HiddenLayer(object):
    def __init__(self, num_neurons, neuron_type=neurons.Neuron, neuron_params=()):
        self._neurons = [neuron_type(*neuron_params) for x in range(num_neurons)]

    @property
    def neurons(self):
        return self._neurons

    def connect_layer(self, other):
        for n in self._neurons:
            for m in other.neurons:
                self._connect_neurons(n, m)

    def _connect_neurons(self, back, front, weight=1):
        connection = neurons.Connection(back, front, weight)
        back.attach_forward(connection)
        front.attach_back(connection)

    def propagate(self):
        for n in self._neurons:
            n.propagate()

    def backpropagate1(self):
        for n in self._neurons:
            n.backpropagate1()

    def backpropagate2(self):
        for n in self._neurons:
            n.backpropagate2()

