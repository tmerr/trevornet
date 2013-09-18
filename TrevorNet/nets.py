from layers import InputLayer, HiddenLayer, OutputLayer
import math
import random

class FeedForwardNet(object):
    def __init__(self, inlayersize, layersize, outlayersize):
        self._inlayer = InputLayer(inlayersize)
        self._hiddenlayer = HiddenLayer(layersize)
        self._outlayer = OutputLayer(outlayersize)

        self._inlayer.connect_layer(self._hiddenlayer)
        self._hiddenlayer.connect_layer(self._outlayer)

    @property
    def neurons(self):
        return [self._inlayer.neurons, self._hiddenlayer.neurons, self._outlayer.neurons]

    def train(self, inputs, targets, verbose=False):
        '''
        inputs: a sequence of floats that map to the input neurons
        targetlabels: a sequence of floats that are the desired output neuron
                      values.
        '''

        self._inlayer.inputs = inputs
        self._hiddenlayer.propagate()
        self._outlayer.propagate()
        
        self._outlayer.backpropagate1(targets)
        self._hiddenlayer.backpropagate1()

        self._outlayer.backpropagate2()
        self._hiddenlayer.backpropagate2()

        if verbose:
            print("Training results")
            print("\tInput: {0}".format(inputs))
            print("\tTarget output: {0}".format(targets))
            print("\tActual output: {0}".format(self._outlayer.outputs))
            self.display_signals()
            self.display_weights_from(0, 0)
            print("")
            raw_input()

    def predict(self, inputs):
        '''
        inputs: a sequence of floats that map to the input neurons
        return: a sequence of floats mapped from the output neurons
        '''
        self._inlayer.inputs = inputs
        self._hiddenlayer.propagate()
        self._outlayer.propagate()
        return self._outlayer.outputs

    def display_signals(self):
        col1 = self._inlayer.inputs
        col2 = [x.signal for x in self._hiddenlayer.neurons]
        col3 = self._outlayer.outputs
        numrows = max(len(col1), len(col2), len(col3))

        roundto = 3 #round to
        print("Signals")
        print("\tInput\tHidden\tOutput")
        for row in range(numrows):
            line = []
            for col in col1, col2, col3:
                if len(col)-1 < row:
                    line.append("")
                else:
                    element = round(col[row], roundto)
                    element = str(element)
                    line.append(element)
            print('\t' + '\t'.join(line))

    def display_weights_from(self, layer, neuron):
        l = (self._inlayer, self._hiddenlayer, self._outlayer)[layer]
        n = l.neurons[neuron]
        out = []
        for i, conn in enumerate(n._forward):
            out.append('[{0}]={1}'.format(i, conn.weight))
        print(' '.join(out))
        print('')
