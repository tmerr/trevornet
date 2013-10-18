#include <vector>
#include <algorithm>
#include "layer.hpp"

#define fvect std::vector<float>

using namespace tmerr;

Net::Net(std::vector<int> &layerSizes) {
	// construct each layer hooked up to the next layer
	layers = std::vector<tmerr::Layer>();
	for (auto it = layerSizes.begin(); it != layerSizes.end()-1; ++it) {
		layers.push_back(tmerr::Layer(*it, *(it+1)));
	}
	layers.back() = tmerr:Layer(layerSizes.back(), 0);
}

Net::setWeights(std::vector<std::vector<fvect>> weights) {
}

std::vector<std::vector<fvect>> Net::getWeights() {
}

void Net::train(fvect inputs, fvect target) {
}

fvect Net::predict(fvect inputs) {
}

void Net::propagate() {
	// Per layer per neuron, sum the weight*signals of the prior neurons
	// and apply a sigmoid.
	
	// transform each current layer to next layer signals
	for (auto it = layers.begin(); it != layers.end(); ++it) {
		curlayer = *it;
		nextlayer = *(it+1);

		std::transform(curlayer.signals.begin(), curlayer.signals.end(),
				curlayer.weights.begin(), nextlayer.signals.begin()
				[](float signal, std::vector<float> weights){
			return std::accumulate
		});

		for (int i=0; i<nextlayer.signals.size(); i++) {
			for (int j=0; j<curlayer.signals.size(); j++) {
				// what about bias?
				nextlayer.signals[i] += curlayer.signals[j]*curlayer.weights[j][i]
			}
			nextlayer.signals[i] = activationfunc(nextlayer.signals[i]);
		}
	}
}

void Net::backpropagate() {
}
