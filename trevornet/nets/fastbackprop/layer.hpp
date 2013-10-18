#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>

#define fvect std::vector<float>

namespace tmerr {

// A layer of neurons.
struct Layer {	
	// Construct the layer with all signals, errsignals, and weights set
	// to 0.
	Layer(int neuronCount, int nextCount);

	fvect signals;
	fvect errsignals;
	std::vector<fvect> weights;
};

}

#endif
