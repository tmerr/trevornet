#ifndef NET_H
#define NET_H

#include <vector>
#include "layer.hpp"

#define fvect std::vector<float>

namespace tmerr {

// A neural net that implements backpropagation.
class Net {
	public:

	// Construct the net with the given layer sizes.
	// For example the vector {3, 4, 6, 5} will make a 4 layer net.	
	// The first layer is the input layer, and the last is the output.
	Net(std::vector<int> sizes);

	void setWeights(std::vector<std::vector<fvect>>);
	std::vector<std::vector<fvect>> getWeights();

	void train(fvect inputs, fvect target);
	fvect predict(fvect inputs);
	
	private:
	std::vector<tmer::Layer> layers;
	void propagate();
	void backpropagate();
}

}

#endif
