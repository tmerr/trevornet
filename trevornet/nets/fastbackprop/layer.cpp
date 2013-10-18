#include <iostream>
#include <vector>
#include "layer.hpp"

#define fvect std::vector<float>

using namespace tmerr;

Layer::Layer(int neuronCount, int nextCount) {
	signals = fvect(neuronCount, 0);
	errsignals = fvect(neuronCount, 0);
	weights = std::vector<fvect>(neuronCount, fvect(nextCount, 0));
}
