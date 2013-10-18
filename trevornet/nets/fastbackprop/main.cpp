#include <iostream>
#include <vector>
#include "layer.hpp"

int main(int argc, char **argv) {

	// get inputs
	int numNeurons = 30;
	int numForward = 30;

	std::cout << "hello world";

	tmerr::Layer layer = tmerr::Layer(30, 30);

	for (int i=0; i<30; i++) {
		std::cout << layer.signals[i] << std::endl;
	}

	return 0;
}
