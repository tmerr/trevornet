#TrevorNet

A simple feedforward neural net that learns through backpropagation. It's set up with a the example tasks xor, the sin function, and character recognition. The character recognition takes a couple hours to train but it works with an 83% success rate on the [the MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/). I'm sure I could get better by fiddling with the learning difficulty and number of neurons. I used 784 x 25 x 25 x 10 with a learning rate of .02.

If you want to test it out
- check out the repository
- extract the four files from [here](http://yann.lecun.com/exdb/mnist/) into the database folder
- cd to the folder this readme is in and run python -m trevornet.tasks

Distributed under the MIT License.
