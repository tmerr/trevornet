#TrevorNet

This repo is for any code I write related to AI. So far there's an implementation of a neural net using backpropagation, paired with some example tasks that can be launched with

'''
python -m trevornet ocr
python -m trevornet xor
python -m trevornet sin
'''

The optical character recognition one has a gui where you can scribble in numbers and see how bad of a job it does predicting what they are. It trains on the  [the MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/) and is ungodly slow. After 2 hours of training on a 784 x 25 x 25 x 10 net (learning rate .02) it scored 83% correct guesses on the test set. It should be possible to do better with different parameters.

The only dependency is the MNIST database. Extract the four files [here](http://yann.lecun.com/exdb/mnist/) to the dataset folder and you're god to go. Also for the lazy (*cough* me) there's an included net already trained called ocrnet.dat.

Distributed under the MIT License.
