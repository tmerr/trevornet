#TrevorNet

A neural net using backpropagation including three example tasks: optical character recognition, sin function approximation, and xor approximation. If you're looking for a library that's good and fast, you've come to the wrong place! All that matters here is that it's readable. There aren't any dependencies other than Python3 yet.


###Optical character recognition: 
The optical character recognition has a GUI that lets you draw numbers as a challenge for the AI to recognize it. To train this neural net you'll need each of the four files in [the MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/) extracted to the dataset folder. I haven't played with the neuron count or learning rate much, but 784 x 25 x 25 x 10 net with learning rate of 0.02 scores 83% correct on the test set. Also included is the pre-trained `ocrnet.dat`. To get started enter

```python -m trevornet.ocr```

###Xor

An approximation of the xor function.
```python -m trevornet.xor```

###Sin

An approximation of the sin function.
```python -m trevornet.sin```

# License

MIT
