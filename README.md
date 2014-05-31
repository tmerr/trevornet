#TrevorNet

This is the repo I'm using to keep track of my AI related projects. I'm just learning the basics, working on questions like "What, neural nets are a thing?" and "How can I use that to take over the planet?". So far I have
- A (slow!) python implementation of a neural network using backpropagation as the learning algo. 
- Code to read in the MNIST database of handwritten digits.
- A GUI that lets you draw letters/numbers by hand and see the neural net's guess of what it is.

These commands do things if you run them from within this folder...

```python -m trevornet.ocr``` ```python -m trevornet.sin``` ```python -m trevornet.xor```

It works, sort of. The saved neural net in this directory (ocrnet.dat) gets 83% of predictions correct in MNIST's test set (note: it was trained with a 784 x 25 x 25 x 10 net with learning rate of 0.02). But when you test out that same net with the GUI it doesn't work nearly as well. That's probably because the GUI doesn't size normalize whereas MNIST does. But even then there are more problems- 83% leaves much to be desired. Is it the error function? Activation function? Layer sizes? Learning rate? No idea. I don't think there's much to gain from unguided trial and error so I'll come back to this when I learn more theory.

License: MIT
