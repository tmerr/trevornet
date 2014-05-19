import idx
from trevornet.nets.feedforward import PyFeedForwardNet
import os

class Ocr(object):
    def __init__(self, net):
        self._net = net

    @classmethod
    def fromfile(cls, fpath):
        with open(fpath) as f:
            thelist = eval(f.read())

        net = PyFeedForwardNet.fromlist(thelist)
        return cls(net)

    @classmethod
    def fromrandom(cls):
        net = PyFeedForwardNet.fromfunc((28*28, 25, 25, 10), .05)
        return cls(net)

    def train(self):
        traindatapath = os.path.join('trevornet', 'dataset', 'train-images.idx3-ubyte')
        trainlabelpath = os.path.join('trevornet', 'dataset', 'train-labels.idx1-ubyte')

        print("Parsing training data...")
        with open(traindatapath, 'rb') as f:
            traindata = idx.idx_to_list(f.read())

        print("Parsing training labels..")
        with open(trainlabelpath, 'rb') as f:
            trainlabels = idx.idx_to_list(f.read())

        print("Training...")

        numimages = len(traindata)
        count = 0
        for image, targetint in zip(traindata, trainlabels):
            count += 1

            pixels = [row for col in image for row in col]
            pixels = [float(x)/256 for x in pixels]
            targetlist = [0 for i in range(10)]
            targetlist[targetint] = 1

            assert len(pixels) == 28*28
            assert len(targetlist) == 10
            self._net.train(pixels, targetlist)

            print("Training image {0} of {1}".format(count, numimages))

        del traindata
        del trainlabels

    def tofile(self, fname):
        print("Done training. Dumping final net to '{0}'".format(fname))
        savednet = str(self._net.tolist())
        with open(fname, 'w') as f:
            f.write(savednet)

    def test(self):
        testdatapath = os.path.join('trevornet', 'dataset', 't10k-images.idx3-ubyte')
        testlabelpath = os.path.join('trevornet','dataset', 't10k-labels.idx1-ubyte')

        print("Parsing test data...")
        with open(testdatapath, 'rb') as f:
            testdata = idx.idx_to_list(f.read())

        print("Parsing test labels...")
        with open(testlabelpath, 'rb') as f:
            testlabels = idx.idx_to_list(f.read())

        successes, failures = 0, 0
        for image, targetint in zip(testdata, testlabels):
            pixels = [row for col in image for row in col]
            pixels = [float(x)/256 for x in pixels]
            targetlist = [0 for i in range(10)]
            targetlist[targetint] = 1

            outputs = self._net.predict(pixels)
            outputint = max(range(len(outputs)), key=outputs.__getitem__)

            if outputint == targetint:
                successes += 1
            else:
                failures += 1

            total = successes + failures
            thestr = "Predictions: {0}, Success rate: {1}%".format(
                total, (successes/total)*100
            )
            print(thestr)

        print("Prediction done!")
        thestr = "Predictions: {0}, Success rate: {1}%".format(
            total, (successes/total)*100
        )
        print(thestr)
