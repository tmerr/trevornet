from trevornet import idx
from trevornet.nets.feedforward import PyFeedForwardNet
import os
from tkinter import *
from tkinter import filedialog
import multiprocessing
import queue
import sys


class OcrPresentation(object):
    '''
    Lets the user modify self.pixels via a GUI by dragging left click.
    Points are automatically sent to the neural net which updates a label.
    '''
    def __init__(self):
        self.root = Tk()
        self.root.title("Character recognition")
        self.root.resizable(0, 0)

        frame = Frame(self.root)
        frame.pack()
        loadbtn = Button(frame, text='Load existing net', command=self.load)
        loadbtn.pack(side=LEFT)
        trainbtn = Button(frame, text='Create and train new net', command=self.trainnew)
        trainbtn.pack(side=LEFT)

        self.canvas = Canvas(self.root, bg="white", width=28*8, height=28*8)
        self.canvas.configure(cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.drawpoint)

        frame2 = Frame(self.root)
        frame2.pack()
        clearbtn = Button(frame2, text='Clear', command=self.clearpressed)
        clearbtn.pack(side=LEFT)
        self.label = Label(self.root, text='-1')
        self.label.pack()
        exit = Button(frame2, text='Exit', command=self.exitpressed)
        exit.pack(side=LEFT)

        self.pixels = [[0 for x in range(28)] for y in range(28)]

        self.worker = None

        self.root.after(0, self.update())
        self.root.mainloop()

    def startworker(self, fname):
        if not self.worker is None:
            self.worker.terminate()
        self.output_queue = multiprocessing.Queue(1)
        self.input_queue = multiprocessing.Queue(1)
        self.worker = multiprocessing.Process(target=OcrWorker,
                                              args=(fname, self.output_queue, self.input_queue),
                                              daemon=True)
        self.worker.start()
        self.output_queue.put_nowait(self.pixels)

    def trainnew(self):
        fname = filedialog.asksaveasfilename(parent=self.root, title='Choose where to save the trained net to')
        net = Ocr.fromrandom()
        net.train()
        net.tofile(fname)
        self.startworker(fname)

    def load(self):
        fname = filedialog.askopenfilename(parent=self.root, title='Choose a file to neural net file to use')
        self.startworker(fname)

    def exitpressed(self):
        self.root.quit()
        if not self.worker is None:
            self.worker.terminate()

    def clearpressed(self):
        self.pixels = [[0 for x in range(28)] for y in range(28)]
        self.canvas.create_rectangle(0, 0, 28*8, 28*8, fill='white', outline='white')

    def drawpoint(self, event):
        gridx, gridy = event.x//8, event.y//8
        if 0 < gridx > 27 or 0 < gridy > 27:
            return
        darkness = min(self.pixels[gridy][gridx] + 1, 1)
        self.pixels[gridy][gridx] = darkness

        screenx, screeny = gridx*8, gridy*8
        hexdigits = '0123456789abcdef'
        fillcolor = '#{0}{0}{0}'.format(hexdigits[15-int(darkness*15)])
        self.canvas.create_rectangle(screenx, screeny, screenx+7, screeny+7,
                                     fill=fillcolor, outline=fillcolor)

    def update(self):
        # Try to push an image out to the net, and check for any results that
        # came back.
        if not self.worker is None:
            try:
                prediction = self.input_queue.get_nowait()
                self.label.config(text='Prediction: {0}'.format(prediction))
                self.output_queue.put_nowait(self.pixels)
            except queue.Empty:
                pass
            except queue.Full:
                print("This shouldn't happen.")
        self.root.after(20, self.update)


class OcrWorker(object):
    def __init__(self, fname, input_queue, output_queue):
        self.net = PyFeedForwardNet.fromfile(fname)
        if len(self.net.inputlayer) != 28*28:
            raise ValueError('Expected a 28*28 input layer')
        if len(self.net.outputlayer) != 10:
            raise ValueError('Expected a size 10 output layer')

        self.input_queue = input_queue
        self.output_queue = output_queue
        self.loop()

    def loop(self):
        while True:
            image = self.input_queue.get()
            flatpixels = [row for col in image for row in col]
            outputs = self.net.predict(flatpixels)
            predictedint = max(range(len(outputs)), key=outputs.__getitem__)
            self.output_queue.put(predictedint)


class Ocr(object):
    '''
    Neural net with a 28*28 input layer and size 10 output layer with methods
    for training and testing using the MNIST database.
    '''

    def __init__(self, net):
        self._net = net

    @classmethod
    def fromfile(cls, fpath):
        net = PyFeedForwardNet.fromfile(fpath)
        if len(net.inputlayer) != 28*28:
            raise ValueError('Ocr net expected a 28*28 input layer')
        if len(net.outputlayer) != 10:
            raise ValueError('Ocr net expected a size 10 output layer')
        return cls(net)

    @classmethod
    def fromrandom(cls):
        net = PyFeedForwardNet.fromfuncs((28*28, 25, 25, 10), .05)
        return cls(net)

    def tofile(self, fname):
        savednet = str(self._net.tolist())
        with open(fname, 'w') as f:
            f.write(savednet)

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

if __name__ == '__main__':
    OcrPresentation()
