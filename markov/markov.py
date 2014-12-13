import string
import random
import time
import re
import sys


'''
This is an implementation of a markov chain used for text generation.
Just pass a file name as an argument and it should load it up, build a markov
chain with a state for each word, and start walking through the chain, writing
incoherent text to the terminal.
'''


asciiset = set(string.ascii_letters)
asciiset.add(' ')
asciiset.add('.')
def strip2ascii(txt):
    return ''.join([ch for ch in txt if ch in asciiset])


def tokenize(fname):
    '''
    Generate tokens defined by
    - Sequences of characters that aren't spaces
    - Periods

    For example, 'This is a test. Ok.' => ('This', 'is', 'a', 'test', '.', 'Ok, '.')
    '''
    with open(fname, 'r') as f:
        for line in f:
            stripped = strip2ascii(line)
            for word in stripped.split():
                
                if word[-1] == '.':
                    yield word[:-1]
                    yield '.'
                else:
                    yield word


def buildtransitionmap(tokens):
    dct = {}
    prev = '.'

    for word in tokens:
        if prev in dct:
            dct[prev].append(word)
        else:
            dct[prev] = [word]

        prev = word
        
    return dct


def transition(word, transmap):
    return random.choice(transmap[word])


def eternalramble(fname):
    '''Walk through the markov chain printing out words to the terminal one at a time'''
    transmap = buildtransitionmap(tokenize(fname))
    word = '.'
    while True:
        word = transition(word, transmap)
        print(word, end=' ')
        sys.stdout.flush()
        time.sleep(0.25)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        eternalramble(sys.argv[1])
    else:
        print('Usage: markov filename')
        print('  filename: the filename of the text to base the markov chain on.')
