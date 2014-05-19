from trevornet.nets.feedforward import PyFeedForwardNet


def test_fromfuncs_doesnt_crash():
    PyFeedForwardNet.fromfuncs((2, 2, 1), 0.02)


def test_constructfromlist_doesnt_crash():
    inputsize = 2
    hidden = ((0.02, -0.5, (0.5, -0.3)), (0.02, 0.3, (0.2, 0.9)))
    output = ((0.02, 0.3, (-0.1, 0.6)), (0.02, 0.6, (-0.7, 0.2)))
    PyFeedForwardNet.fromlist((inputsize, hidden, output))


def test_tolist_doesnt_crash():
    net = PyFeedForwardNet.fromfuncs((2, 2, 1), 0.02)
    net.tolist()


def _nested_collections_equal(collection1, collection2):
    ''' Check if two nested collections are equal, ignoring collection
        types.
    '''
    first_iter = hasattr(collection1, '__iter__')
    second_iter = hasattr(collection2, '__iter__')

    if (first_iter != second_iter):
        return False

    if (first_iter):
        for e1, e2 in zip(collection1, collection2):
            if not _nested_collections_equal(e1, e2):
                return False
    else:
        if collection1 != collection2:
            return False

    return True


def test_listroundtrip():
    inputsize = 2
    hidden = ((0.02, -0.5, (0.5, -0.3)), (0.02, 0.3, (0.2, 0.9)))
    output = ((0.02, 0.3, (-0.1, 0.6)), (0.02, 0.6, (-0.7, 0.2)))
    inlist = (inputsize, hidden, output)
    net = PyFeedForwardNet.fromlist(inlist)
    outlist = net.tolist()

    assert(_nested_collections_equal(inlist, outlist))
