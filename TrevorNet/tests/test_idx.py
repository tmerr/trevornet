from .. import idx
import os

def test__find_depth():
    yield check__find_depth, 9, 0
    yield check__find_depth, [1, 2], 1
    yield check__find_depth, [[1, 2], [3, 6, 2]], 2
    yield check__find_depth, [[[1,2], [2]]], 3

def check__find_depth(lst, i):
    assert idx._find_dimensions(lst) == i

# these two are equivalent according to the format on http://yann.lecun.com/exdb/mnist/
_somelist = [[1, 2], [3, 4]]
_somebytes = '\x00\x00\x0C\x02' + '\x01\x02\x03\x04'

_testfolder = os.path.dirname(os.path.realpath(__file__))
_somepath = os.path.join(_testfolder, 'test_idx_file')

def test_list_to_idx():
    idx.list_to_idx(_somelist, _somepath, 'i')
    with open(_somepath, 'rb') as f:
        data = f.read() 
    os.remove(_somepath)

    assert data == _somebytes

def test_idx_to_list():
    with open(_somepath, 'wb') as f:
        f.write(_somebytes)
    lst = idx.idx_to_list(_somepath)
    os.remove(_somepath)

    assert lst == _somelist
