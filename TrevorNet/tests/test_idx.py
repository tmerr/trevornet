from trevornet import idx
import os

def test_find_dimensions():
    yield check_find_dimensions, 9, []
    yield check_find_dimensions, [1, 2], [2,]
    yield check_find_dimensions, [[1, 2, 3], [4, 5, 6]], [2, 3]
    yield check_find_dimensions, [[[1, 2], [2, 3]]], [1, 2, 2]

def check_find_dimensions(lst, i):
    assert idx.find_dimensions(lst) == i

# these two are equivalent according to the format on http://yann.lecun.com/exdb/mnist/
_somelist = [[1, 2], [3, 4]]
def _get_somebytes():
    header = b'\x00\x00\x0C\x02'
    dimensionsizes = b'\x00\x00\x00\x02' + b'\x00\x00\x00\x02'
    data = b'\x00\x00\x00\x01' + b'\x00\x00\x00\x02'
    data += b'\x00\x00\x00\x03' + b'\x00\x00\x00\x04'
    return header + dimensionsizes + data
_somebytes = _get_somebytes()

def test_list_to_idx():
    data = idx.list_to_idx(_somelist, 'i')
    assert data == _somebytes

def test_idx_to_list():
    lst = idx.idx_to_list(_somebytes)
    assert lst == _somelist
