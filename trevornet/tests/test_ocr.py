from trevornet import ocr


_image = [[0, 2, 0],
          [0, 2, 0],
          [2, 5, 2]]


def test_read_point_bilinear_integral():
    result = ocr.read_point_bilinear(1, 1, _image)
    return result == 2


def test_read_point_bilinear_between():
    result = ocr.read_point_bilinear(0.5, 0.5, _image)
    target = 1
    tolerance = 0.01
    return abs(target - result) < tolerance


def test_read_point_bilinear_between2():
    result = ocr.read_point_bilinear(0.5, 1.5, _image)
    target = 2
    tolerance = 0.01
    return abs(target - result) < tolerance
