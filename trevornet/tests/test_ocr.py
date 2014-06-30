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


def test_resize_lengths():
    result = ocr.resize(2, 3, _image)
    return len(result) == 3 and len(result[0]) == 2


_image2 = [[0, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0]]


def test_get_occupied_region():
    region = ocr.get_occupied_region(_image2)
    return region == (1, 0, 4, 5)


def test_crop():
    result = ocr.crop((1, 0, 4, 5), _image2)
    target = [[1, 0, 1, 0],
              [0, 0, 0, 0],
              [0, 1, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 1]]
    return result == target


def test_upscale_to_aspect():
    result = ocr.upscale_to_aspect((3, 1), 1/2)
    return result == (3, 6)


def test_upscale_to_aspect2():
    result = ocr.upscale_to_aspect((1, 3), 2)
    return result == (6, 3)


def test_upscale_to_aspect3():
    result = ocr.upscale_to_aspect((1, 3), 1/3)
    return result == (1, 3)
