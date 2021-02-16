import numpy


class DuplicateLogFilter:
    def __init__(self):
        self.msgs = set()

    def __call__(self, record) -> int:
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return int(rv)


def get_max_projection_img(img: numpy.ndarray):
    assert len(img.shape) == 5, img.shape
    b, c, z, y, x = img.shape
    zmax = img.max(2)  # bcyx
    ymax = img.max(3)  # bczx
    xmax = img.max(4)  # bczy
    free = numpy.zeros((b, c, z, z), dtype=zmax.dtype)

    p1 = numpy.concatenate([zmax, ymax], axis=-2)
    p2 = numpy.concatenate([xmax.transpose(0, 1, 3, 2), free], axis=-2)
    return numpy.concatenate([p1, p2], axis=-1)


if __name__ == "__main__":
    print(get_max_projection_img(numpy.zeros((2, 1, 3, 4, 5))).shape)
