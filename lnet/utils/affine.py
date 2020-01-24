import numpy
import torch


def scipy_form2torch_form_2d(scipy_form, img_shape):
    assert all(zero == 0 for zero in scipy_form[2, :2])
    assert scipy_form[2, 2] == 1
    transposed_scipy = numpy.eye(3, dtype=scipy_form.dtype)
    transposed_scipy[:2, :2] = scipy_form[:2, :2].T
    offset = scipy_form[:2, 2][::-1]
    transposed_scipy[:2, 2] = offset

    h, w = img_shape

    theta = numpy.zeros((2, 3))
    # wolfram alpha input: {{2/w, 0, -1}, {0, 2/h, -1}, {0, 0, 1}} * {{a, b, c}, {d, e, f}, {0, 0, 1}} * {{2/w, 0, -1}, {0, 2/h, -1}, {0, 0, 1}}^-1
    theta[0, 0] = transposed_scipy[0, 0]
    theta[0, 1] = transposed_scipy[0, 1] * h / w
    theta[0, 2] = transposed_scipy[0, 2] * 2 / w + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = transposed_scipy[1, 0] * w / h
    theta[1, 1] = transposed_scipy[1, 1]
    theta[1, 2] = transposed_scipy[1, 2] * 2 / h + theta[1, 0] + theta[1, 1] - 1
    return torch.from_numpy(theta[None, ...])


def inv_scipy_form2torch_form_2d(inv_scipy_form, ipt_shape, out_shape):
    """like scipy_form2torch_form_2d, but allows for ipt_shape != out_shape. Takes the inverse scipy form (and inverts it after scaling)"""
    assert all(zero == 0 for zero in inv_scipy_form[2, :2])
    assert inv_scipy_form[2, 2] == 1
    assert len(ipt_shape) == len(out_shape) == 2
    scaling = [si / so for si, so in zip(ipt_shape, out_shape)] + [1.0]
    scaled_scipy = numpy.linalg.inv(numpy.diag(scaling).dot(inv_scipy_form))

    transposed_scipy = numpy.eye(3, dtype=scaled_scipy.dtype)
    transposed_scipy[:2, :2] = scaled_scipy[:2, :2].T
    offset = scaled_scipy[:2, 2][::-1]
    transposed_scipy[:2, 2] = offset

    h, w = ipt_shape

    theta = numpy.zeros((2, 3))
    # wolfram alpha input: {{2/w, 0, -1}, {0, 2/h, -1}, {0, 0, 1}} * {{a, b, c}, {d, e, f}, {0, 0, 1}} * {{2/w, 0, -1}, {0, 2/h, -1}, {0, 0, 1}}^-1
    theta[0, 0] = transposed_scipy[0, 0]
    theta[0, 1] = transposed_scipy[0, 1] * h / w
    theta[0, 2] = transposed_scipy[0, 2] * 2 / w + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = transposed_scipy[1, 0] * w / h
    theta[1, 1] = transposed_scipy[1, 1]
    theta[1, 2] = transposed_scipy[1, 2] * 2 / h + theta[1, 0] + theta[1, 1] - 1
    return torch.from_numpy(theta[None, ...])


def scipy_form2torch_form_3d(scipy_form, img_shape):
    transposed_scipy = numpy.eye(4, dtype=scipy_form.dtype)
    map_pos = {
        (0, 0): (2, 2),
        (0, 1): (2, 1),
        (0, 2): (2, 0),
        (0, 3): (2, 3),
        (1, 0): (1, 2),
        (1, 1): (1, 1),
        (1, 2): (1, 0),
        (1, 3): (1, 3),
        (2, 0): (0, 2),
        (2, 1): (0, 1),
        (2, 2): (0, 0),
        (2, 3): (0, 3),
    }
    for spos, tpos in map_pos.items():
        transposed_scipy[tpos] = scipy_form[spos]

    t, h, w = img_shape

    # extended 2d case with depth as 't' and additional parameters klmnop
    # wolfram alpha input: {{2/w, 0, 0, -1}, {0, 2/h, 0, -1}, {0, 0, 2/t, -1}, {0, 0, 0, 1}} * {{a, b, k, c}, {d, e, l, f}, {m, n, o, p}, {0, 0, 0, 1}} * {{2/w, 0, 0, -1}, {0, 2/h, 0, -1},{0, 0, 2/t, -1}, {0, 0, 0, 1}}^-1
    a = transposed_scipy[0, 0]
    b = transposed_scipy[0, 1]
    k = transposed_scipy[0, 2]  # 3d
    c = transposed_scipy[0, 3]
    d = transposed_scipy[1, 0]
    e = transposed_scipy[1, 1]
    l = transposed_scipy[1, 2]  # 3d
    f = transposed_scipy[1, 3]

    m = transposed_scipy[2, 0]  # 3d
    n = transposed_scipy[2, 1]  # 3d
    o = transposed_scipy[2, 2]  # 3d
    p = transposed_scipy[2, 3]  # 3d

    theta = numpy.array(
        [
            [a, b * h / w, k * t / w, a + 2 * c / w + b * h / w + k * t / w - 1],
            [d * w / h, e, l * t / h, e + l * t / h + d * w / h + 2 * f / h - 1],
            [m * w / t, h * n / t, o, h * n / t + o + m * w / t + 2 * p / t - 1],
            # [0, 0, 0, 1],
        ]
    )
    return torch.from_numpy(theta[None, ...])


def inv_scipy_form2torch_form_3d(inv_scipy_form, ipt_shape, out_shape):
    """like scipy_form2torch_form_3d, but allows for ipt_shape != out_shape. Takes the inverse scipy form (and inverts it after scaling)"""
    assert all(zero == 0 for zero in inv_scipy_form[3, :3])
    assert inv_scipy_form[3, 3] == 1
    assert len(ipt_shape) == len(out_shape) == 3
    scaling = [si / so for si, so in zip(ipt_shape, out_shape)] + [1.0]
    scaled_scipy = numpy.linalg.inv(numpy.diag(scaling).dot(inv_scipy_form))

    transposed_scipy = numpy.eye(4, dtype=scaled_scipy.dtype)
    map_pos = {
        (0, 0): (2, 2),
        (0, 1): (2, 1),
        (0, 2): (2, 0),
        (0, 3): (2, 3),
        (1, 0): (1, 2),
        (1, 1): (1, 1),
        (1, 2): (1, 0),
        (1, 3): (1, 3),
        (2, 0): (0, 2),
        (2, 1): (0, 1),
        (2, 2): (0, 0),
        (2, 3): (0, 3),
    }
    for spos, tpos in map_pos.items():
        transposed_scipy[tpos] = scaled_scipy[spos]

    t, h, w = ipt_shape

    # extended 2d case with depth as 't' and additional parameters klmnop
    # wolfram alpha input: {{2/w, 0, 0, -1}, {0, 2/h, 0, -1}, {0, 0, 2/t, -1}, {0, 0, 0, 1}} * {{a, b, k, c}, {d, e, l, f}, {m, n, o, p}, {0, 0, 0, 1}} * {{2/w, 0, 0, -1}, {0, 2/h, 0, -1},{0, 0, 2/t, -1}, {0, 0, 0, 1}}^-1
    a = transposed_scipy[0, 0]
    b = transposed_scipy[0, 1]
    k = transposed_scipy[0, 2]  # 3d
    c = transposed_scipy[0, 3]
    d = transposed_scipy[1, 0]
    e = transposed_scipy[1, 1]
    l = transposed_scipy[1, 2]  # 3d
    f = transposed_scipy[1, 3]

    m = transposed_scipy[2, 0]  # 3d
    n = transposed_scipy[2, 1]  # 3d
    o = transposed_scipy[2, 2]  # 3d
    p = transposed_scipy[2, 3]  # 3d

    theta = numpy.array(
        [
            [a, b * h / w, k * t / w, a + 2 * c / w + b * h / w + k * t / w - 1],
            [d * w / h, e, l * t / h, e + l * t / h + d * w / h + 2 * f / h - 1],
            [m * w / t, h * n / t, o, h * n / t + o + m * w / t + 2 * p / t - 1],
            # [0, 0, 0, 1],
        ]
    )
    return torch.from_numpy(theta[None, ...])
