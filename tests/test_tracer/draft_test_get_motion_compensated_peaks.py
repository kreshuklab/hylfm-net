import numpy

from hylfm.utils.tracer import get_motion_compensated_peaks


def test_get_motion_compensated_peaks(output_path):
    H, W = 50, 50
    numpy.random.seed(0)
    frame = numpy.random.random_sample((H, W))
    abs_motion = [(0, 0), (1, 0), (1, 1), (1, 1)]
    rois = [(slice(5 + a, H - 5 + a), slice(5 + b, W - 5 + b)) for a, b in abs_motion]
    tensor = numpy.stack([frame[roi] for roi in rois])
    peaks = numpy.array([[20, 20, 3]])
    # “ES” –> exhaustive search
    # “3SS” –> 3-step search
    # “N3SS” –> “new” 3-step search [1]
    # “SE3SS” –> Simple and Efficient 3SS [2]
    # “4SS” –> 4-step search [3]
    # “ARPS” –> Adaptive Rood Pattern search [4]
    # “DS” –> Diamond search [5]
    compensated_peaks = get_motion_compensated_peaks(
        tensor=tensor,
        peaks=peaks,
        output_path=output_path,
        method="ES",
        n_radii=2,
        motion_decay=0.9,
        accumulate_relative_motion="decaying cumsum",
    )
    # print("compensated_peaks.shape", [cp.shape for cp in compensated_peaks])
    # print("compensated_peaks")
    print(compensated_peaks)

    assert False


def test_get_motion_home_brewed_compensated_peaks(output_path):
    H, W = 50, 50
    numpy.random.seed(0)
    frame = numpy.random.random_sample((H, W))
    abs_motion = [(0, 0), (1, 0), (1, 1), (1, 1)]
    rois = [(slice(5 + a, H - 5 + a), slice(5 + b, W - 5 + b)) for a, b in abs_motion]
    tensor = numpy.stack([frame[roi] for roi in rois])
    peaks = numpy.array([[20, 20, 3]])
    # “ES” –> exhaustive search
    # “3SS” –> 3-step search
    # “N3SS” –> “new” 3-step search [1]
    # “SE3SS” –> Simple and Efficient 3SS [2]
    # “4SS” –> 4-step search [3]
    # “ARPS” –> Adaptive Rood Pattern search [4]
    # “DS” –> Diamond search [5]
    compensated_peaks = get_motion_compensated_peaks(
        tensor=tensor,
        peaks=peaks,
        output_path=output_path,
        method="home_brewed",
        n_radii=2,
        accumulate_relative_motion="cumsum",
    )
    # print("compensated_peaks.shape", [cp.shape for cp in compensated_peaks])
    # print("compensated_peaks")
    print(compensated_peaks)

    assert False
