import logging
from typing import List, Tuple, Union, Sequence

import numpy
from skimage.feature import blob_dog

# import matplotlib.pyplot as plt
from tifffile import imread

logger = logging.getLogger(__name__)


def get_bead_pos(
    img: numpy.ndarray,
    *,
    min_sigma: Union[float, Sequence[float]],
    max_sigma: Union[float, Sequence[float]],
    sigma_ratio: float,
    threshold: float,
    overlap: float,
    exclude_border: Union[Tuple[int, ...], int, bool]
) -> List[numpy.ndarray]:
    assert len(img.shape) == 5, img.shape
    assert img.shape[1] == 1, "grey expected_scale only"
    return [
        blob_dog(
            bimg,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            sigma_ratio=sigma_ratio,
            threshold=threshold,
            overlap=overlap,
            exclude_border=exclude_border,
        )[:, :3]
        for bimg in img[:, 0]
    ]


if __name__ == "__main__":
    from hylfm.detect_beads import plot_img_projections_with_beads, plot_img_projections
    import matplotlib.pyplot as plt

    tgt = imread("/g/kreshuk/LF_computed/lnet/logs/beads/01highc/20-04-21_11-41-43/test/output/0/pred.tif")[None, ...]
    print(tgt.min(), tgt.max(), tgt.shape)
    scaling = (2.0, 0.7, 0.7)

    min_sigma = 1.0
    max_sigma = 6.0
    sigma_ratio = 3.0
    threshold = 0.05
    overlap = 0.5
    exclude_border = False

    min_sigma = [min_sigma / s for s in scaling]
    max_sigma = [max_sigma / s for s in scaling]

    bead_pos_tgt = get_bead_pos(
        tgt,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        sigma_ratio=sigma_ratio,
        threshold=threshold,
        overlap=overlap,
        exclude_border=exclude_border,
    )
    bead_pos_tgt = bead_pos_tgt[0]
    print(bead_pos_tgt.shape)
    gaussians = bead_pos_tgt[:, 3:]
    print(gaussians)
    print(len(gaussians))
    bead_pos_tgt = bead_pos_tgt[:, :3]
    # plot_bead_projections(bead_pos_tgt[None, ...])
    plot_img_projections(tgt)
    plt.show()
    plot_img_projections_with_beads(tgt, [bead_pos_tgt])
    plt.show()
