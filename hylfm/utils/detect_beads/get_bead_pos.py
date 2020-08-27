import logging
from typing import List, Tuple, Union, Sequence

import numpy
from skimage.feature import blob_dog

# import matplotlib.pyplot as plt
from tifffile import imread

logger = logging.getLogger(__name__)

def get_bead_pos(img: numpy.ndarray, *, min_sigma: Union[float, Sequence[float]], max_sigma: Union[float, Sequence[float]], sigma_ratio: float, threshold: float, overlap: float, exclude_border: Union[Tuple[int, ...], int, bool]) -> List[numpy.ndarray]:
    assert len(img.shape) == 5, img.shape
    assert img.shape[1] == 1, "grey scale only"
# home_brew:
#     smooth = ndimage.gaussian_filter(img, [0, 0, 1, 2 * xy_factor, 2 * xy_factor], mode="constant", cval=img.mean())
#     smooth2 = ndimage.gaussian_filter(img, [0, 0, 3, 7 * xy_factor, 7 * xy_factor], mode="constant", cval=img.mean())
#     gdiff = numpy.array(smooth, dtype=numpy.float64) - smooth2
#
#     # from hylfm.utils.plotting import turbo_colormap
#     # fig = plt.figure()
#     # plt.imshow(smooth[0, 0].max(0), cmap=turbo_colormap)
#     # plt.colorbar()
#     # plt.title('smooth')
#     # plt.show()
#     #
#     # fig = plt.figure()
#     # plt.imshow(smooth2[0, 0].max(0), cmap=turbo_colormap)
#     # plt.colorbar()
#     # plt.title('smooth2')
#     # plt.show()
#     #
#     # fig = plt.figure()
#     # plt.imshow(gdiff[0, 0].max(0), cmap=turbo_colormap)
#     # plt.colorbar()
#     # plt.title('gdiff')
#     # plt.show()
#
#     thresh = gdiff.mean() + gdiff.std() * 3
#     logger.debug("thresh", thresh)
#     local_maxi = numpy.stack(
#         [
#             peak_local_max(im[0], indices=False, min_distance=3, threshold_abs=thresh, exclude_border=True)
#             for im in gdiff
#         ]
#     )
#     markers = [ndimage.label(lm, structure=ndimage.generate_binary_structure(3, 2))[0] for lm in local_maxi > 0]
#     bead_pos = [
#         numpy.asarray(ndimage.measurements.center_of_mass(im[0], labels=m, index=numpy.arange(1, m.max() + 1)))
#         for im, m in zip(img, markers)
#     ]
#     if bead_pos:
#         logger.debug("found %s beads", [len(bp) for bp in bead_pos])
#         # inbetween = numpy.array(bead_pos[0, :, 0])
#         # inbetween = inbetween[10 < inbetween]
#         # inbetween = inbetween[inbetween < 40]
#         # logger.debug("found %s beads, %s between between z=10 and z=40", bead_pos.shape[1], inbetween.shape[0])
#         #
#         # plt.figure()
#         # plt.scatter(bead_pos[0, :, 2], bead_pos[0, :, 1], marker="2")
#         # plt.imshow(markers[0].max(0), cmap=mycm)
#         # plt.title('markers')
#         # plt.show()
#     else:
#         logger.debug("no beads found")

    return [blob_dog(bimg, min_sigma=min_sigma, max_sigma=max_sigma, sigma_ratio=sigma_ratio, threshold=threshold, overlap=overlap, exclude_border=exclude_border)[:, :3] for bimg in img[:, 0]]



if __name__ == "__main__":
    from hylfm.utils.detect_beads.plot_bead_proj import plot_bead_projections
    from hylfm.utils.detect_beads.plot_img_proj import plot_img_projections_with_beads, plot_img_projections
    import matplotlib.pyplot as plt

    tgt = imread("/g/kreshuk/LF_computed/lnet/logs/beads/01highc/20-04-21_11-41-43/test/output/0/pred.tif")[None, ...]
    print(tgt.min(), tgt.max(), tgt.shape)
    scaling = (2., 0.7, 0.7)

    min_sigma=1.0
    max_sigma=6.0
    sigma_ratio=3.0
    threshold=.05
    overlap=0.5
    exclude_border=False

    min_sigma = [min_sigma / s for s in scaling]
    max_sigma = [max_sigma / s for s in scaling]

    bead_pos_tgt = get_bead_pos(tgt, min_sigma=min_sigma, max_sigma=max_sigma, sigma_ratio=sigma_ratio, threshold=threshold, overlap=overlap, exclude_border=exclude_border)
    bead_pos_tgt = bead_pos_tgt[0]
    print(bead_pos_tgt.shape)
    gaussians = bead_pos_tgt[:, 3:]
    print(gaussians)
    print(len(gaussians))
    bead_pos_tgt= bead_pos_tgt[:, :3]
    # plot_bead_projections(bead_pos_tgt[None, ...])
    plot_img_projections(tgt)
    plt.show()
    plot_img_projections_with_beads(tgt, [bead_pos_tgt])
    plt.show()
