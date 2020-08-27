import logging
from pathlib import Path

from hylfm.utils.tracer import add_paths_to_plots, trace_and_plot

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    paths_for_tags = {}
    rois = {}

    for tag in [
        "09_3__2020-03-09_06.43.40__SinglePlane_-320",
        "11_2__2020-03-11_10.25.41__SinglePlane_-305",
        "11_2__2020-03-11_08.12.13__SinglePlane_-310",
    ]:
        assert tag not in paths_for_tags
        paths_for_tags[tag] = {
            "ls_slice": f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/ls_f4/20-06-23_14-20-50/brain.{tag}/run000/ds0-0"
        }
        # if tag.startswith("11_2"):
        #     rois[tag] = (slice(25, 225), slice(55, 305))

    # add pred
    for tag in [
    ]:
        paths_for_tags[tag]["pred"] = Path(
            f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/f4/brain2/z_out49/f4_b2_bm2_with11_2_noSP/20-06-18_16-13-38/train2/v1_checkpoint_2000_MS_SSIM=0.6581869482994079/brain.{tag}/run000/ds0-0"
        )

    for i, (tag, time_range, manual_peaks) in enumerate(
        [
            ("11_2__2020-03-11_07.30.39__SinglePlane_-310", (20, None), False),
        ]
    ):
        paths = paths_for_tags[tag]
        output_path = Path(f"/g/kreshuk/LF_computed/lnet/traces/09_1__07-24/{tag}")
        tgt = "ls_slice"
        plots = add_paths_to_plots(
            [
                # {"lr_slice": {"smooth": [(None, None)]}, "pred": {"smooth": [(None, None)]}},
                # {
                #     "lr_slice": {"smooth": [(("flat", 11), ("flat", 11))]},
                #     "pred": {"smooth": [(("flat", 11), ("flat", 11))]},
                # },
                # # {"lr_slice": {"smooth": [(None, ("flat", 3))]}, "pred": {"smooth": [(None, ("flat", 3))]}},
                # # {"lr_slice": {"smooth": [(None, None)]}},
                # # {"lr_slice": {"smooth": [(("flat", 11), ("flat", 11))]}},
                # # {"lr_slice": {"smooth": [(None, ("flat", 3))]}},
                # {
                #     "lr_slice": {
                #         "smooth": [
                #             (
                #                 ("savgol_filter", {"window_length": 11, "polyorder": 3}),
                #                 ("savgol_filter", {"window_length": 11, "polyorder": 3}),
                #             )
                #         ]
                #     },
                #     "pred": {
                #         "smooth": [
                #             (
                #                 ("savgol_filter", {"window_length": 11, "polyorder": 3}),
                #                 ("savgol_filter", {"window_length": 11, "polyorder": 3}),
                #             )
                #         ]
                #     },
                # },
                {"pred": {"smooth": [(None, None)]}},
                {"pred": {"smooth": [(("flat", 11), ("flat", 11))]}},
                {
                    "pred": {
                        "smooth": [
                            (
                                ("savgol_filter", {"window_length": 11, "polyorder": 3}),
                                ("savgol_filter", {"window_length": 11, "polyorder": 3}),
                            )
                        ]
                    }
                },
            ],
            paths=paths,
        )
        try:
            peaks, traces, correlations, figs, motion = trace_and_plot(
                tgt_path=paths[tgt],
                tgt=tgt,
                roi=rois.get(tag, (slice(0, 9999), slice(0, 9999))),
                plots=plots,
                output_path=output_path,
                nr_traces=32 * 4,
                background_threshold=0.1,
                overwrite_existing_files=False,
                smooth_diff_sigma=1.3,
                peak_threshold_abs=0.05,
                reduce_peak_area="max",
                plot_peaks=False,
                compute_peaks_on="max",  # std, diff, min, max, mean
                peaks_min_dist=3,
                trace_radius=3,
                # time_range=(0, 600),
                # time_range=(0, 50),
                # time_range=(660, 1200),
                time_range=time_range,
                # compensate_motion={"compensate_ref": tgt, "method": "ES", "mbSize": 50, "p": 4},
                # compensate_motion={
                #     "of_peaks": True,
                #     "only_on_tgt": True,
                #     "method": "home_brewed",
                #     "n_radii": 2,
                #     "accumulate_relative_motion": "decaying cumsum",
                #     "motion_decay": 0.8,
                #     # "accumulate_relative_motion": "cumsum",
                #     # "motion_decay": None,
                #     "upsample_xy": 2,
                #     "upsample_t": 1,
                #     "presmooth_sigma": None,
                # },
                # "ES" --> exhaustive search
                # "3SS" --> 3-step search
                # "N3SS" --> "new" 3-step search [#f1]_
                # "SE3SS" --> Simple and Efficient 3SS [#f2]_
                # "4SS" --> 4-step search [#f3]_
                # "ARPS" --> Adaptive Rood Pattern search [#f4]_
                # "DS" --> Diamond search [#f5]_
                tag=tag,
                peak_path=Path(f"/g/kreshuk/LF_computed/lnet/manual_traces/2d_coordinates/{tag}.yml")
                if manual_peaks
                else None,
            )
        except Exception as e:
            logger.error(e, exc_info=True)