import logging
from pathlib import Path

from hylfm.utils.tracer import add_paths_to_plots, trace_and_plot

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    paths_for_tags = {}
    rois = {}

    for tag in ["09_3__2020-03-09_06.43.40__SinglePlane_-330"]:
        assert tag not in paths_for_tags
        paths_for_tags[tag] = {
            "lr_slice": Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.{tag}/run000/ds0-0"
            ),
            "ls_slice": Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.{tag}/run000/ds0-0"
            ),
            "pred": Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/f4/z_out49/f4_b2_only09_3/20-05-30_10-41-55/v1_checkpoint_82000_MSSSIM=0.8523718668864324/brain.{tag}/run000/ds0-0/"
            ),
        }

    assert "09_3__2020-03-09_06.43.40__SinglePlane_-340" not in paths_for_tags
    paths_for_tags["09_3__2020-03-09_06.43.40__SinglePlane_-340"] = {
        "lr_slice": Path(
            "/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.09_3__2020-03-09_06.43.40__SinglePlane_-340/run000/ds0-0"
        ),
        "ls_slice": Path(
            "/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.09_3__2020-03-09_06.43.40__SinglePlane_-340/run000/ds0-0"
        ),
        "pred": Path("missing"),
    }

    for tag in [
        "09_3__2020-03-09_06.43.40__SinglePlane_-320",
        "11_2__2020-03-11_10.25.41__SinglePlane_-305",
        "11_2__2020-03-11_08.12.13__SinglePlane_-310",
    ]:
        assert tag not in paths_for_tags
        paths_for_tags[tag] = {
            "ls_slice": f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/ls_f4/20-06-23_14-20-50/brain.{tag}/run000/ds0-0"
        }
        if tag.startswith("11_2"):
            rois[tag] = (slice(25, 225), slice(55, 305))

    for tag in [
        "11_2__2020-03-11_06.53.14__SinglePlane_-330",
        "11_2__2020-03-11_07.30.39__SinglePlane_-310",
        "11_2__2020-03-11_07.30.39__SinglePlane_-320",
        "11_2__2020-03-11_10.13.20__SinglePlane_-290",
        "11_2__2020-03-11_10.17.34__SinglePlane_-280",
        "11_2__2020-03-11_10.17.34__SinglePlane_-330",
        "11_2__2020-03-11_10.21.14__SinglePlane_-295",
        "11_2__2020-03-11_10.21.14__SinglePlane_-305",
        "11_2__2020-03-11_10.25.41__SinglePlane_-295",
        "11_2__2020-03-11_10.25.41__SinglePlane_-340",
    ]:
        assert tag not in paths_for_tags
        paths_for_tags[tag] = {
            name: Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-06-12_22-07-43/brain.{tag}/run000/ds0-0"
            )
            for name in ["ls_slice", "lr_slice"]
        }
        paths_for_tags[tag]["pred"] = Path(
            f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/f4/z_out49/f4_b2_only11_2/20-06-06_17-59-42/v1_checkpoint_29500_MS_SSIM=0.8786535175641378/brain.{tag}/run000/ds0-0"
        )
        rois[tag] = (slice(25, 225), slice(55, 305))

    # overwrite pred
    for tag in [
        "09_3__2020-03-09_06.43.40__SinglePlane_-320",
        "09_3__2020-03-09_06.43.40__SinglePlane_-330",
        "09_3__2020-03-09_06.43.40__SinglePlane_-340",
        "11_2__2020-03-11_08.12.13__SinglePlane_-310",
        "11_2__2020-03-11_06.53.14__SinglePlane_-330",
        "11_2__2020-03-11_10.25.41__SinglePlane_-305",
        "11_2__2020-03-11_07.30.39__SinglePlane_-310",
        "11_2__2020-03-11_07.30.39__SinglePlane_-320",
        "11_2__2020-03-11_10.13.20__SinglePlane_-290",
        "11_2__2020-03-11_10.17.34__SinglePlane_-280",
        "11_2__2020-03-11_10.17.34__SinglePlane_-330",
        "11_2__2020-03-11_10.21.14__SinglePlane_-295",
        "11_2__2020-03-11_10.21.14__SinglePlane_-305",
        "11_2__2020-03-11_10.25.41__SinglePlane_-295",
        "11_2__2020-03-11_10.25.41__SinglePlane_-340",
    ]:
        paths_for_tags[tag]["pred"] = Path(
            f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/f4/brain2/z_out49/f4_b2_bm2_with11_2_noSP/20-06-18_16-13-38/train2/v1_checkpoint_2000_MS_SSIM=0.6581869482994079/brain.{tag}/run000/ds0-0"
        )

    for i, (tag, time_range, manual_peaks) in enumerate(
        [
            # ("09_3__2020-03-09_06.43.40__SinglePlane_-320", (20, 600), False),
            # ("09_3__2020-03-09_06.43.40__SinglePlane_-320", (620, None), False),
            # ("09_3__2020-03-09_06.43.40__SinglePlane_-340", (20, 600), False),
            # ("09_3__2020-03-09_06.43.40__SinglePlane_-340", (620, None), False),
            # ("09_3__2020-03-09_06.43.40__SinglePlane_-330", (20, 600), False),
            # ("09_3__2020-03-09_06.43.40__SinglePlane_-330", (680, None), False),
            # ("11_2__2020-03-11_06.53.14__SinglePlane_-330", (20, None), False),
            # ("11_2__2020-03-11_06.53.14__SinglePlane_-330", (20, None), False),
            ("11_2__2020-03-11_07.30.39__SinglePlane_-310", (20, None), True),
            # ("11_2__2020-03-11_07.30.39__SinglePlane_-320", (20, None), False),
            # ("11_2__2020-03-11_10.13.20__SinglePlane_-290", (10, 545), False),
            # ("11_2__2020-03-11_10.17.34__SinglePlane_-280", (70, 520), False),
            # ("11_2__2020-03-11_10.17.34__SinglePlane_-330", (20, None), False),
            # ("11_2__2020-03-11_10.21.14__SinglePlane_-295", (20, None), False),
            # ("11_2__2020-03-11_10.21.14__SinglePlane_-305", (20, None), False),
            # ("11_2__2020-03-11_10.25.41__SinglePlane_-295", (20, None), False),
            # ("11_2__2020-03-11_10.25.41__SinglePlane_-340", (20, None), False),
        ]
    ):
        paths = paths_for_tags[tag]
        # paths = paths_09_3_a[330]
        output_path = Path(f"/g/kreshuk/LF_computed/lnet/traces/0707/{tag}")
        # output_path = Path(f"C:/repos/lnet/traces_max_r3/{tag}")
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
