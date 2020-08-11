from pathlib import Path


def postprocess_beads_f4_01highc(restored):
    assert False
    assert len(restored.shape) == 3
    restored = restored[3:-2, 2:-2, :]
    assert restored.shape == (51, 180, 280)
    return restored


def postprocess_beads_f8_01highc(restored):
    assert len(restored.shape) == 3
    # restored = restored[3:-2, :, :]
    assert restored.shape == (51, 376, 576)
    return restored


def postprocess_heart_static(restored):
    # assert len(restored.shape) == 3
    # restored = restored[:, :, :]
    assert restored.shape in [(49, 244, 284), (49, 224, 348)], restored.shape
    return restored


def postprocess_heart_dynamic(restored):
    assert len(restored.shape) == 3
    #         restored =  restored[:, :, :]
    assert restored.shape == (49, 244, 284), restored.shape
    return restored


# def get_hylfm_setup(model_name: str):
#     # data_path = None
#     result_path = Path("/g/kreshuk/LF_computed/lnet/hylfm/results") / model_name
#     tensor_name = "hylfm"
#     gt = "ls_reg" if "beads" in model_name else "ls_slice" if "on_dynamic" in model_name else "ls_trf"
#     model_path = Path("/g/kreshuk/LF_computed/lnet")
#
#     if model_name == "dynamic_on_dynamic_heart":
#         gt_path = (
#             Path(
#                 "/g/kreshuk/LF_computed/lnet/logs/heart2/test_z_out49/lr_f4/heart_dynamic.2019-12-09_04.54.38/run000/ds0-0/"
#             )
#             / gt
#         )
#         model_name = "HyLFM-Net dyn"
#         subpath = "logs/heart2/test_z_out49/f4/z_out49/f4_b2/20-05-20_10-18-11/v1_checkpoint_MSSSIM=0.6722144321961836/heart_dynamic.2019-12-09_04.54.38/run000/ds0-0/pred"
#     elif model_name == "dynamic_on_static_heart":
#         model_name = "HyLFM-Net dyn"
#         subpath = "logs/heart3/test_z_out49/f4_pred4care/heart2/z_out49/f4_b2/20-05-20_10-18-11/v1_checkpoint_MSSSIM=0.6722144321961836/heart_static.2019-12-08_*/run000/ds0-0/pred"
#     elif model_name == "static_on_static_heart2":
#         model_name = "HyLFM-Net stat"
#         subpath = "logs/heart3/test_z_out49/f4/heart2/z_out49/static_f4_b2_with5_pois/20-06-13_20-26-33/train2/v1_checkpoint_498_MS_SSIM=0.9710696664723483/heart_static.2019-12-09_08.41.41/run000/ds0-0/pred"
#     elif model_name == "lr_on_static_heart":
#         model_name = "LFD"
#         subpath = "logs/heart3/test_z_out49/lr_f4/20-08-10_12-26-38/heart_static.2019-12-08_*/run000/ds0-0/lr"
#     elif model_name == "lr_on_static_heart2":
#         model_name = "LFD"
#         subpath = "logs/heart3/test_z_out49/lr_f4/20-08-10_21-24-49/heart_static.2019-12-09_09.52.38/run000/ds0-0/lr"
#     else:
#         raise NotImplementedError(model_name)
#
#     raw_result_path = model_path / subpath
#     return {
#         "model_name": model_name,
#         "subpath": subpath,
#         "model_path": model_path,
#         # "model_subpath": model_subpath,
#         "gt": gt,
#         # "data_path": data_path,
#         "result_path": result_path,
#         "raw_result_path": raw_result_path,
#         # "postprocess": postprocess,
#         # "ls_postprocess": ls_postprocess,
#         # "gt_path": gt_path,
#         "tensor_name": tensor_name,
#     }


def get_setup(*, subpath: str):
    # subpath = "/".join(model_name.split("/")[:-1])
    # model_name = model_name.split("/")[0]
    *subdirs, model_name = subpath.split("/")
    subpath = "/".join(subdirs)

    model_subpath = "care/" + subpath
    root = Path("/g/kreshuk/LF_computed/lnet/plain")
    gt_name = "ls_reg" if "beads" in subpath else "ls_slice" if "dynamic" in subpath else "ls_trf"
    train_data_path = Path("/scratch/beuttenm/lnet/plain/") / subpath / "train"
    test_data_path = root / subpath / "test"

    if subpath == "heart/static1":
        postprocess = postprocess_heart_static
    elif subpath == "heart/static2":
        postprocess = postprocess_heart_static
    elif subpath == "heart/dynamic1":
        postprocess = postprocess_heart_dynamic
    elif subpath == "beads/f8_01highc":
        postprocess = postprocess_beads_f8_01highc
        # gt_postprocess = postprocess
    # if subpath == "beads/f4_01highc":
    #     model_subpath = "care/" + model_subpath
    #     assert model_name in ["ep100", "ep400"]
    #     postprocess = postprocess_beads_f4_01highc
    #     gt_postprocess = postprocess
    # elif subpath == "beads/f8_01highc":
    #     model_subpath = "care/" + model_subpath
    #     assert model_name in ["v0_spe400_on_56x64x64", "v0_spe400_on_56x80x80", "v0_spe1000_on_56x80x80"]
    #     postprocess = postprocess_beads_f8_01highc
    #     gt_postprocess = postprocess
    # elif subpath == "heart/static4":
    #     model_subpath = "care/heart/static1"
    #     assert model_name in ["v0_on_48x88x88"], model_name
    #     postprocess = postprocess_heart_static
    #     gt_postprocess = postprocess
    # elif subpath == "heart/dynamic":
    #     raw_result_path = Path("/scratch/beuttenm/lnet/care/results") / subpath / model_name
    #     model_subpath = "care/heart/static"
    #     assert model_name in ["v0_spe1000_on_48x88x88"]
    #     postprocess = postprocess_heart_dynamic
    #     gt_postprocess = None
    #     gt_path = (
    #         Path(
    #             "/g/kreshuk/LF_computed/lnet/logs/heart2/test_z_out49/lr_f4/heart_dynamic.2019-12-09_04.54.38/run000/ds0-0/"
    #         )
    #         / gt
    #     )
    #     assert gt_path.exists(), gt_path.absolute()
    # elif subpath == "heart/static3":
    #     model_subpath = "care/" + model_subpath
    #     assert model_name in ["v0_spe1000_on_48x88x88"]
    #     postprocess = postprocess_heart_static
    #     gt_postprocess = postprocess
    else:
        raise NotImplementedError(subpath)


    model_path = root / "models" / model_subpath / model_name
    (test_data_path / model_name).mkdir(parents=True, exist_ok=True)
    return {
        "model_name": model_name,
        # "subpath": subpath,
        "model_path": model_path,
        # "model_subpath": model_subpath,
        # "ls": gt,  # deprecated
        "train_data_path": train_data_path,
        "test_data_path": test_data_path,
        # "result_data_path": result_data_path,
        "gt_name": gt_name,
        # "data_path": data_path,
        # "result_path": result_path,
        # "raw_result_path": raw_result_path,
        "postprocess": postprocess,
        # # "ls_postprocess": gt_postprocess,  # deprecated
        # "gt_postprocess": gt_postprocess,
        # "gt_path": gt_path,
        "tensor_name": "care",
    }
