from pathlib import Path


def postprocess_beads_f8_01highc(restored):
    assert len(restored.shape) == 3
    assert restored.shape == (51, 376, 576)
    return restored


def postprocess_heart_static(restored):
    assert restored.shape in [(49, 244, 284), (49, 224, 348)], restored.shape
    return restored


def postprocess_heart_dynamic(restored):
    assert len(restored.shape) == 3
    assert restored.shape == (49, 244, 284), restored.shape
    return restored


def get_setup(*, subpath: str):
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
    else:
        raise NotImplementedError(subpath)

    model_path = root / "models" / model_subpath / model_name
    (test_data_path / model_name).mkdir(parents=True, exist_ok=True)
    return {
        "model_name": model_name,
        "model_path": model_path,
        "train_data_path": train_data_path,
        "test_data_path": test_data_path,
        "gt_name": gt_name,
        "postprocess": postprocess,
        "tensor_name": "care",
    }
