from hylfm.datasets import TensorInfo


def get_tensor_info(tag: str, name: str, meta: dict):
    your_tensor_info = TensorInfo(
        name="lf",
        # root="my_data_root", #optional root name for location (as specified in hylfm._settings.local.py)
        location="/path/to/data/with/glob_expr/in_*folders/and/or/files_*.tif",  # or .h5
        # insert_singleton_axes_at: hylfm expects an explicit batch dimension as first axis
        insert_singleton_axes_at=[0, 0],  # e.g. [0, 0] for image as xy: xy -> bcxy
        meta=meta,
    )
    raise NotImplementedError(f"tag: {tag}, name: {name}, meta: {meta}")
    return your_tensor_info
