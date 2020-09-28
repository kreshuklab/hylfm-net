from hylfm.datasets.online import OnlineTensorInfo


def get_tensor_info(tag: str, name: str, meta: dict):
    samples_per_dataset = 1
    insert_singleton_axes_at = [0, 0]
    if tag == "small_0":
        doi = "10.5281/zenodo.4019246"
        file_name = f"{tag}_{name}.zip"
        in_file_glob = "TP_*.tif"
    elif tag == "small_1":
        doi = "10.5281/zenodo.4020352"
        file_name = f"{tag}_{name}.zip"
        in_file_glob = "TP_*.tif"
    elif tag == "small_2":
        doi = "10.5281/zenodo.4020404"
        if name == "ls_reg":
            file_name = "small_2_ls_reg.h5"
            in_file_glob = "ls_reg"
            samples_per_dataset = 137
            insert_singleton_axes_at = []
        else:
            file_name = f"{tag}_{name}.zip"
            in_file_glob = "TP_*.tif"

    else:
        raise NotImplementedError(tag, name)

    info = OnlineTensorInfo(
        name=name,
        doi=doi,
        file_name=file_name,
        in_file_glob=in_file_glob,
        meta=meta,
        insert_singleton_axes_at=insert_singleton_axes_at,
        tag=f"{tag}_{name}",
        samples_per_dataset=samples_per_dataset,
    )
    info.download()
    info.extract()
    return info
