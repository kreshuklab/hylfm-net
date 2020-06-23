from pathlib import Path


def pad_file_names(folder: Path, file_glob: str, pad_with: str = "0"):
    assert folder.exists(), folder
    assert file_glob.count("*") == 1
    assert file_glob[-1] != "*"
    expanded_slice = slice(file_glob.find("*"), -file_glob[::-1].find("*"))
    print("asterisk position", expanded_slice)
    max_file_name_len = max([len(file_path.name) for file_path in folder.glob(file_glob)])
    pad_to = max_file_name_len - len(file_glob) + 1
    for file_path in folder.glob(file_glob):
        expanded = file_path.name[expanded_slice]
        file_path_padded = folder / file_glob.replace("*", f"{expanded:f'{pad_with}{pad_to}'}")
        if file_path_padded.exists():
            raise FileExistsError(f"Cannot pad {file_path} as {file_path_padded} already exists.")

        file_path.rename(file_path_padded)


if __name__ == "__main__":
    pad_file_names(Path("/g/kreshuk/LF_computed/LenseLeNet_Microscope/dualview_060918/Rectified_LC"), "Cam_Left_*.tif")
