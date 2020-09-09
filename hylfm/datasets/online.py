import shutil
import zipfile

import requests
from tqdm import tqdm

from hylfm import settings
from .base import TensorInfo


class OnlineTensorInfo(TensorInfo):
    def __init__(self, doi: str, archive_name_with_suffix: str, glob_expression: str, **super_kwargs):
        self.doi = doi
        self.archive_name_with_suffix = archive_name_with_suffix
        self.archive_name = archive_name_with_suffix.split(".")[0]
        super().__init__(root=settings.cache_dir / doi / self.archive_name, location=glob_expression, **super_kwargs)
        self.download_file_path = settings.download_dir / doi / self.archive_name_with_suffix

    def download(self):
        if self.download_file_path.exists():
            # todo: checksum
            return

        url = "https://doi.org/" + self.doi
        r = requests.get(url)
        if not r.ok:
            raise RuntimeError("DOI could not be resolved.")

        zenodo_record_id = r.url.split("/")[-1]

        url = f"https://zenodo.org/record/{zenodo_record_id}/files/{self.archive_name_with_suffix}"
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True, desc="Downloading {self.archive_name_with_suffix}")
        self.download_file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.download_file_path.with_suffix(".part").open("wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

        progress_bar.close()
        # todo: checksum
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise RuntimeError(f"downloading {url} to {self.download_file_path} failed")

        shutil.move(self.download_file_path.with_suffix(".part"), self.download_file_path)

    def extract(self):
        if self.archive_name_with_suffix.endswith(".zip"):
            with zipfile.ZipFile(self.download_file_path, "r") as zf:
                for member in tqdm(zf.infolist(), desc=f" Extracting {self.archive_name_with_suffix}"):
                    if member.is_dir():
                        raise NotImplementedError("zipped folder")

                    if (self.root / member.filename).exists():
                        continue

                    zf.extract(member, self.root)
        else:
            raise NotImplementedError(self.archive_name_with_suffix)
