import shutil
import zipfile

import requests
from tqdm import tqdm

from hylfm import settings
from .base import TensorInfo


class OnlineTensorInfo(TensorInfo):
    def __init__(self, doi: str, file_name: str, in_file_glob: str, **super_kwargs):
        self.doi = doi
        self.file_name = file_name
        extraced_name = self._init_extract()
        super().__init__(root=settings.cache_dir / doi / extraced_name, location=in_file_glob, **super_kwargs)
        self.download_file_path = settings.download_dir / doi / self.file_name

    def download(self):
        if self.download_file_path.exists():
            # todo: checksum
            return

        url = "https://doi.org/" + self.doi
        r = requests.get(url)
        if not r.ok:
            raise RuntimeError("DOI could not be resolved.")

        zenodo_record_id = r.url.split("/")[-1]

        url = f"https://zenodo.org/record/{zenodo_record_id}/files/{self.file_name}"
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True, desc=f"Downloading {self.file_name}")
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

    def _link_download(self):
        if self.root.exists():
            assert self.root.is_symlink()
            assert self.root.resolve() == self.download_file_path.resolve()
        else:
            self.root.symlink_to(self.download_file_path)

    def _unzip(self):
        assert self.file_name.endswith(".zip")
        with zipfile.ZipFile(self.download_file_path, "r") as zf:
            for member in tqdm(zf.infolist(), desc=f" Extracting {self.file_name}"):
                if member.is_dir():
                    raise NotImplementedError("zipped folder")

                if (self.root / member.filename).exists():
                    continue

                zf.extract(member, self.root)

    def _init_extract(self) -> str:
        if self.file_name.endswith(".zip"):
            self.extract = self._unzip
            return self.file_name[: -len(".zip")]
        elif self.file_name.endswith(".h5"):
            self.extract = self._link_download
            return self.file_name
        else:
            raise NotImplementedError(self.file_name)

    def extract(self) -> None:
        raise RuntimeError("extract called before '_init_extract'")
