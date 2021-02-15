import zipfile

from tqdm import tqdm

from hylfm import settings
from .base import TensorInfo
from hylfm.utils.io import download_file_from_zenodo


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

        download_file_from_zenodo(self.doi, self.file_name, self.download_file_path)

    def _link_download(self):
        if self.root.exists():
            assert self.root.resolve() == self.download_file_path.resolve()
        else:
            self.root.link_to(self.download_file_path)

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
