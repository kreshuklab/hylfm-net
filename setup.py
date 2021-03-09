import versioneer

from setuptools import find_packages, setup
from pathlib import Path

# Get the long description from the README file
long_description = (Path(__file__).parent / "README.md").read_text()

setup(
    name="hylfm-net",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="HyLFM-Net",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kreshuklab/hylfm-net",
    author="Fynn Beuttenmueller",
    author_email="thefynnbe@gmail.com",
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=["tests"]),  # Required
    install_requires=[
        "dill",
        "h5py",
        "imagecodecs",
        "inferno",
        "jupyter",
        "matplotlib",
        "numpy",
        "pandas",
        "pip",
        "plotly",
        "python",
        "pytorch",
        "pytorch-msssim",
        "pyyaml",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "six",
        "tensorboard",
        "tensorflow-gpu",
        "tifffile",
        "torchvision",
        "tqdm",
        "typer",
        "typing_extensions",
        "wandb",
        "z5py",
        "csbdeep @ git+ssh://git@github.com/csbdeep/csbdeep",  ##egg=csbdeep",
    ],
    entry_points={"console_scripts": ["hylfm=hylfm.__main__:main"]},
    # extras_require={"test": ["pytest"]},
    project_urls={  # Optional
        "Bug Reports": "https://github.com/kreshuklab/hylfm-net/issues",
        "Source": "https://github.com/kreshuklab/hylfm-net/",
    },
)
