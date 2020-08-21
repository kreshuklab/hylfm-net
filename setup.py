from io import open
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="hylfm-net",
    version="0.1.0",
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
        "ignite",
        "imagecodecs",
        "jupyter",
        "matplotlib",
        "numpy",
        "pandas",
        "pip",
        "plotly",
        "python",
        "pytorch-msssim",
        "pytorch",
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
