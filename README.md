# HyLFM-Net

## Requirements
- [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus#compute)
- Linux OS (tested on CentOs Linux 7 (Core))
- [[mini]conda](https://docs.conda.io/en/latest/miniconda.html#linux-installers) (tested with miniconda 4.8.4)

## Installation
##### Clone repository
Clone and navigate to this repository
```
git clone git@github.com:kreshuklab/hylfm-net.git
cd hylfm-net
```

##### Install hylfm conda environment
```
conda env create -f environment.yml
conda activate hylfm
conda develop .
```

##### Install time
The install time greatly depends on download speed (several hundred MB).<br>
🕐 Without download (or very fast download), the [installation](#install-hylfm-conda-environment) takes around **9 min**.

## Demo
##### Activate hylfm conda environment
```
conda activate hylfm
```

##### [optional] Choose a CUDA device
A cuda device may be selected before running hylfm (default 0), e.g.
```
export CUDA_VISIBLE_DEVICES=3
```

#### Use Weights and Biases logging
- Login/Sign up at https://wandb.ai/login
- Get your api token at https://wandb.ai/authorize (you'll be ask to provide this token on the first hylfm run, specifically on `import wandb`)

#### Train HyLFM-Net on beads
```
python scripts/train_presets/beads.py
```
🕐 Excluding download time, this training configuration runs for approximately **6 hours** (on a GTX 2080 Ti). Note that the network will likely not have fully converged; increase `max_epochs` to allow for longer training (HyLFM-Net beads used in the paper was trained for 26.5 hours).


#### Test HyLFM-Net on beads (no previous training required)
To download and test HyLFM-net beads run
```
python hylfm/tst.py small_beads_demo
```
🕐 Excluding download time, this test configuration runs for approximately **6,5 min** in total with 12 s per sample (on a GTX 2080 Ti). Most time is spend on computing metrics.


## Data used in [paper](https://rdcu.be/cktHs)
Wagner, N., Beuttenmueller, F., _et al_. Deep learning-enhanced light-field imaging with continuous validation. _Nat Methods_ __18__, 557–563 (2021).

https://doi.org/10.1038/s41592-021-01136-0

beads:
https://www.ebi.ac.uk/biostudies/studies/S-BSST622

Medaka heart: 
https://www.ebi.ac.uk/biostudies/studies/S-BSST604

zebrafish neural activity:
https://www.ebi.ac.uk/biostudies/studies/S-BSST633

## On Your Data
- Implement a `get_tensor_info` function in `hylfm/datasets/local/<your dataset group>.py` analogously to `hylfm/datasets/local/example.py`.
- Add your `DatasetChoice` (defined in hylfm_types.py) and extent `get_dataset_sections` (in datasets/named.py) and `get_transforms_pipeline` (in datasets/transform_pipelines.py) analogously to `DatasetChoice.beads_highc_a`
- Train or test HyLFM-Net as described in [Demo](#demo).


## Settings
To overwrite default settings, like the number of worker threads per pytorch Dataloader, adapt `hylfm/_settings/local.py` (copy from `hylfm/_settings/local.template.py`)
