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
```
##### Install time
The install time greatly depends on download speed (several hundred MB).<br>
🕐 Without download (or very fast download), the [installation](#install-hylfm-conda-environment) takes around **9 min**.

## Demo
##### Activate hylfm conda environment
```
conda activate hylfm
```

##### Choose a CUDA device
When running hylfm ([train](#train-hylfm-net-on-beads) or [test](#test-hylfm-net-on-beads)) a CUDA device will be used. Run `nvidia-smi` to see available CUDA devices on your machine. If more than one CUDA device is available you need to specify which one to use either bei setting the 'CUDA_VISIBLE_DEVICES' environment variable or by command line argument `--cuda`:
```
python -m hylfm --cuda <cuda device, e.g. 0> ...
```

#### Train HyLFM-Net on beads
```
python -m hylfm configs/train/beads/z_out51/small_f8.yml
```
🕐 Excluding download time, this training configuration runs for approximately **1,5 days** (on a GTX 2080 Ti).


#### Test HyLFM-Net on beads
Download (to hylfm-net/download).
```
python -m hylfm configs/test/beads/z_out51/small_f8.yml --checkpoint small_beads_demo
```
🕐 Excluding download time, this test configuration runs for approximately **6,5 min** in total with 12 s per sample (on a GTX 2080 Ti). Most time is spend on computing metrics specified in `configs/metrics/beads.yml`.


##### Monitoring with TensorBoard
With default settings and in the [hylfm repository directory](#clone-repository), run
```
$ tensorboard --logdir=logs
```
which will yield an output like:
```
TensorBoard 1.15.0 at <address>:<port> (Press CTRL+C to quit)
```
With the TensorBoard running, open `<address>:<port>` in your browser to see logged scalars (screenshot from [training](#train-hylfm-net-on-beads)):<br>
![tensorboard screenshot](images/tensorboard_scalars.png "Tensorboard Scalars in training")

Switch to `<address>:<port>/#images` for 2d (and max projections of 3d) tensors and plots (screenshot from [testing](#test-hylfm-net-on-beads)):<br>
![tensorboard screenshot](images/tensorboard_images.png "Tensorboard Images of test run")


## On Your Data
- Write a `get_tensor_info` function in `hylfm/datasets/local/<your dataset group>.py` analogously to `hylfm/datasets/local/example.py`.
- Write a config yaml file in `configs` analogously to `configs/train/beads/z_out51/small_f8.yml`.
    - Use tensors from \<your dataset group\>:
   
        in demo `small_f8.yml`:
        ```yaml
        - tensors: {lf: beads.small_0, ls_reg: beads.small_0}
        ```
        
        in your `configs/**/<your config>.yml`:
        ```yaml
        - tensors: {<name>: local.<your dataset group>.<tag>, <other name>: local.<your dataset group>.<tag>}
        ```

- Train or test HyLFM-Net as described in [Demo](#demo).

## HyLFM config yaml
```yaml
toolbox:                             # defines yaml anchors for convenience (is ignored by hylfm setup)
  eval_batch_size: &eval_batch_size 1  # example for yaml placeholders 
  <key>: <file name.yml>  # any subconfig part can be substituted by a yml file in configs/<key>/<file name.yml>

model:                   # model setup
  name: <str>              # model name as specified in hyflm.models
  kwargs: {...}            # model key word arguments
  checkpoint: <str>        # path to hylfm checkpoint to load instead of random initialization
  partial_weights: <bool>  # if true ignore additionl/missing tensors in model checkpoint with different architecture
  precision: <float|half>  # model precision

stages:  # list of stages, each stage may be a training or a test stage, and will be run consecutively
  - train:  # stage name
      optimizer:         # stages with 'optimizer' are training stages
        name: <str>      # optimizer class as specified in hylfm.optimizers
        kwargs: {...}

      max_epochs: <int>  # stop after max epochs even if validation score is still improving
      metrics: [...]     # metrics to evaluate (and log according to 'log')
      log:               # loggers as specified in hylfm.loggers
        TqdmLogger: {}                          # show tqdm progress bar
        TensorBoardLogger:                      # log to TensorBoard event file
          scalars_every: {value: <int>, unit: <epoch|iteration>}  #  how often to log scalar metrics and loss
          tensors_every: {value: <int>, unit: <epoch|iteration>}  #  how often to log below specified tensors and plots
          tensor_names: [<str>, ...]            # names of tensors to be logged as 2d (max projection) images
        FileLogger:                             # individual output files
          scalars_every: {value: <int>, unit: <epoch|iteration>}
          tensors_every: {value: <int>, unit: <epoch|iteration>}
          tensor_names: [<str>, ...]            # names of tensors to be logged as .tif files

      criterion:                  # criterion to optimize ('loss')
        name: <str>                 # `LossOnTensors` child class specified in hylfm.losses
        kwargs: {...}               # key word arguments
        tensor_names: {...}         # tensor name mapping

      sampler:                    # data sampling strategy
        base: <str>               # `torch.utils.data.sampler.Sampler` child class in torch.utils.data
        drop_last: <bool>           # drop last samples if less samples than 'batch_size' remain

      batch_preprocessing: [...]  # List of `Transform` child classes as specified in hylfm.transformations with kwargs 
      batch_preprocessing_in_step: [...]  # like 'batch_preprocessing', but in the iteration step (on GPU)
      batch_postprocessing: [...]  # like 'batch_preprocessing', but after `model.forward()`

      data:
        - batch_size: <int>
          sample_preprocessing: [...]  # like 'batch_preprocessing', but before batch assembly on single sample
          datasets:                    # list of datasets=named tensors
            - tensors: {<tensor_name>: <str>, ...}  # named tensors, each resolved by `hylfm.datasets.get_tensor_info()`
              [sample_transformations: [...]]       # overwrites 'sample_preprocessing' for this dataset (optional)    
              # subselect indices as List[int], int, or string resolved by `hylfm.setup._utils.indice_string_to_list()`
              indices: null                         # null = "all indices"

      validate:            # validation stage of this training stage
        ...                  # a validation stage is an evaluation stage with the following additional keys: 
        score_metric: <str>  # metric name (that has to exist in this stage's 'metrics') to use as validation score
        period: {value: <int>, unit: <epoch|iteration>}  # how often to validate wrt to parent training stage
        patience: <int>      # stop after not improvement of 'score_metric' for 'patience' validations

  - test:  # stage name of an evaluation stage (no 'optimizer' defined)
      # the following fields are described above in 'train':
      metrics:  [...]
      log: {...}      
      batch_preprocessing: [...]
      batch_preprocessing_in_step: [...]
      batch_postprocessing: [...]
      data: [...]
```


## Settings
To overwrite default settings, like the number of worker threads per pytorch Dataloader, adapt `hylfm/_settings/local.py` (copy from `hylfm/_settings/local.template.py`)
