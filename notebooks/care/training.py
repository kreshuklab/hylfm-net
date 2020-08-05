# from __future__ import print_function, unicode_literals, absolute_import, division
# # Demo: Neural network training for denoising of *Tribolium castaneum*
#
# This notebook demonstrates training a CARE model for a 3D denoising task, assuming that training data was already generated via [1_datagen.ipynb](1_datagen.ipynb) and has been saved to disk to the file ``data/my_training_data.npz``.
#
# Note that training a neural network for actual use should be done on more (representative) data and with more training time.
#
# More documentation is available at http://csbdeep.bioimagecomputing.com/doc/.

# The TensorFlow backend uses all available GPU memory by default, hence it can be useful to limit it:
from argparse import ArgumentParser

parser = ArgumentParser(description="care")
parser.add_argument("cuda_visible_devices")
parser.add_argument("model_name")
parser.add_argument("subpath")
parser.add_argument("--train_patches", default="train_patches")
parser.add_argument("--train_steps_per_epoch", type=int, default=100)

args = parser.parse_args()

# limit_gpu_memory(fraction=1/2)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
import tensorflow as tf

assert tf.test.is_built_with_cuda()

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


import numpy as np
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

from pathlib import Path
from tifffile import imread
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE

# # Training data
#
# Load training data generated via [1_datagen.ipynb](1_datagen.ipynb), use 10% as validation data.

# In[6]:


data_path = Path("/scratch/beuttenm/lnet/care/") / args.subpath
(X, Y), (X_val, Y_val), axes = load_training_data(
    str(data_path / f"{args.train_patches}.npz"), validation_split=0.1, verbose=True
)

c = axes_dict(axes)["C"]
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

# plt.figure(figsize=(12, 5))
# plot_some(X_val[:5], Y_val[:5])
# plt.suptitle("5 example validation patches (top row: source, bottom row: target)")

# # CARE model
#
# Before we construct the actual CARE model, we have to define its configuration via a `Config` object, which includes
# * parameters of the underlying neural network,
# * the learning rate,
# * the number of parameter updates per epoch,
# * the loss function, and
# * whether the model is probabilistic or not.
#
# The defaults should be sensible in many cases, so a change should only be necessary if the training process fails.
#
# ---
#
# <span style="color:red;font-weight:bold;">Important</span>: Note that for this notebook we use a very small number of update steps per epoch for immediate feedback, whereas this number should be increased considerably (e.g. `train_steps_per_epoch=400`) to obtain a well-trained model.

config = Config(axes, n_channel_in, n_channel_out, train_steps_per_epoch=args.train_steps_per_epoch)
print(config)
print(vars(config))


# We now create a CARE model with the chosen configuration:

model_basedir = Path("/g/kreshuk/LF_computed/lnet/care/models") / args.subpath.strip("/")
model_name = f"{args.model_name}_spe{args.train_steps_per_epoch}_on_{args.train_patches.replace('train_patches_', '')}"
model = CARE(config=config, name=model_name, basedir=str(model_basedir))

# # Training
#
# Training the model will likely take some time. We recommend to monitor the progress with [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) (example below), which allows you to inspect the losses during training.
# Furthermore, you can look at the predictions for some of the validation images, which can be helpful to recognize problems early on.
#
# You can start TensorBoard from the current working directory with `tensorboard --logdir=.`
# Then connect to [http://localhost:6006/](http://localhost:6006/) with your browser.
#
# ![](http://csbdeep.bioimagecomputing.com/img/tensorboard_denoising3D.png)

history = model.train(X, Y, validation_data=(X_val, Y_val))


# Plot final training history (available in TensorBoard during training):

print(sorted(list(history.history.keys())))
plt.figure(figsize=(16, 5))
plot_history(history, ["loss", "val_loss"], ["mse", "val_mse", "mae", "val_mae"])
plt.savefig(str(model_basedir / f"{model_name}_history.svg"))

# # Evaluation
#
# Example results for validation images.

plt.figure(figsize=(12, 7))
_P = model.keras_model.predict(X_val[:5])
if config.probabilistic:
    _P = _P[..., : (_P.shape[-1] // 2)]
plot_some(X_val[:5], Y_val[:5], _P, pmax=99.5)
plt.suptitle(
    "5 example validation patches\n"
    "top row: input (source),  "
    "middle row: target (ground truth),  "
    "bottom row: predicted from source"
)
plt.savefig(str(model_basedir / f"{model_name}_validation_samples.png")) 

# # Export model to be used with CSBDeep **Fiji** plugins and **KNIME** workflows
#
# See https://github.com/CSBDeep/CSBDeep_website/wiki/Your-Model-in-Fiji for details.

model.export_TF()
