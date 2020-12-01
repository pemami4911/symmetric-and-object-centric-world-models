# symmetric-and-object-centric-world-models
Code accompanying "A Symmetric and Object-Centric World Model for Stochastic Environments" (LINK HERE)

## Installation

The following scripts will create the `multi-object` conda environment (assuming a Linux environment) and CUDA 10.0.

```
$ conda env create -f environment.yml
```

## Data

Download each dataset [from here](https://doi.org/10.5281/zenodo.3673897) and store in a desired folder.

## Experiments

### Training

JSON configurations for training the models for each experiment are organized like so:

```
experiments/
----train/
--------moving_sprites/
------------SDVAE.json
------------SDVAE_DaVOS.json
------------VRNN.json
------------VRNN_star.json
------------OP3.json
--------bair_action_free/
--------bair_actions/
--------planning/
----test/
```

To check the configuration, run
```
$ python train_model.py print_config with experiments/train/bair_action_free/SDVAE.json dataset.data_path=PATH_WHERE_YOU_DOWNLOADED_DATA
```

Train a model with the following, setting the number of GPUs with `CUDA_VISIBLE_DEVICES`:
```
$ CUDA_VISIBLE_DEVICES=0,1 python train_model.py with experiments/train/bair_action_free/SDVAE.json dataset.data_path=PATH_WHERE_YOU_DOWNLOADED_DATA --file_storage ./experiments/runs
```

`sacred` run logs are saved under `./experiments/runs`. Model checkpoints are stored in `./experiments/weights`.

#### Visualization

Monitor training progress with tensorboard. The tensorboard logdir is `./experiments/tb`. Hence, run
```
tensorboard --logdir ./experiments/tb
```
and navigate to `localhost:6006` in your browser.

### Evaluation

We use a similar approach for evaluating the trained models

#### Accuracy

Example:

```
$ CUDA_VISIBLE_DEVICES=0 python eval_model.py with experiments/test/bair_action_free/accuracy/SDVAE_DaVOS.json datasets.data_path=PATH_WHERE_YOU_DOWNLOADED_DATA
```

#### Diversity/Realism

We generate an .h5 file to process with the FVD jupyter notebook to compute diversity/realism metrics.
For example,
```
$ CUDA_VISIBLE_DEVICES=0 python generate_videos.py with experiments/test/bair_action_free/diversity/SDVAE_DaVOS.json datasets.data_path=PATH_WHERE_YOU_DOWNLOADED_DATA
```

The .h5 file will be located under `./experiments/results/BAIR-action-free-diversity/$CHECKPOINT_NAME-seed=$SEED/videos-diversity.h5`