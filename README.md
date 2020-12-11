# symmetric-and-object-centric-world-models
Code accompanying "A Symmetric and Object-Centric World Model for Stochastic Environments" [[pdf](https://github.com/orlrworkshop/orlrworkshop.github.io/blob/master/pdf/ORLR_3.pdf)]

## Installation

The following will create the `multi-object` conda environment (assuming Linux and CUDA 10.0).

```
$ conda env create -f environment.yml
```

## Data

Download the BAIR Towel Pick 30k data as an h5py file [from here](https://www.dropbox.com/s/bodzlbzrzduxagn/towel_pick_30k_64x64.h5?dl=0) and store in a desired folder.

## Experiments

### Training

JSON configuration files for training the models are organized like so:

```
experiments/
----train/
--------bair_actions/
------------Ours.json
------------VRNN.json
------------OP3.json
----eval/
--------bair_actions/
------------accuracy/
----------------Ours.json
----------------VRNN.json
----------------OP3.json
```

To check the configuration, run
```
$ python train_model.py print_config with experiments/train/bair_actions/Ours.json dataset.data_path=PATH_WHERE_YOU_DOWNLOADED_DATA training.out_dir=PATH_WHERE_YOU_STORE_OUTPUTS
```

Train a model with the following, setting the number of GPUs with `CUDA_VISIBLE_DEVICES` (shown here with 2):
```
$ CUDA_VISIBLE_DEVICES=0,1 python train_model.py with experiments/train/bair_actions/Ours.json dataset.data_path=PATH_WHERE_YOU_DOWNLOADED_DATA training.out_dir=PATH_WHERE_YOU_STORE_OUTPUTS seed=YOUR_SEED --file_storage PATH_WHERE_YOU_STORE_OUTPUTS/experiments/runs
```

`sacred` run logs are saved under `PATH_WHERE_YOU_STORE_OUTPUTS/experiments/runs`. Model checkpoints are stored in `PATH_WHERE_YOU_STORE_OUTPUTS/experiments/weights`.

#### Visualization

Monitor training progress with tensorboard. The tensorboard logdir is `PATH_WHERE_YOU_STORE_OUTPUTS/experiments/tb`. Hence, run
```
tensorboard --logdir PATH_WHERE_YOU_STORE_OUTPUTS/experiments/tb
```
and navigate to `localhost:6006` in your browser.

### Evaluation

We use a similar approach for evaluating the trained models, e.g., for computing the accuracy metrics (PSNR/SSIM).

#### Accuracy

Example:

```
$ CUDA_VISIBLE_DEVICES=0 python eval_model.py with experiments/test/bair_actions/accuracy/Ours.json datasets.data_path=PATH_WHERE_YOU_DOWNLOADED_DATA out_dir=PATH_WHERE_YOU_STORE_OUTPUTS
```
#### Realism

We generate an .h5 file to process with the FVD jupyter notebook to compute the realism metric. First make sure to create 5 ground truth video sample populations. Change `store_ground_truth` to `true` in `experiments/test/bair_actions/realism/Ours.json` and run with 5 different random seeds.

Then,
```
$ CUDA_VISIBLE_DEVICES=0 python generate_videos.py with experiments/test/bair_actions/realism/Ours.json datasets.data_path=PATH_WHERE_YOU_DOWNLOADED_DATA out_dir=PATH_WHERE_YOU_STORE_OUTPUTS seed=1000
```
#### Diversity

We compute this by loading the results from the accuracy evaluation and computing the difference in the average SSIM over the 100 rollouts per video.
