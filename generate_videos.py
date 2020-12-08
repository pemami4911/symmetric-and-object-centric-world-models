import torch
import torch.nn as nn
import torchvision

import h5py
import numpy as np
from sacred import Experiment

from lib.datasets import ds
from lib.datasets import SequentialHdF5Dataset
from lib.model import net
from lib.model import WorldModel, VRNN
from lib.geco import GECO
from lib.utils import create_video
from tqdm import tqdm
from pathlib import Path
import shutil
import random; random.seed(0)

ex = Experiment('VIDEOS', ingredients=[ds, net])


@ex.config
def cfg():
    test = {
            'batch_size': 16,
            'output_size': [3,64,64],
            'max_num_videos': 8,
            'mode': 'test',
            'rollouts': 5,
            'rollouts_to_keep': 64,
            'store_slots': False,
            'model': 'OP3',
            'context_len': 1,
            'seq_len': 1,
            'num_workers': 8,
            'checkpoint_dir': 'weights',
            'checkpoint': '',
            'all_checkpoints': False,
            'checkpoint_freq': 10000,
            'geco_reconstruction_target': -20500,
            'geco_ema_alpha': 0.99,
            'geco_beta_stepsize': 1e-6,
            'experiment_name': 'NAME_HERE',
            'video_suffix': '',
            'save_FVD_format': False,
            'store_ground_truth': False,
            'save_dynamics': False,  # for visualization,
            'out_dir': ''
        }

# @ex.capture ??
def restore_from_checkpoint(test, checkpoint):
    state = torch.load(checkpoint)
    num_gpus = torch.cuda.device_count()
    if test['model'] == 'Ours' or test['model'] == 'OP3':
        model = WorldModel(batch_size=test['batch_size'] // num_gpus, context_len=test['context_len'])
    elif test['model'] == 'VRNN':
        model = VRNN(batch_size=test['batch_size'] // num_gpus, context_len=test['context_len'])

    model = nn.DataParallel(model).to('cuda')
    model.load_state_dict(state['model'])
    model_geco = GECO(test['geco_reconstruction_target'], test['geco_ema_alpha'])
    return model, model_geco


@ex.capture
def do_eval(test, seed):

    # Fix random seed
    print(f'setting random seed to {seed}')
    torch.manual_seed(seed)
    num_gpus = torch.cuda.device_count()
    # Data
    te_dataset = SequentialHdF5Dataset(d_set=test['mode'])
    te_dataloader = torch.utils.data.DataLoader(te_dataset, batch_size=test['batch_size'],
         shuffle=False, num_workers=test['num_workers'], drop_last=True)
    checkpoints = [Path(test['out_dir'], 'experiments', test['checkpoint_dir'], test['checkpoint'])]
    
    if test['experiment_name'] == 'NAME_HERE':
        print('Please provide a valid name for this experiment to keep things organized!')
        exit(1)

    # save videos as [NUM_VIDEOS, SEQ_LEN, W, H, 3] (TF format for FVD)
    for c_idx,c in enumerate(checkpoints):
        out_dir = Path(test['out_dir'], 'experiments', 'results', test['experiment_name'], c.stem + '.pth' + f'-seed={seed}')
        if not out_dir.exists():
            out_dir.mkdir(parents=True)
        video_name = 'videos.h5'
        if test['video_suffix'] != '':
            video_name = 'videos-{}.h5'.format(test['video_suffix'])
        out_h5_name = out_dir /  video_name
        print(f'Saving videos to {out_h5_name}')
        out_file = h5py.File(out_h5_name, 'w')
        
        model, model_geco = restore_from_checkpoint(test, c)
        model.eval()
        model.module.stochastic_samples = test['rollouts']
        if test['model'] == 'Ours' or test['model'] == 'OP3':
            if len(model.module.time_inference_schedule) != test['seq_len']:
                model.module.time_inference_schedule = ["I" for _ in range(test['context_len'])] + ["R" for _ in range(test['seq_len'] - test['context_len'])]
            print(model.module.time_inference_schedule)
        print('!!! Model {} rollouts'.format(model.module.stochastic_samples))

        C, H, W = test['output_size']
        if test['store_slots']:
            K = model.module.K
            vid_dataset_np = np.zeros((test['max_num_videos'], test['rollouts_to_keep'],
                test['seq_len'], K, C, H, W)).astype('float32')

            full_vid_dataset_np = np.zeros((test['max_num_videos'], 2, test['seq_len'], C, H, W)).astype('float32')
        else:
            vid_dataset_np = np.zeros((test['max_num_videos'], test['rollouts_to_keep'],
                test['seq_len'], C, H, W)).astype('float32')
        vid_idx = 0
        dynamics = []
        posteriors = []

        for i,batch in enumerate(tqdm(te_dataloader)):
            if vid_idx == test['max_num_videos']:
                break
            seq_start = 0

            imgs = batch['imgs'].to('cuda')
            imgs = imgs[:,seq_start:seq_start+test['seq_len']]  # [batch_size, T, C, H, W]

            if 'actions' in batch:
                actions = batch['actions'].to('cuda')
                actions = actions[:,seq_start:seq_start+test['seq_len']]
            else:
                actions = None
           
            if test['store_ground_truth']:
                video = imgs.permute(1,0,2,3,4)  # [T, batch_size, C, H, W]
                video = video.unsqueeze(2)
            else:
                
                model_outs = model(imgs, actions, model_geco, i)

                if test['save_dynamics']:
                    dynamics += [np.stack([_.data.cpu().numpy() for _ in model_outs['dynamics']])]
                    posteriors += [np.stack([_.data.cpu().numpy() for _ in model_outs['lambdas']])]
                if test['model'] == 'VRNN':
                    video = torch.stack(model_outs['x_means'])  # [T, batch_size, rollouts, C, H, W]
                else:
                    video = create_video(model_outs['x_means'], model_outs['masks'],
                            sum_over_k=not test['store_slots'])  # [T, batch_size, rollouts, C, H, W]
            if test['store_slots']:
                video = video.permute(1,2,0,3,4,5,6).contiguous()  # [batch_size, rollouts, T, K,C, H, W]
            else:
                video = video.permute(1,2,0,3,4,5).contiguous()  # [batch_size, rollouts, T, C, H, W]
            upper = min(vid_idx+test['batch_size'], test['max_num_videos'])
            vid_dataset_np[vid_idx:upper] = video[:,:test['rollouts_to_keep']].detach().data.cpu().numpy()
            if test['store_slots']: # also save ground truth at same time for speed
                full_reconstruction = create_video(model_outs['x_means'], model_outs['masks'], sum_over_k=True)
                full_vid_dataset_np[vid_idx:upper,0] = imgs.data.cpu().numpy()
                full_vid_dataset_np[vid_idx:upper,1] = full_reconstruction.permute(1,2,0,3,4,5).contiguous()[:,0].data.cpu().numpy()
            vid_idx += imgs.shape[0]
        if test['save_FVD_format']:
            # save as [num_videos, video_len, H, W, C] and [0-255]
            vid_dataset_np = 255 * vid_dataset_np
            num_videos, num_rollouts, T, _, _, _ = vid_dataset_np.shape
            vid_dataset_np = vid_dataset_np.reshape(num_videos * num_rollouts, T, C, H, W)
            vid_dataset_np = np.transpose(vid_dataset_np, (0,1,3,4,2))
        out_file['videos'] = vid_dataset_np
        if test['store_slots']:
            out_file['ground_truth'] = full_vid_dataset_np
        if test['save_dynamics']:
            out_file['dynamics'] = np.array(dynamics)
            out_file['lambda_0'] = model.module.lamda_0.data.cpu().numpy()
            out_file['lambda_I'] = np.array(posteriors)
        out_file.close()

@ex.automain
def run(_run, seed):
    do_eval(seed=seed)
