import torch
import torch.nn as nn
import torchvision

import numpy as np
from sacred import Experiment
from movis.datasets import ds
from movis.datasets import SequentialHdF5Dataset
from movis.model import net
from movis.model import SDVAE, VRNN
from movis.metrics import compute_video_ar, compute_video_psnr_and_ssim
from movis.visualization import plot_per_frame_metric
from movis.utils import create_video, get_log_var_grad
from movis.geco import GECO
from tqdm import tqdm
from pathlib import Path
import shutil
import pickle
import pprint

ex = Experiment('EVAL', ingredients=[ds, net])


@ex.config
def cfg():
    test = {
            'batch_size': 16,
            'model': 'SDVAE',
            'mode': 'test',
            'context_len': 1,
            'seq_len': 1,
            'num_workers': 0,
            'experiment_name': 'NAME_HERE',
            'checkpoint': '',
            'metric': ['AR'],
            'rollouts': 64,
            'geco_reconstruction_target': -20500,
            'geco_ema_alpha': 0.99,
            'geco_beta_stepsize': 1e-6,
            'max_num_videos': 16,
            'logvargrad': False
        }

# @ex.capture ??
def restore_from_checkpoint(test, checkpoint):
    state = torch.load(checkpoint)
    num_gpus = torch.cuda.device_count()
    if test['model'] == 'OP3':
        model = OP3(batch_size=test['batch_size'] // num_gpus, context_len=test['context_len'])
    elif test['model'] == 'SDVAE':
        model = SDVAE(batch_size=test['batch_size'] // num_gpus, context_len=test['context_len'])
    elif test['model'] == 'VRNN':
        model = VRNN(batch_size=test['batch_size'] // num_gpus, context_len=test['context_len'])

    model = nn.DataParallel(model).to('cuda')
    model.load_state_dict(state['model'])
    model_geco = GECO(test['geco_reconstruction_target'], test['geco_ema_alpha'])
    return model, model_geco


@ex.capture
def do_eval(test, info, seed):
    # Fix random seed
    print(f'setting random seed to {seed}')
    torch.manual_seed(seed)
    num_gpus = torch.cuda.device_count()

    # Data
    te_dataset = SequentialHdF5Dataset(d_set=test['mode'])
    te_dataloader = torch.utils.data.DataLoader(te_dataset, batch_size=test['batch_size'], shuffle=True, num_workers=test['num_workers'])

    checkpoints = [Path('experiments', 'weights', test['checkpoint'])]
    print("Found checkpoints: {}".format(checkpoints))
    
    
    if test['experiment_name'] == 'NAME_HERE':
        print('Please provide a valid name for this experiment to keep things organized!')
        exit(1)

    exper_name = test['experiment_name']
    if info._id is not None:
        exper_name += f'-{info._id}'
    out_dir = Path('experiments', 'results', exper_name, \
            test['checkpoint'] + f'-seed={seed}')
    
    print(f'Saving to... {out_dir}')
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    for c_idx,c in enumerate(checkpoints):
        model, model_geco = restore_from_checkpoint(test, c)
        model.eval()
        model.module.movis_samples = test['rollouts']
        if test['logvargrad']:
            model.train()
        print('!!! Model doing {} rollouts'.format(model.module.movis_samples))

        metrics = []
        if 'AR' in test['metric']:
            metrics += ['ar']
        if 'PSNR,SSIM' in test['metric']:
            metrics += ['psnr', 'ssim']
            
        tiers = ['best', 'random', 'worst']
        test_stats = {}
        for m in metrics:
            test_stats[m] = {}
            for tier in tiers:
                test_stats[m][tier] = []
            test_stats[m]['mean'] = []
        test_stats['iou_video'] = []

        if test['logvargrad']:
            lvg=get_log_var_grad(te_dataloader, model, model_geco, seq_len=test['seq_len'])
            print(lvg)
            test_stats['logvargrad'] = lvg
            filepath = out_dir / 'stats.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(test_stats, f)
            exit(0)
        
        model.module.time_inference_schedule = ["I" for _ in range(test['context_len'])] + ["R" for _ in range(test['seq_len'] - test['context_len'])]
        print(model.module.time_inference_schedule)

        total_num_videos = 0
        for i,batch in enumerate(tqdm(te_dataloader)):
            if total_num_videos > test['max_num_videos']:
                break
            seq_start = 0
            if 'flow' in batch:
                flow_batch = batch['flow'].to('cuda')
                flow_batch = flow_batch[:,seq_start : seq_start + test['seq_len']]
                seq_start += 1
            else:
                flow_batch = None

            imgs = batch['imgs'].to('cuda')
            imgs = imgs[:,seq_start:seq_start+test['seq_len']]
            time_steps = torch.arange(0,test['seq_len']).to('cuda').unsqueeze(0).repeat(num_gpus,1)
            if 'masks' in batch:
                masks = batch['masks'].to('cuda')
                masks = masks[:,seq_start:seq_start+test['seq_len']]
            if 'actions' in batch:
                actions = batch['actions'].to('cuda')
                actions = actions[:,seq_start:seq_start+test['seq_len']]
            else:
                actions = None
            total_num_videos += imgs.shape[0]            
            
            model_outs = model(imgs, actions, model_geco, i, time_steps, flow_batch)

            if 'AR' in test['metric']:
                for b_idx in range(masks.shape[0]):
                    ground_truth_mask_video = masks[b_idx]
                    generated_masks = torch.stack([_.detach() for _ in model_outs['masks']])  # [seq_len, batch, num_rollouts, K, 1, H, W]
                    #if model.module.multi_step == 'max':
                    #    generated_masks = generated_masks.unsqueeze(2)
                    generated_mask_video = generated_masks[:,b_idx].permute(1,0,2,3,4,5).contiguous()
                    video_stats, iou_video = compute_video_ar(ground_truth_mask_video,
                            generated_mask_video, False)
                    if iou_video is not None:
                        test_stats['iou_video'] += [(ground_truth_mask_video, iou_video)]
                    for m in ['ar']:
                        for tier in tiers:
                            test_stats[m][tier] += [video_stats[m][tier]['per_frame']]
                        test_stats[m]['mean'] += [video_stats[m]['mean']]
            if 'PSNR,SSIM' in test['metric']:
                with torch.no_grad():
                    if test['model'] == 'VRNN':
                        generated_video = np.stack([_.data.cpu().numpy() for _ in model_outs['x_means']])
                        generated_video = np.transpose(generated_video, (1,2,0,3,4,5))
                    for b_idx in range(imgs.shape[0]):
                        ground_truth_video = imgs[b_idx]
                        if test['model'] == 'VRNN':
                            generated_video = generated_video[b_idx]
                        else:
                            generated_video = create_video(model_outs['x_means'], model_outs['masks'])[:,b_idx]
                            #if model.module.multi_step == 'max':
                                # [seq_len, C, H, W]
                            #    generated_video = generated_video.unsqueeze(1)
                            generated_video = generated_video.permute(1,0,2,3,4)
                        video_stats = compute_video_psnr_and_ssim(ground_truth_video,
                                generated_video, False)
                        
                        for m in ['psnr', 'ssim']:
                            for tier in tiers:
                                test_stats[m][tier] += [video_stats[m][tier]['per_frame']]
                            test_stats[m]['mean'] += [video_stats[m]['mean']]
            del imgs
            del model_outs
                
        for m in metrics:
            means = []
            stds = []
            labels = []
            for tier in tiers:
                stats = np.stack(test_stats[m][tier])
                mean_stat = np.mean(stats, 0)
                std_stat = np.std(stats, 0)
                means += [mean_stat]
                stds += [std_stat]
                labels += [tier]
                q = np.nanquantile(stats,q=[0.05,0.5,0.95], axis=0)
                print("[{}-{} quantiles] 5%: {:.4f}, 50%: {:.4f}, 95%: {:.4f}".format(m,tier,
                    np.mean(q[0][test['context_len']:]), np.mean(q[1][test['context_len']:]), np.mean(q[2][test['context_len']:])))

            save_file = out_dir / '{}.pdf'.format(m)
            plot_per_frame_metric(means, stds, labels, m.upper(), test['rollouts'], save_file)
                        
    filepath = out_dir / 'stats.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(test_stats, f)


@ex.automain
def run(_run, seed):
    do_eval(info=_run, seed=seed)
