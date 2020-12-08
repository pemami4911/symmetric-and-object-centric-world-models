import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import PIL.Image
import numpy as np
from sacred import Experiment

from lib.datasets import ds
from lib.datasets import SequentialHdF5Dataset
from lib.model import net
from lib.model import WorldModel, VRNN 
from lib.geco import GECO
from lib.metrics import compute_video_psnr_and_ssim
from lib.visualization import plot_per_frame_metric
from lib.visualization import create_video_numpy, visualize_images, visualize_slots

from tqdm import tqdm
from pathlib import Path
import shutil
import pprint

ex = Experiment('TRAINING', ingredients=[ds, net])

@ex.config
def cfg():
    training = {
            'batch_size': 16,  # training mini-batch size
            'num_workers': 8,  # pytorch dataloader workers
            'model': 'SDVAE',  # model name
            'context_len': 2,  # how many inference steps before rollout during train
            'seq_len': 3, # total length of seqs for training
            'full_seq_len': 10,
            'iters': 500000,  # train steps if no curriculum
            'use_curriculum': False,  # do curriculum training
            'curriculum_lengths': [3, 5, 8, 10],  # seq_len during curriculum stages
            'curriculum_iters': [],  # train steps at each stage
            'curriculum_batch_sizes': [],
            'curriculum_inference': [],
            'curriculum_geco': [],
            'second_dataset': '',
            'lr': 3e-4,  # Adam LR
            'tensorboard_freq': 100,  # how often to write to TB
            'tensorboard_delete_prev': False,  # delete TB dir if already exists
            'checkpoint_freq': 10000,  # save checkpoints every % steps
            'load_from_checkpoint': False,  # whether to load from a checkpoint or not
            'checkpoint': '',  # name of .pth file to load model state
            'run_suffix': 'debug',  # string to append to run name
            'geco_reconstruction_target': -23000,  # GECO C
            'geco_ema_alpha': 0.99,  # GECO EMA step parameter
            'geco_beta_stepsize': 1e-6,  # GECO Lagrange parameter beta
            'random_seq_start': False,  # if False, start training seqs at frame 0
            'val_batch_size': 8,  # validation mini-batch size
            'val_rollouts': 64,  # number of random rollouts for eval
            'val_freq': 25000  # eval every % steps,
            'out_dir': ''
        }


def save_checkpoint(step, model, model_opt, filepath):
    state = {
        'step': step,
        'model': model.state_dict(),
        'model_opt': model_opt.state_dict(),
    }
    torch.save(state, filepath)


def restore_from_checkpoint(training, checkpoint):
    state = torch.load(checkpoint)
    num_gpus = torch.cuda.device_count()
    if training['model'] == 'Ours' or training['model'] == 'OP3':
        model = WorldModel(batch_size=(training['batch_size'] // num_gpus), context_len=training['context_len'])
    elif training['model'] == 'VRNN':
        model = VRNN(batch_size=(training['batch_size'] // num_gpus), context_len=training['context_len'])

    model = nn.DataParallel(model).to('cuda')
    model.load_state_dict(state['model'], strict=True)
    model_opt = torch.optim.Adam(model.parameters(), lr=training['lr'])
    model_opt.load_state_dict(state['model_opt'])
    model_geco = GECO(training['geco_reconstruction_target'], training['geco_ema_alpha'])
    step = state['step']
    return model, model_opt, model_geco, step


def get_curriculum_seq_len_and_batch(cur_step, curriculum_lengths,
        curriculum_batch_sizes, curriculum_iters, curriculum_inference,
        curriculum_geco, curriculum_stage):
    
    assert len(curriculum_lengths) == len(curriculum_iters)
    for idx, (lens, batches, itrs, infern, geco_) in enumerate(zip(curriculum_lengths,
            curriculum_batch_sizes, curriculum_iters, curriculum_inference, curriculum_geco)):
        if cur_step <= itrs:
            if idx != curriculum_stage:
                new_curriculum_stage = True
            else:
                new_curriculum_stage = False
            return lens, batches, infern, geco_, new_curriculum_stage, idx
    else:
        return curriculum_lengths[-1], curriculum_batch_sizes[-1], curriculum_inference[-1], curriculum_geco[-1],\
                False, len(curriculum_lengths)-1


@ex.automain
def run(training, seed):
    # maybe create
    run_dir = Path(training['out_dir'], 'experiments', 'runs')
    checkpoint_dir = Path(training['out_dir'], 'experiments', 'weights')
    tb_dir = Path(training['out_dir'], 'experiments', 'tb')
    for dir_ in [run_dir, checkpoint_dir, tb_dir]:
        if not dir_.exists():
            dir_.mkdir()

    # Delete if exists
    tb_dbg = tb_dir / training['run_suffix']
    if training['tensorboard_delete_prev'] and tb_dbg.exists():
        shutil.rmtree(tb_dbg)
        tb_dbg.mkdir()
    
    writer = SummaryWriter(tb_dbg, flush_secs=15)
    # Fix random seed
    print(f'setting random seed to {seed}')
    torch.manual_seed(seed)
    
    new_curriculum_stage = False
    curriculum_stage = 0
    num_gpus = torch.cuda.device_count()

    if not training['load_from_checkpoint']:
        # Models
        if training['model'] == 'Ours' or training['model'] == 'OP3':
            model = WorldModel(batch_size=(training['batch_size'] // num_gpus), context_len=training['context_len'])
        elif training['model'] == 'VRNN':
            model = VRNN(batch_size=(training['batch_size'] // num_gpus), context_len=training['context_len'])
        # Optimization
        model_opt = torch.optim.Adam(model.parameters(), lr=training['lr'])
        model_geco = GECO(training['geco_reconstruction_target'], training['geco_ema_alpha'])
        step = 0 
        model = torch.nn.DataParallel(model).to('cuda')
        checkpoint_step = 0
    else:
        model, model_opt, model_geco, step = \
                restore_from_checkpoint(training, checkpoint_dir / training['checkpoint'])
        checkpoint_step = step

    tr_dataset = SequentialHdF5Dataset(d_set='train')
    val_dataset = SequentialHdF5Dataset(d_set='val')
    
    # update training if using curriculum
    if training['use_curriculum']:
        max_seq_len, batch_size, time_inference_sched, geco_target,  _, curriculum_stage = \
                get_curriculum_seq_len_and_batch(step, training['curriculum_lengths'],
                    training['curriculum_batch_sizes'], training['curriculum_iters'], 
                    training['curriculum_inference'], training['curriculum_geco'], curriculum_stage)
        if curriculum_stage > 0 and training['second_dataset'] != '':
            tr_dataset = SequentialHdF5Dataset(d_set='train', h5_path=training['second_dataset'])
            val_dataset = SequentialHdF5Dataset(d_set='val', h5_path=training['second_dataset'])
        model.module.batch_size = (batch_size // num_gpus)
        model.module.time_inference_schedule = time_inference_sched
        model_geco.C = geco_target
    else:
        max_seq_len = training['seq_len']
        batch_size = training['batch_size']

    # Data
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
            batch_size=batch_size, shuffle=True, num_workers=training['num_workers'], drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
            batch_size=training['val_batch_size'], shuffle=False, num_workers=training['num_workers'], drop_last=True)


    max_iters = training['curriculum_iters'][-1] if training['use_curriculum'] else training['iters']
    while step <= max_iters:
        
        for batch in tqdm(tr_dataloader):
            
            img_batch = batch['imgs'].to('cuda')
            seq_start = 0
            # TODO: use curriculum_len here?
            if curriculum_stage > 0:
                sample_seq_len = training['full_seq_len']
            else:
                sample_seq_len = training['seq_len']
            if (training['seq_len'] > max_seq_len) and training['random_seq_start']:
                seq_start = np.random.randint(0, sample_seq_len-max_seq_len)
       
            img_batch = img_batch[:,seq_start:seq_start + max_seq_len]
            if 'actions' in batch:
                action_batch = batch['actions'].to('cuda')
                action_batch = action_batch[:,seq_start:seq_start+max_seq_len]
            else:
                action_batch = None
            out_dict = model(img_batch, action_batch, model_geco, step)
    
            model_opt.zero_grad()
            if num_gpus > 1:
                total_loss = torch.mean(out_dict['total_loss'])
                kl = torch.mean(out_dict['kl'])
                nll = torch.mean(out_dict['nll'])
                inf_steps = torch.mean(out_dict['inference_steps'])
            else:
                total_loss = out_dict['total_loss']
                kl = out_dict['kl']
                nll = out_dict['nll']
                inf_steps = out_dict['inference_steps']

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)

            model_opt.step()

            if step == model.module.geco_warm_start:
                model.module.geco_C_ema = model_geco.init_ema(model.module.geco_C_ema, nll / max_seq_len)
            elif step > model.module.geco_warm_start:
                model.module.geco_C_ema = model_geco.update_ema(model.module.geco_C_ema, nll / max_seq_len)
                model.module.geco_beta = model_geco.step_beta(model.module.geco_C_ema,
                        model.module.geco_beta, training['geco_beta_stepsize'])
            
            # logging
            if step % training['tensorboard_freq'] == 0:
                writer.add_scalar('train/total_loss', total_loss.data.cpu().numpy(), step)
                writer.add_scalar('train/KL', kl.data.cpu().numpy(), step)
                writer.add_scalar('train/reconstruction', nll.data.cpu().numpy(), step)
                writer.add_scalar('train/geco_beta', model.module.geco_beta.data.cpu().numpy(), step)
                writer.add_scalar('train/geco_C_ema', model.module.geco_C_ema.data.cpu().numpy(), step)
                writer.add_scalar('train/avg_inference_steps', inf_steps, step)
                if training['model'] == 'VRNN':
                    visualize_images(writer, img_batch, out_dict, step)
                else:
                    visualize_slots(writer, img_batch, out_dict, step)

            if step > 0 and step % training['checkpoint_freq'] == 0 and step != checkpoint_step:
                prefix = training['run_suffix']
                save_checkpoint(step, model, model_opt, 
                       checkpoint_dir / f'{prefix}-state-{step}.pth')
            

            # do validation
            if step > 0 and step % training['val_freq'] == 0: # and step != checkpoint_step:
                model.eval()
                # single sample at test time
                num_train_samples = model.module.stochastic_samples
                prev_time_inference_schedule = model.module.time_inference_schedule
                dynamics_unc = model.module.dynamics_uncertainty
                model.module.time_inference_schedule = ["I" for _ in range(training['context_len'])] + ["R" for _ in range(max_seq_len-(training['context_len']))]
                print(model.module.time_inference_schedule)
                model.module.stochastic_samples = training['val_rollouts']
                model.module.batch_size = (training['val_batch_size'] // num_gpus)
                model.module.dynamics_uncertainty = False

                metrics = ['psnr', 'ssim']
                tiers = ['best', 'random', 'worst']

                val_stats = {}
                for m in metrics:
                    val_stats[m] = {}
                    for tier in tiers:
                        val_stats[m][tier] = []

                display = list(range(5))
                #display = []
                val_idx = 0
                for batch in tqdm(val_dataloader):
                    img_batch = batch['imgs'].to('cuda')                    

                    if 'actions' in batch:
                        action_batch = batch['actions'].to('cuda')
                        action_batch = action_batch[:,seq_start:seq_start + max_seq_len]
                    else:
                        action_batch = None
                    
                    out_dict = model(img_batch, action_batch, model_geco, step)
                    
                    if training['model'] == 'VRNN':
                        batched_generated_video = np.stack([_.data.cpu().numpy() for _ in out_dict['x_means']])
                        batched_generated_video = np.transpose(batched_generated_video, (1,2,0,3,4,5))
                    else:
                        batched_generated_video = create_video_numpy(out_dict['x_means'], out_dict['masks'])
                    for b_idx in range(training['val_batch_size']):
                        
                        video_stats = compute_video_psnr_and_ssim(img_batch[b_idx], batched_generated_video[b_idx])
                        for m in metrics:
                            for tier in tiers:
                                val_stats[m][tier] += [video_stats[m][tier]['per_frame']]
                    
                                if val_idx in display:
                                    img_seq = torchvision.utils.make_grid(torch.from_numpy(
                                        video_stats[m][tier]['video']).permute(0,3,1,2).contiguous(), nrow=max_seq_len)
                                    writer.add_image('val/{}/{}/{}'.format(m, tier, val_idx), img_seq, step)
                        val_idx += 1
                    
                for m in metrics:
                    means = []
                    stds = []
                    labels = []
                    for tier in tiers:
                        stats = np.stack(val_stats[m][tier])  # [num_val_videos, seq_len]
                        mean_stat = np.mean(stats, 0)  # [seq_len]
                        std_stat = np.std(stats, 0)  # [seq_len]
                        means += [mean_stat]
                        stds += [std_stat]
                        labels += [tier]

                    plot_buf = plot_per_frame_metric(means, stds, labels, m, y_label="{}".format(m), 
                            num_samples=training['val_rollouts'])
                    plot_image = PIL.Image.open(plot_buf)
                    plot_image = torchvision.transforms.ToTensor()(plot_image)

                    writer.add_image('val/{}'.format(m), plot_image, step)

                # reset
                model.module.time_inference_schedule = prev_time_inference_schedule
                model.module.stochastic_samples = num_train_samples
                model.module.batch_size = (batch_size // num_gpus) 
                model.module.dynamics_uncertainty = dynamics_unc
                model.train()

            if step >= max_iters:
                step += 1
                break
            step += 1
            # update training if using curriculum
            if training['use_curriculum']:
                max_seq_len, batch_size, time_inference_sched, geco_target, new_curriculum_stage, curriculum_stage = \
                        get_curriculum_seq_len_and_batch(step, training['curriculum_lengths'],
                            training['curriculum_batch_sizes'], training['curriculum_iters'],
                            training['curriculum_inference'], training['curriculum_geco'], curriculum_stage)
                if new_curriculum_stage:
                    if curriculum_stage == 1 and training['second_dataset'] != '':
                        tr_dataset = SequentialHdF5Dataset(d_set='train', h5_path=training['second_dataset'])
                        val_dataset = SequentialHdF5Dataset(d_set='val', h5_path=training['second_dataset'])
                    else:
                        tr_dataset = SequentialHdF5Dataset(d_set='train')
                        val_dataset = SequentialHdF5Dataset(d_set='val')
                    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                            batch_size=batch_size, shuffle=True, num_workers=training['num_workers'], drop_last=True)
                    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                            batch_size=training['val_batch_size'], shuffle=False, num_workers=training['num_workers'], drop_last=True)
                    new_curriculum_stage = False
                    model_geco.C = geco_target
                    model.module.batch_size = batch_size // num_gpus
                    model.module.time_inference_schedule = time_inference_sched
                    break
                
            else:
                max_seq_len = training['seq_len']
                batch_size = training['batch_size']
            del out_dict
            del batch
