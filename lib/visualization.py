import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_palette('Dark2')
import io
import numpy as np
import torch
import torchvision


def create_video_numpy(means, masks):
    T = len(means)
    batched_videos = []
    for t in range(T):
        batched_videos += [np.sum(means[t].data.cpu().numpy() * masks[t].data.cpu().numpy(), 2)]
    batched_videos = np.stack(batched_videos).transpose(1,2,0,3,4,5)
    return batched_videos


def plot_per_frame_metric(per_frame_means, per_frame_stds, labels, metric_name, y_label=None, num_samples=None, save_file=None):
    plt.figure(figsize=(8,8))
    fontsize=24
    titlefontsize=26
    title = metric_name
    if num_samples:
        title += ' N={}'.format(num_samples)
    plt.title(metric_name, fontsize=titlefontsize, fontweight='bold')
    if y_label is None:
        y_label = metric_name
    plt.ylabel('{}'.format(y_label), fontsize=fontsize)
    plt.xlabel('frame', fontsize=fontsize)
    plt.xticks(np.arange(per_frame_means[0].shape[0]), labels=list(range(1,per_frame_means[0].shape[0]+1)), fontweight='bold', fontsize=fontsize)
    plt.yticks(fontweight='bold', fontsize=fontsize)
    plt.grid(True)

    for means, std, label in zip(per_frame_means, per_frame_stds, labels):
        plt.errorbar(np.arange(means.shape[0]), means, yerr=std, linewidth=5,
                capsize=4, elinewidth=2, label=label)
    plt.legend(fontsize=20)
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg',bbox_inches='tight')
        buf.seek(0)
        return buf


def visualize_images(writer, batch_data, model_outs, step):
    with torch.no_grad():
        batch_size, T, C, H, W = batch_data.shape
        imgs = []
        for i in range(T):
            imgs += [batch_data[0,i]]
    
        T = len(model_outs['x_means'])
        recon_seq = []
        for i in range(T):
            recon_seq += [model_outs['x_means'][i][0]]
        recon_seq = torch.stack(recon_seq)

        img_seq = torchvision.utils.make_grid(torch.stack(imgs))
        writer.add_image('image', img_seq, step)

        recon_seq = torchvision.utils.make_grid(recon_seq)
        writer.add_image('reconstruction', recon_seq, step)


def visualize_slots(writer, batch_data, model_outs, step):
    """
    Render images for each mask and slot reconstruction,
    as well as mask*slot 
    """

    with torch.no_grad():
        batch_size, T, C, H, W = batch_data.shape
        imgs = []
        for i in range(T):
            imgs += [batch_data[0,i]]
        if 'x_means' in model_outs:
            T_ = len(model_outs['x_means'])
            
            x_means_seq = []
            pis_seq = []
            recon_seq = []

            for i in range(T_):
                x_loc = model_outs['x_means'][i]
                x_loc = x_loc.view(batch_size, -1, C, H, W)
                _, K, _, _, _ = x_loc.shape
                pis = model_outs['masks'][i].view(batch_size, K, 1, H, W)
                reconstruction = torch.zeros((C,H,W)).to('cuda')
                for j in range(K):
                    reconstruction += (x_loc[0,j] * pis[0,j])

                comp_grid = torchvision.utils.make_grid(x_loc[0])
                mask_grid = torchvision.utils.make_grid(pis[0])

                x_means_seq += [comp_grid]
                pis_seq += [mask_grid]
                recon_seq += [reconstruction]
            
            x_means_seq = torch.transpose(
                    torchvision.utils.make_grid(torch.transpose(torch.stack(x_means_seq), 3, 2)), 2, 1)
            pis_seq = torch.transpose(
                    torchvision.utils.make_grid(torch.transpose(torch.stack(pis_seq), 3, 2)), 2, 1)
            recon_seq = torchvision.utils.make_grid(torch.stack(recon_seq))
            
            writer.add_image('components', x_means_seq, step)
            writer.add_image('masks', pis_seq, step)
            writer.add_image('reconstruction', recon_seq, step)
        
        img_seq = torchvision.utils.make_grid(torch.stack(imgs))
        writer.add_image('image', img_seq, step)
        
        if 'all_x_means' in model_outs:
            all_samples = torch.sum(model_outs['all_x_means'] * model_outs['all_masks'], 2)
            all_samples = torchvision.utils.make_grid(all_samples[0], nrow=10)
            writer.add_image('samples', all_samples, step)
