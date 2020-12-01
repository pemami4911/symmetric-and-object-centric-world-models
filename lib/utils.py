import torch
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from scipy.stats import truncnorm


def truncated_normal_initializer(shape, mean, stddev):
    # compute threshold at 2 std devs
    values = truncnorm.rvs(mean - 2 * stddev, mean + 2 * stddev, size=shape)
    return torch.from_numpy(values).float()

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Modified from: https://github.com/baudm/MONet-pytorch/blob/master/models/networks.py

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'truncated_normal':
                m.weight.data = truncated_normal_initializer(m.weight.shape, 0.0, stddev=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def _softplus_to_std(softplus):
    softplus = torch.min(softplus, torch.ones_like(softplus)*80)
    return torch.sqrt(torch.log(1. + softplus.exp()) + 1e-5)

def mvn(loc, softplus, temperature=1.0):
    return torch.distributions.independent.Independent(
            torch.distributions.normal.Normal(loc, _softplus_to_std(softplus) * (1./temperature)), 1)

def std_mvn(shape, device):
    loc = torch.zeros(shape).to(device)
    scale = torch.ones(shape).to(device)
    return torch.distributions.independent.Independent(
            torch.distributions.normal.Normal(loc, scale), 1)


def gmm_loglikelihood(x_t, x_loc, log_var, mask_logprobs):
    """
    mask_logprobs: [N, K, 1, H, W]
    """
    # NLL [batch_size, 1, H, W]
    sq_err = (x_t.unsqueeze(1) - x_loc).pow(2)
    # log N(x; x_loc, log_var): [N, K, C, H, W]
    normal_ll = -0.5 * log_var - 0.5 * (sq_err / torch.exp(log_var))
    # [N, K, C, H, W]
    log_p_k = (mask_logprobs + normal_ll)
    # logsumexp over slots [N, C, H, W]
    log_p = torch.logsumexp(log_p_k, dim=1)
    # [batch_size]
    nll = -torch.sum(log_p, dim=[1,2,3])

    #upper_bound = torch.logsumexp(mask_logprobs - 0.5 * (sq_err / torch.exp(log_var)), dim=1)  # [N, C, H, W]
    #upper_bound = torch.sum(upper_bound, dim=[1,2,3])  # [N]

    return nll, {'log_p_k': log_p_k, 'normal_ll': normal_ll, 'ub': None}

def gaussian_loglikelihood(x_t, x_loc, log_var):
    sq_err = (x_t - x_loc).pow(2)  # [N,C,H,W]
    # log N(x; x_loc, log_var): [N,C, H, W]
    normal_ll = -0.5 * log_var - 0.5 * (sq_err / torch.exp(log_var))
    nll = -torch.sum(normal_ll, dim=[1,2,3])   # [N]
    return nll


def create_video(means, masks, sum_over_k=True):
    T = len(means)
    frames = []
    if len(means[0].shape) == 6:
        k_dim = 2
    else:
        k_dim = 1
    for i in range(T):
        if sum_over_k:
            frame = torch.sum(means[i].detach() * masks[i].detach(), k_dim)  # [batch_size, rollouts, C, H, W]
        else:
            frame = (means[i] * masks[i])  + (1 - masks[i]) * torch.ones(means[i].shape).to(means[i].device)# [batch_size, rollouts, K, C, H, W]
        frames += [frame]
    video = torch.stack(frames)  # [seq_len, batch_size, rollouts, [K], C, H, W]
    return video

def rename_state_dict(state_dict, old_strings, new_strings):
    new_state_dict = {}
    for old_string, new_string in zip(old_strings, new_strings):
        for k,v in state_dict.items():
            if old_string in k:
                new_key = k.replace(old_string, new_string)
                new_state_dict[new_key] = v
    for k,v in state_dict.items():
        for old_string in old_strings:
            if old_string in k:
                break
        else:
            new_state_dict[k] = v
    return new_state_dict

def get_log_var_grad(val_dataloader, model, geco, seq_len, aggregate_over=20):
    """
    assume batch_size is 1, compu 
    """
    opt = torch.optim.SGD(model.parameters(), lr=1)
    model.train()

    num_grads = 0
    for p in model.module.relational_dynamics.parameters():
        if p.requires_grad:
            num_grads += 1

    grads = [[] for i in range(num_grads)]
    logvargrads = [[] for i in range(num_grads)]

    for i,batch in enumerate(val_dataloader):
        imgs = batch['imgs'].to('cuda')
        imgs = imgs[:,:seq_len]

        model_outs = model(imgs, geco, i, None)

        opt.zero_grad()
        total_loss = model_outs['total_loss']  # [1] for sample size 1
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)

        # for each parameter, flatten and store
        j = 0
        for p in model.module.relational_dynamics.parameters():
            if p.requires_grad:
                grads[j] += [p.grad.view(-1)]
                j += 1

        if len(grads[0]) == aggregate_over:
            print('aggregating at {}'.format(i))
            for j,g in enumerate(grads):
                all_grad = torch.stack(g).data.cpu().numpy()  # [aggregate, dim]
                logvargrads[j] += [np.mean(np.log(np.var(all_grad, 1)+1e-6))]
            # reset
            grads = [[] for i in range(num_grads)]

    lvg_ = 0
    for lvg in logvargrads:
        lvg_ += np.mean(lvg)
    lvg_ = lvg_ / len(logvargrads)
    return lvg_
