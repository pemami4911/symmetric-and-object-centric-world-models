import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def adjusted_rand_index(true_mask, pred_mask):
    """
    compute the ARI for a single image. N.b. ARI 
    is invariant to permutations of the cluster IDs.

    See https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index.

    true_mask: LongTensor of shape [N, C, H, W]
        background == 0
        object 1 == 1
        object 2 == 2
        ...
    pred_mask: FloatTensor of shape [N, K, 1, H, W]  (mask probs)

    Returns: ari [N]
    """
    
    N, C, H, W = true_mask.shape
    # only need one channel
    true_mask = true_mask[:,0]  # [N, H, W]
    # convert into oh  [N, num_points, max_num_entities]
    true_group_ids = true_mask.view(N, H * W).long()
    true_group_oh = torch.nn.functional.one_hot(true_group_ids).float()
    # exclude background
    true_group_oh = true_group_oh[...,1:]
    max_num_entities = true_group_oh.shape[-1]

    # take argmax across slots for predicted masks
    pred_mask = pred_mask.squeeze(2)  # [N, K, H, W]
    pred_groups = pred_mask.shape[1]
    pred_mask = torch.argmax(pred_mask, dim=1)  # [N, H, W]
    pred_group_ids = pred_mask.view(N, H * W).long()
    pred_group_oh = torch.nn.functional.one_hot(pred_group_ids, pred_groups).float()
    
    if max_num_entities == 1 and pred_groups == 1:
        return 1.

    n_points = H*W
    if n_points <= max_num_entities and n_points <= pred_groups:
        raise ValueError(
                "adjusted_rand_index requires n_groups < n_points. We don't handle "
                "the special cases that can occur when you have one cluster "
                "per datapoint")

    nij = torch.einsum('bji,bjk->bki', pred_group_oh, true_group_oh)
    a = torch.sum(nij, 1)
    b = torch.sum(nij, 2)

    rindex = torch.sum(nij * (nij - 1), dim=[1,2])
    aindex = torch.sum(a * (a - 1), 1)
    bindex = torch.sum(b * (b - 1), 1)
    expected_rindex = aindex * bindex / (n_points*(n_points-1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

    return ari

def pixel_mse(true_image, colors, masks):
    """
    true_image is a FloatTensor of shape [N, C, H, W]
    colors is a FloatTensor of shape [N, K, C, H, W]
    masks is a FloatTensor of shape [N, K, 1, H, W]
    """
    pred_image = torch.sum(masks * colors, 1)
    mse = torch.mean((true_image - pred_image) ** 2, dim=[1,2,3])
    return mse
    

def matching_iou(true_mask, pred_mask, video=False):
    """
    true_mask: [batch_size or seq_len,C,H,W] each pixel has a label for its assigned object instance
    pred_mask: [batch_size or seq_len,K,1,H,W]
    """
    N,C, H, W = true_mask.shape
    pred_mask = pred_mask.squeeze(2)
    _,K,_,_ = pred_mask.shape
    pred_mask_ = pred_mask.clone()
    pred_mask = torch.argmax(pred_mask, dim=1)  # [N,H, W]
    pred_mask = pred_mask.view(N,H*W).long()
    true_mask = true_mask[:,0]  # [N, H, W]
    true_object_pixels = true_mask.view(N,H*W).long()
    pixel_ids = torch.arange(H*W)
    num_objects = true_object_pixels.max() # [N]
    iou = torch.zeros(N,num_objects,K)

    if video:
        true_video = torch.zeros(N,num_objects+1,H,W)
        for i in range(1,num_objects+1):
            true_video[:,i][true_mask == i] = 1

    for batch_id in range(N):
        for i in range(1,num_objects+1):
            true_object = pixel_ids[(true_object_pixels[batch_id] == i)]
            for j in range(K):
                pred_object = pixel_ids[(pred_mask[batch_id] == j)]
                intersection = np.intersect1d(true_object, pred_object).shape[0]
                union = len(torch.cat([true_object,pred_object]).unique(sorted=False))
                if union == 0:
                    iou[batch_id,i-1,j] = 0.
                else:
                    iou[batch_id,i-1,j] = (intersection / union)
    # NN matching
    # take max over columns (per groundtruth object) and average
    best_iou = torch.max(iou, dim=2)[0]  # [N,num_objects]
    #mean_iou = torch.mean(best_iou,1)  # [N]
    AR = []
    thresholds = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for t in thresholds:
        TP = (best_iou >= t).sum(1)  # [N]
        # ground truth masks that went unmatched
        FN = (best_iou < t).sum(1)  # [N]
        recall = TP.float() / (TP + FN).float()  # [N]
        AR += [recall]
    AR = torch.mean(torch.stack(AR),0)  # [N]

    if video:
        # align the nn matches and concat together to make a visualiation
        best_idxs = torch.max(iou, dim=2)[1]  # [N,num_objects]
        pred_video = torch.zeros(N,num_objects+1,H,W) 
        pred_video[:,0] = pred_mask_[:,0]
        for i in range(1,num_objects+1):
            pred_video[:,i] = torch.stack([pred_mask_[_, best_idxs[_,i-1]] for _ in range(N)])
        out_video = torch.cat([true_video, pred_video], 2)
        return AR, out_video
    return AR


def compute_video_ar(ground_truth_mask_video, generated_mask_video, create_iou_video=False):
    """
    ground_truth_mask_video is Tensor of shape [seq_len, C, H, W]
    generated_videos is Tensor of shape [num_rollouts, seq_len, K, 1, H, W]

    videos are in FloatTensors in [0-1]
    """
    num_rollouts, seq_len, _, _, _, _ = generated_mask_video.shape
    
    all_ar = [[] for _ in range(seq_len)]
    for i in range(num_rollouts):
        # [seq_len]
        if create_iou_video:
            seq_iou, iou_video = \
                    matching_iou(ground_truth_mask_video, generated_mask_video[i], create_iou_video)
        else:
            seq_iou = matching_iou(ground_truth_mask_video, generated_mask_video[i], create_iou_video)
        seq_iou = seq_iou.data.cpu().numpy()
        for j in range(seq_len):
            all_ar[j] += [seq_iou[j]]
        
    # best/random/worst/best-worst AR summed over seq_len - video and per-frame AR
    # mean & std per-frame AR across all rollouts
    all_ar = np.array(all_ar)  # [seq_len, num_rollouts]
    metrics = {'ar': all_iou}

    stats = {}
    for metric in ['ar']:
        stats[metric] = {}
        for tier in ['best', 'random', 'worst']:
            stats[metric][tier] = {}
            
            sum_over_frames_metric = np.sum(metrics[metric], 0)

            if tier == 'best':
                idx = np.argmax(sum_over_frames_metric)
            elif tier == 'random':
                idx = np.random.randint(0,num_rollouts)
            elif tier == 'worst':
                idx = np.argmin(sum_over_frames_metric)
            
            stats[metric][tier]['per_frame'] = metrics[metric][:,idx]
        stats[metric]['mean'] = np.mean(metrics[metric], 1)  # [seq_len]
        stats[metric]['std'] = np.std(metrics[metric], 1)  # [seq_len]

    if create_iou_video:
        return stats, iou_video
    else:
        return stats, None
    

def compute_video_psnr_and_ssim(ground_truth_video, generated_video, disable_video=False):
    """
    ground_truth_video is Tensor of shape [seq_len, C, H, W]
    generated_videos is Tensor of shape [num_rollouts, seq_len, C, H, W]

    videos are in FloatTensors in [0-1]
    """
    if isinstance(ground_truth_video, torch.Tensor):
        ground_truth_video = ground_truth_video.data.cpu().numpy()
    if isinstance(generated_video, torch.Tensor):
        generated_video = generated_video.data.cpu().numpy()
    ground_truth_video = ground_truth_video.transpose(0,2,3,1)
    generated_video = generated_video.transpose(0,1,3,4,2)
    num_rollouts, seq_len, _, _, _ = generated_video.shape
    
    all_psnr = [[] for _ in range(seq_len)]
    all_ssim = [[] for _ in range(seq_len)]
    for i in range(num_rollouts):
        for j in range(seq_len):
            image_true = ground_truth_video[j]  # [C,H,W]
            image_test = generated_video[i,j]
            all_psnr[j] += [psnr(image_true, image_test)]
            all_ssim[j] += [ssim(image_true, image_test, multichannel=True, gaussian_weights=True)]
    # stats

    # best/random/worst psnr/ssim summed over seq_len - video and per-frame psnr/ssim
    # mean & std per-frame psnr across all rollouts
    # mean & std per-frame ssim across all rollouts
    all_psnr = np.array(all_psnr)  # [seq_len, num_rollouts]
    all_ssim = np.array(all_ssim)  # [seq_len, num_rollouts]
    metrics = {'psnr': all_psnr, 'ssim': all_ssim}

    stats = {}
    for metric in ['psnr', 'ssim']:
        stats[metric] = {}
        for tier in ['best', 'random', 'worst']:
            stats[metric][tier] = {}
            
            sum_over_frames_metric = np.sum(metrics[metric], 0)

            if tier == 'best':
                idx = np.argmax(sum_over_frames_metric)
            elif tier == 'random':
                idx = np.random.randint(0,num_rollouts)
            elif tier == 'worst':
                idx = np.argmin(sum_over_frames_metric)
            
            if not disable_video:
                stats[metric][tier]['video'] = generated_video[idx]
            stats[metric][tier]['per_frame'] = metrics[metric][:,idx]
        stats[metric]['mean'] = np.mean(metrics[metric], 1)  # [seq_len]
        stats[metric]['std'] = np.std(metrics[metric], 1)  # [seq_len]

    return stats


if __name__ == '__main__':

    true_mask = torch.LongTensor([[1, 1, 1],[0,2,2],[0,0,0]])
    true_mask = true_mask.view(1,1,3,3)

    print(true_mask)

    pred_mask = torch.FloatTensor([[[0., 0., 1.], [1., 0., 0],[1,1,1]],
            [[1., 1., 0.], [0,0,0], [0,0,0]],
            [[0,0,0], [0,1,1], [0,0,0]]])

    pred_mask = pred_mask.view(1,3,1,3,3)

    print(pred_mask)

    print(adjusted_rand_index(true_mask, pred_mask))
