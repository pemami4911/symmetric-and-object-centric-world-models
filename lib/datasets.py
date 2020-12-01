import torch
import torchvision
import numpy as np
import h5py
from pathlib import Path
from sacred import Ingredient
import cv2

ds = Ingredient('dataset')

@ds.config
def cfg():
    data_path = ''  # base directory for data
    h5_path = '' # dataset name
    masks = False
    labels = False
    actions = False
    halve_images = False


class SequentialHdF5Dataset(torch.utils.data.Dataset):
    """
    Dataset class for reading seqs of images from an HdF5 file
    """

    @ds.capture
    def __init__(self, data_path, h5_path, masks, actions, halve_images=False, d_set='train'):
        super(SequentialHdF5Dataset, self).__init__()
        self.h5_path = str(Path(data_path, h5_path))
        self.d_set = d_set.lower()
        self.masks = masks
        self.actions = actions
        self.halve_images = halve_images


    def preprocess(self, seq):
        T,H,W,C = seq.shape

        if np.any( seq > 1.0 ):
            seq /= 255.
        
        if self.halve_images:
            seq_ = np.split(seq,T,0)
            seq = np.stack([cv2.resize(_.squeeze(0), (int(W / 2.), int(H / 2.))) for _ in seq_])
        # [T,H,W,C] -> [T,C,H,W]
        seq = np.transpose(seq, (0,3,1,2))
        return seq


    def __len__(self):
        with h5py.File(self.h5_path,  'r') as data:
            _, data_size, _, _, _ = data[self.d_set]['imgs'].shape
            return data_size
    
    def seq_len(self):
        with h5py.File(self.h5_path,  'r') as data:
            seq_len, _, _, _, _ = data[self.d_set]['imgs'].shape
            return seq_len

    def __getitem__(self, i):
        with h5py.File(self.h5_path,  'r') as data:
            outs = {}
            outs['imgs'] = self.preprocess(data[self.d_set]['imgs'][:,i].astype('float32'))
            
            if self.masks:
                outs['masks'] = np.transpose(data[self.d_set]['masks'][:,i].astype('float32'), (0,3,1,2))

            if self.actions:
                outs['actions'] = data[self.d_set]['actions'][:,i].astype('float32')

            return outs

