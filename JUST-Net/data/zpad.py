import numpy as np
import torch
from torch.utils.data import Dataset
import tables

from ..utils import ifft2c, combine_all_coils


class MAGICDatasetZpad(Dataset):
    """Dataloader that Load & Augment & change to Torch Tensors (For Training)
    Inputs : path to Dataset (.h5)
    Outputs : [X_JLORAKS, Y_JLORAKS, Sens, X_kJLORAKS, mask]"""

    def __init__(self, h5_path, augment_flipud, augment_fliplr, augment_scale, verbosity=False):
        self.fname = h5_path
        self.tables = tables.open_file(self.fname)
        self.nslices = self.tables.root.X_JLORAKS.shape[0]
        self.tables.close()
        self.Y_JLORAKS = None  # Fully sampled images
        self.Sens = None  # Sensitivity maps
        self.augment_flipud = augment_flipud
        self.augment_fliplr = augment_fliplr
        self.augment_scale = augment_scale
        self.verbose = verbosity  # Print out what is going on?

    def AugmentFlip(self, im, axis):
        im = im.swapaxes(0, axis)
        im = im[::-1]
        im = im.swapaxes(axis, 0)
        return im.copy()

    def AugmentScale(self, im, scale_val):
        im = im * scale_val
        return im

    def __getitem__(self, ind):

        if self.Y_JLORAKS is None:  # Open in thread
            self.tables = tables.open_file(self.fname, "r")
            self.Y_JLORAKS = self.tables.root.Y_JLORAKS
            self.Sens = self.tables.root.Sens
            self.X_kJLORAKS = self.tables.root.X_kJLORAKS

        Y_JLORAKS = np.float32(self.Y_JLORAKS[ind])
        Sens = np.float32(self.Sens[ind])
        X_kJLORAKS = np.float32(self.X_kJLORAKS[ind])

        """ Data Loading """
        mask = np.float32(np.abs(X_kJLORAKS[:, 0:1, 0:1]) > 0)

        """ Augmentation (Random Flipping (None, Left-Right, Up-Down), Scaling (0.9 - 1.1) """
        if self.augment_flipud:
            """ Random Flipping """
            if np.random.random() < 0.5:
                #                pdb.set_trace()
                Y_JLORAKS = self.AugmentFlip(Y_JLORAKS, 1)
                Sens = self.AugmentFlip(Sens, 2)
                X_kJLORAKS = self.AugmentFlip(X_kJLORAKS, 2)
                mask = self.AugmentFlip(mask, 2)

        if self.augment_fliplr:
            if np.random.random() < 0.5:
                Y_JLORAKS = self.AugmentFlip(Y_JLORAKS, 2)
                Sens = self.AugmentFlip(Sens, 3)
                X_kJLORAKS = self.AugmentFlip(X_kJLORAKS, 3)
                mask = self.AugmentFlip(mask, 3)

        if self.augment_scale:
            scale_f = np.random.uniform(0.9, 1.1)
            Y_JLORAKS = self.AugmentScale(Y_JLORAKS, scale_f)
            X_kJLORAKS = self.AugmentScale(X_kJLORAKS, scale_f)

        Y = torch.from_numpy(Y_JLORAKS)
        S = torch.from_numpy(Sens)
        X_k = torch.from_numpy(X_kJLORAKS)
        m = torch.from_numpy(mask)

        X = ifft2c(X_k)
        X = combine_all_coils(X, S, 0)
        return X, Y, S, X_k, m

    def __len__(self):
        return self.nslices
