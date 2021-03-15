import numpy as np
import torch
from torch.utils.data import Dataset
import tables


class MAGICDatasetLORAKS(Dataset):
    """Dataloader that Load & Augment & change to Torch Tensors (For Training)
    Inputs : path to Dataset (.h5)
    Outputs : [X_JLORAKS, Y_JLORAKS, Sens, Y_kJLORAKS, mask]"""

    def __init__(
        self, h5_path, augment_flipud, augment_fliplr, augment_scale, verbosity=False
    ):
        self.fname = h5_path
        self.tables = tables.open_file(self.fname)
        print(self.tables)
        self.nslices = self.tables.root.X_JLORAKS.shape[0]
        self.tables.close()
        self.X_JLORAKS = None  # reconstructed images from JLORAKS
        self.Y_JLORAKS = None  # Fully sampled images
        self.Sens = None  # Sensitivity maps
        self.Y_kJLORAKS = None  # Acquired K-space
        self.mask = None
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

        if self.X_JLORAKS is None:  # Open in thread
            self.tables = tables.open_file(self.fname, "r")
            self.X_JLORAKS = self.tables.root.X_JLORAKS
            self.Y_JLORAKS = self.tables.root.Y_JLORAKS
            self.Sens = self.tables.root.Sens
            self.Y_kJLORAKS = self.tables.root.Y_kJLORAKS
            self.mask = self.tables.root.mask

        #        t0 = time.time()
        """
        X_JLORAKS = np.float32(np.array(self.X_JLORAKS[ind]))
        Y_JLORAKS = np.float32(np.array(self.Y_JLORAKS[ind]))
        Sens = np.float32(np.array(self.Sens[ind]))
        Y_kJLORAKS = np.float32(np.array(self.Y_kJLORAKS[ind]))
        """
        X_JLORAKS = np.float32(self.X_JLORAKS[ind])
        Y_JLORAKS = np.float32(self.Y_JLORAKS[ind])
        Sens = np.float32(self.Sens[ind])
        Y_kJLORAKS = np.float32(self.Y_kJLORAKS[ind])
        mask = np.float32(self.mask[ind])
        #        '''

        #        print(time.time()-t0)

        """ Data Loading """
        # mask = np.float32(np.abs(Y_kJLORAKS[:, 0:1, 0:1]) > 0)
        #        Sens = np.tile(Sens,[1,8,1,1,1])

        if self.verbose:
            print("X_JLORAKS:", X_JLORAKS.shape, X_JLORAKS.dtype)
            print("Y_JLORAKS:", Y_JLORAKS.shape, Y_JLORAKS.dtype)
            print("Sens:", Sens.shape, Sens.dtype)
            print("Y_kJLORAKS:", Y_kJLORAKS.shape, Y_kJLORAKS.dtype)
            print("mask:", mask.shape, mask.dtype)

        """ Augmentation (Random Flipping (None, Left-Right, Up-Down), Scaling (0.9 - 1.1) """
        if self.augment_flipud:
            """ Random Flipping """
            if np.random.random() < 0.5:
                #                pdb.set_trace()
                X_JLORAKS = self.AugmentFlip(X_JLORAKS, 1)
                Y_JLORAKS = self.AugmentFlip(Y_JLORAKS, 1)
                Sens = self.AugmentFlip(Sens, 2)
                Y_kJLORAKS = self.AugmentFlip(Y_kJLORAKS, 2)
                mask = self.AugmentFlip(mask, 2)

        if self.augment_fliplr:
            if np.random.random() < 0.5:
                X_JLORAKS = self.AugmentFlip(X_JLORAKS, 2)
                Y_JLORAKS = self.AugmentFlip(Y_JLORAKS, 2)
                Sens = self.AugmentFlip(Sens, 3)
                Y_kJLORAKS = self.AugmentFlip(Y_kJLORAKS, 3)
                mask = self.AugmentFlip(mask, 3)

        if self.augment_scale:
            scale_f = np.random.uniform(0.9, 1.1)
            X_JLORAKS = self.AugmentScale(X_JLORAKS, scale_f)
            Y_JLORAKS = self.AugmentScale(Y_JLORAKS, scale_f)
            Y_kJLORAKS = self.AugmentScale(Y_kJLORAKS, scale_f)

        return (
            torch.from_numpy(X_JLORAKS),
            torch.from_numpy(Y_JLORAKS),
            torch.from_numpy(Sens),
            torch.from_numpy(Y_kJLORAKS),
            torch.from_numpy(mask),
        )

    def __len__(self):
        return self.nslices
