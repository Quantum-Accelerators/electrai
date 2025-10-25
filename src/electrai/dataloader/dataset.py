from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


# !!! Reading in all the data at once is probably not a good idea, we better just read in the filenames, and have __getitem__ read each index in instead.
# The could should be updates as such.
class RhoData(Dataset):
    def __init__(
        self,
        list_data,
        list_label,
        list_data_gridsizes,
        list_label_gridsizes,
        data_augmentation=True,
        downsample_data=1,
        downsample_label=1,
    ):
        """
        list_data: list of flattened (input) rho data of size batchsize. Each element of size gridsize_x * gridsize_y * gridsize_z.
        list_label: list of flattened (target) rho data of size batchsize.
        list_data_gridsizes: list of gridsizes of size batchsize.  Each element of size 3.
        list_label_gridsizes: list of gridsizes of size batchsize * 3.
        """
        self.ds_data = downsample_data
        self.ds_label = downsample_label
        self.da = data_augmentation

        self.data = list_data
        self.data_gs = list_data_gridsizes
        self.label = list_label
        self.label_gs = list_label_gridsizes

        assert len(self.data) == len(self.data_gs)
        assert len(self.data) == len(self.label)
        assert len(self.data) == len(self.label_gs)

    def __len__(self):
        return len(self.data)

    def rotate_x(self, data_in):
        """
        rotate 90 by x axis
        """
        return data_in.transpose(-1, -2).flip(-1)

    def rotate_y(self, data_in):
        return data_in.transpose(-1, -3).flip(-1)

    def rotate_z(self, data_in):
        return data_in.transpose(-2, -3).flip(-2)

    def rand_rotate(self, data_lst):
        rint = np.random.randint(3)
        if rint == 0:

            def rotate(d):
                return self.rotate_x(d)
        elif rint == 1:

            def rotate(d):
                return self.rotate_y(d)
        else:

            def rotate(d):
                return self.rotate_z(d)

        r = np.random.rand()
        if r < 0.1:
            return data_lst
        elif r < 0.4:
            return [rotate(d) for d in data_lst]
        elif r < 0.7:
            return [rotate(rotate(d)) for d in data_lst]
        else:
            return [rotate(rotate(rotate(d))) for d in data_lst]

    def __getitem__(self, idx):
        rho1 = torch.tensor(self.data[idx], dtype=torch.float32)
        size = torch.tensor(self.data_gs[idx], dtype=int)
        rho1 = rho1.reshape(1, *size)

        rho2 = torch.tensor(self.label[idx], dtype=torch.float32)
        size = torch.tensor(self.label_gs[idx], dtype=int)
        rho2 = rho2.reshape(1, *size)

        if self.da:
            rho1, rho2 = self.rand_rotate([rho1, rho2])

        ds1 = self.ds_data
        ds2 = self.ds_label
        nx, ny, nz = rho1.size()[-3:]
        nx = nx // ds1 * ds1
        ny = ny // ds1 * ds1
        nz = nz // ds1 * ds1
        rho1 = rho1[..., :nx:ds1, :ny:ds1, :nz:ds1]
        nx, ny, nz = rho2.size()[-3:]
        nx = nx // ds1 * ds1
        ny = ny // ds1 * ds1
        nz = nz // ds1 * ds1
        rho2 = rho2[..., :nx:ds2, :ny:ds2, :nz:ds2]

        return rho1, rho2
