import numpy as np
import struct
import gzip
from scipy.io import loadmat
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
import utils
import os
from PIL import Image


class MyData(Dataset):
    def __init__(self, func, filename, transform, *args, **kwargs):
        self.data = func(filename, *args, **kwargs)
        self.transform = transform

    def __len__(self):
        if isinstance(self.data, tuple):
            return self.data[0].shape[0]
        return self.data.shape[0]

    def __getitem__(self, idx):
        try:
            return self.transform(self.data[idx])
        except Exception:
            if isinstance(self.data, tuple):
                return tuple(d[idx] for d in self.data)
            return self.data[idx]


class CombinedData(Dataset):
    def __init__(self, datasets, consts=()):
        for d in datasets:
            assert isinstance(d, MyData)
        self.datasets = datasets
        self.consts = consts

    def __len__(self):
        return min(*(len(d) for d in self.datasets))

    def __getitem__(self, idx):
        l = ()
        for d in self.datasets:
            if isinstance(d[idx], tuple):
                l += d[idx]
            else:
                l += (d[idx],)
        return l + self.consts


def svhnToMnist(mnist_fn_im, svhn_fn):
    mnist = MyData(utils.read_idx, mnist_fn_im, lambda t: torch.cat([t] * 3))
    svhn = MyData(utils.read_mat, svhn_fn, None, im_size=28)
    return CombinedData(datasets=(mnist,svhn), consts=(1., 0.))


def test(keyword='train'):
    mnist_fn = os.path.join('./data/mnist', keyword)
    svhn_fn = os.path.join('./data/svhn', keyword)
    data = svhnToMnist(mnist_fn, svhn_fn)
    print(len(data))
    print(type(data[0][0]))
    print(data[0][0].size())

if __name__ == '__main__':
    test()
