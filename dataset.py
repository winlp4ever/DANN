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


class GradData(Dataset):
    def __init__(self, mnist_fn, svhn_fn, classes=10, im_size=28, mnist_only=False, svhn_only=False, mnist_label_fn=''):
        super(GradData, self).__init__()
        self.im_size = im_size
        self.classes = classes
        if not svhn_only:
            self.mnist_size, self.mnist_im = utils.read_idx(mnist_fn)
        if mnist_only:
            self.mnist_label = utils.read_idx(mnist_label_fn, image=False)
        else:
            self.svhn_size, self.svhn_im, self.svhn_label = utils.read_mat(svhn_fn)
        self.mnist_only = mnist_only
        self.svhn_only = svhn_only

    def __len__(self):
        if self.svhn_only:
            return self.svhn_size
        if self.mnist_only:
            return self.mnist_size
        return min(self.mnist_size, self.svhn_size)

    def __getitem__(self, idx):
        if not self.svhn_only:
            t_im = self.mnist_im[idx]
            t_im = torch.cat([t_im] * 3, 0)
        if self.mnist_only:
            t_y = self.mnist_label[idx]
            return t_im, t_y
        d_im = self.svhn_im[idx]
        d_y = self.svhn_label[idx]
        if self.svhn_only:
            return d_im, d_y
        return t_im, d_im, d_y, 1., 0.

def test(keyword='train'):
    mnist_fn = os.path.join('./data/mnist', keyword)
    svhn_fn = os.path.join('./data/svhn', keyword)
    data = GradData(mnist_fn, svhn_fn)
    print(len(data))
    print(type(data[0][0]))

    rand_id = np.random.randint(len(data))
    t_im, d_im, d_y, _, _ = data[rand_id]
    print(d)

    print(t_im.size())
    print(np.amax(t_im.numpy()))
    im = transforms.ToPILImage()(t_im)
    im.show()


if __name__ == '__main__':
    test()
