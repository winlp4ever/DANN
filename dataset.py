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
    def __init__(self, mnist_fn, svhn_fn, classes=10, im_size=28, mnist_only=False, mnist_label_fn=''):
        super(GradData, self).__init__()
        self.im_size = im_size
        self.classes = classes
        self.mnist_size, self.mnist_im = utils.read_idx(mnist_fn)
        if mnist_only:
            self.mnist_label = utils.read_idx(mnist_label_fn, image=False)
        else:
            self.svhn_size, self.svhn_im, self.svhn_label = utils.read_mat(svhn_fn)
        self.mnist_only = mnist_only

    def __len__(self):
        return self.mnist_size

    def __getitem__(self, idx):

        t_im = self.mnist_im[idx]
        t_im = torch.cat([t_im] * 3, 0)
        if not self.mnist_only:
            d_im = self.svhn_im[idx]
            d_y = self.svhn_label[idx]
            return t_im, d_im, d_y, 1., 0.
        t_y = self.mnist_label[idx]
        return t_im, t_y


def test(keyword='train'):
    mnist_fn = os.path.join('./data/mnist', keyword)
    svhn_fn = os.path.join('./data/svhn', keyword)
    data = GradData(mnist_fn, svhn_fn)
    print(len(data))
    print(type(data[0][0]))

    rand_id = np.random.randint(len(data))
    d, im = data[rand_id]
    print(d)
    if d:
        print('label {}'.format(im[1]))
        im = im[0]
    print(im.size())
    print(np.amax(im.numpy()))
    im = transforms.ToPILImage()(im)
    im.show()


if __name__ == '__main__':
    test()
