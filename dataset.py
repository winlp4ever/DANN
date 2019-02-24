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
    def __init__(self, mnist_fn, svhn_fn, im_size=32):
        super(GradData, self).__init__()
        self.im_size = im_size
        self.mnist_size, self.mnist_im = utils.read_idx(mnist_fn)
        #self.mnist_label = utils.read_idx(mnist_fn + '_label', image=False)
        self.svhn_size, self.svhn_im, self.svhn_label = utils.read_mat(svhn_fn)

    def __len__(self):
        return self.mnist_size + self.svhn_size

    def __getitem__(self, idx):
        if idx < self.mnist_size:
            im = self.mnist_im[idx]
            im = torch.cat([im] * 3, 0)
            #im = F.upsample(im, size=(self.im_size, self.im_size), mode='nearest')
            return 0, im
        im = self.svhn_im[idx - self.mnist_size]
        y = utils.one_hot_encoding(self.svhn_label[idx - self.mnist_size])
        return 1, (im, y)


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
