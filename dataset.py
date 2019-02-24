import numpy as np
import struct
import gzip
from scipy.io import loadmat
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import utils


class GradData(Dataset):
    def __init__(self, mnist_fn, svhn_fn, im_size=32):
        supr(GradData, self).__init__()
        self.mnist_size, self.mnist_im = utils.read_idx(mnist_fn + 'im')
        self.mnist_label = utils.read_idx(mnist_fn + 'label')
        self.svhn_size, self.svhn_im, self.svhn_label = utils.read_mat(svhn_fn)
        self.transform = transforms.Compose([transforms.Resize((height, width)),
                                            transforms.ToTensor()])

    def __len__(self):
        return self.mnist_size + self.svhn_size

    def __getitem__(self, idx):
        if idx < self.mnist_size:
            im = self.mnist_im[idx]
            im = np.transpose([im] * 3, (1, 2, 0))
            y = utils.one_hot_encoding(self.mnist_im[idx])
            return self.transform(im), y, 0
        im = self.svhn_im[idx - self.mnist_size]
        y = utils.one_hot_encoding(self.shvn_label[idx - self.mnist_size])
        return self.transform(im), y, 1



dat = loadmat('data/train_32x32.mat')
print(dat['X'].shape)
print(dat['X'][:, :, :, 0])
print(dat['y'].shape)

print(data.shape)
print('')
#print(data[0,:,:])
plt.imshow(data[0,:,:], cmap='gray')
plt.show()
print(labels[:10])
