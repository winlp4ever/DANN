import struct as st
import gzip
import numpy as np
from scipy.io import loadmat
import torch, torchvision
import torch.nn.functional as F

def read_idx(fn, image=True):
    with gzip.open(fn, 'r') as f:
        magic, nb_imgs = st.unpack(">II", f.read(8))
        if image:
            nrows, ncols = st.unpack(">II", f.read(8))
            buf = f.read(nb_imgs * nrows * ncols)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(nb_imgs, 1, nrows, ncols)
            data = torch.from_numpy(data).float()
            data = F.interpolate(data, size=(32, 32), mode='nearest')
            return nb_imgs, data
        buf_ = f.read(nb_imgs)
        labels = np.frombuffer(buf_, dtype=np.uint8)
        return labels

def read_mat(fn):
    data = loadmat(fn)
    X = np.transpose(data['X'], (3, 2, 0, 1))
    X = torch.from_numpy(X).float()
    y = data['y']
    y[y == 10] = 0
    size = X.shape[0]
    return size, X, y

def one_hot_encoding(y, classes=10):
    return np.eye(classes)[y]
