from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import torch, torchvision
from torchvision import transforms, utils
import argparse
import os
import random
from torch.utils.data import DataLoader

import utils
from model import Model
from dataset import svhnToMnist, MyData, CombinedData


try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

url = 'https://www.dropbox.com/s/9ohsf5525jfsw0o/checkpoint-99.pth.tar?dl=1'

ckpt_dir = 'checkpoints'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

fpath = os.path.join(ckpt_dir, 'checkpoint-99.pth.tar')

if os.path.exists(fpath):
    print("{} already exists".format(fpath))
else:
    print('downloading checkpoint ...')
    urlretrieve(url, filename=fpath)

mnist_test = './data/mnist/test'
mnist_test_label = './data/mnist/test_label'
svhn_test = './data/svhn/test'

use_cuda = torch.cuda.is_available()
torch.manual_seed(random.randint(0, 10000))
device = torch.device("cuda" if use_cuda else "cpu")

mnist_test_im = MyData(utils.read_idx, mnist_test, lambda t: torch.cat([t] * 3))
mnist_test_label = MyData(utils.read_idx, mnist_test_label, None, image=False)
mnist_test = CombinedData((mnist_test_im, mnist_test_label))
svhn_test =  MyData(utils.read_mat, svhn_test, None)

mnist_test_loader = DataLoader(mnist_test, batch_size=1000, shuffle=True)
svhn_test_loader = DataLoader(svhn_test, batch_size=1000, shuffle=True)
model = Model(device, 0.01, 0.9)
model.load_checkpoint(fpath)

model.test_epoch(device, mnist_test_loader, None, 'Target: mnist')
model.test_epoch(device, svhn_test_loader, None, 'Source: svhn')
