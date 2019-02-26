# download all datasets, preprocess if necessary
from __future__ import print_function

import os

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

mnist_train = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
mnist_test = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
mnist_test_label = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'


def download_mnist(output_dir='data', mnist_fol='mnist'):
    filenames = ['train', 'test', 'test_label']
    urls = [mnist_train, mnist_test, mnist_test_label]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    path = os.path.join(output_dir, mnist_fol)
    if not os.path.exists(path):
        os.mkdir(path)

    # notfound = []
    for url, filename in zip(urls, filenames):
        output_file = os.path.join(path, filename)

        if os.path.exists(output_file):
            print("{} already exists".format(output_file))
            continue

        print("Downloading from {} ...".format(url))
        urlretrieve(url, filename=output_file)
        print("=> File saved as {}".format(output_file))
