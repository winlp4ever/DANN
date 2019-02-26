# download all datasets, preprocess if necessary
from __future__ import print_function

import os

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

svhn_train = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
svhn_test = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'


def download_svhn(output_dir='data', svhn_fol='svhn'):
    filenames = ['train', 'test']
    urls = [svhn_train, svhn_test]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    path = os.path.join(output_dir, svhn_fol)
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
