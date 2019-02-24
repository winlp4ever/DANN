# download all datasets, preprocess if necessary
from __future__ import print_function

import os

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

train_image_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
test_image_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
train_label_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
test_label_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'


def main(output_dir='data'):
    filenames = ['train_img_ubyte', 'test_img_ubyte', 'train_label_ubyte', 'test_label_ubyte']
    urls = [train_image_url, test_image_url, train_label_url, test_label_url]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # notfound = []
    for url, filename in zip(urls, filenames):
        output_file = os.path.join(output_dir, filename)

        if os.path.exists(output_file):
            print("{} already exists".format(output_file))
            continue

        print("Downloading from {} ...".format(url))
        urlretrieve(url, filename=output_file)
        print("=> File saved as {}".format(output_file))


if __name__ == '__main__':
    main()
