from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import torch, torchvision
from torchvision import transforms, utils
import argparse
import os
import random

from model import Model

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

parser = argparse.ArgumentParser()
parser.add_argument('--filename', '-f', help='image filename in dir ./images', nargs='?', default='test.jpg')
parser.add_argument('--ckpt-path', '-c', help='checkpoint path', nargs='?', default='./checkpoints/checkpoint-99.pth.tar')
args = parser.parse_args()

img = Image.open(os.path.join('images', args.filename)).convert('RGBA')

width, height = img.size

use_cuda = torch.cuda.is_available()
torch.manual_seed(random.randint(0, 10000))
device = torch.device("cuda" if use_cuda else "cpu")

model = Model(device, 1e-2, 0.9)
model.load_checkpoint(args.ckpt_path)

def transform(im, im_size=28):
    trans = transforms.Compose([transforms.Resize((im_size, im_size)),
                                transforms.ToTensor()])
    tens = trans(im)
    if tens.shape[0] > 3:
        tens = tens[:3]
    tens = tens.view(1, *tens.size())
    return tens

t = transform(img)
c = model.net(t.to(device), classify=True)

pred = c.max(1, keepdim=True)[1]
img = transforms.Resize((300, 300))(img)
draw = ImageDraw.Draw(img)
draw.rectangle(((0, 0), (50, 50)), fill='black')
draw.text((20, 20), str(pred.item()), (255, 255, 255), font=ImageFont.truetype('FreeMono.ttf', 20))
img.show()
