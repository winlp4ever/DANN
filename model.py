# main model, includes all modules and how info flows within, forward and backward
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model(object):
    def __init__(self):
        super(Model,self).__init__()
