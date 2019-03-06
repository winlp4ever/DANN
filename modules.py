import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function


class GradReverse(Function):
    def __init__(self):
        super(GradReverse, self).__init__()

    def _set_lambda(self, lbda):
        self.lbda = lbda

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return grad_output.neg() * self.lbda


class GradNet(nn.Module):
    def __init__(self, init_weight):
        super(GradNet, self).__init__()
        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        in_features = 128 * 7 * 7
        self.grad_revers = GradReverse()
        self.G_d = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2),
            nn.LogSoftmax()
        )
        self.G_c = nn.Sequential(
            nn.Linear(in_features, 3072),
            nn.ReLU(True),
            nn.Linear(3072, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 10),
            nn.LogSoftmax()
        )
        if init_weight:
            self._init_weight()

    def _set_lambda(self, lbda):
        self.grad_revers._set_lambda(lbda)

    def forward(self, x, d_classify=False, classify=False):
        x = self.E(x)
        x = x.view(x.size(0), -1)
        if d_classify:
            y = self.grad_revers(x)
            y = self.G_d(x)
            if classify:
                return self.G_c(x), y
            else:
                return y
        return self.G_c(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
