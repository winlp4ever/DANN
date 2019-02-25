import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function


class GradReverse(Function):
    def __init__(self, lbda):
        super(GradReverse, self).__init__()
        self.lbda = lbda

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return grad_output * -self.lbda


def grad_reverse(x, l):
    return GradReverse(l)(x)


class FeaturesExtractor(nn.Module):
    def __init__(self):
        super(FeaturesExtractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 48, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.extractor(x)


class DomainClassifier(nn.Module):
    def __init__(self, in_features, init_weight=True):
        # in features: to be fixed
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2),
            nn.LogSoftmax()
        )

    def forward(self, x, lbda):
        x = grad_reverse(x, lbda)
        return self.classifier(x) # already take log, for binary entropy loss function


class ClassClassifier(nn.Module):
    def __init__(self, in_features):
        super(ClassClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 3072),
            nn.ReLU(True),
            nn.Linear(3072, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 10),
            nn.LogSoftmax()
        )

    def forward(self, x):
        return self.classifier(x)


class GradNet(nn.Module):
    def __init__(self, init_weight):
        super(GradNet, self).__init__()
        self.E = FeaturesExtractor()
        in_features = 48 * 8 * 8
        self.G_d = DomainClassifier(in_features)
        self.G_c = ClassClassifier(in_features)
        if init_weight:
            self._init_weight()

    def forward(self, x, lbda=None, clss=False):
        x = self.E(x)
        x = x.view(x.size(0), -1)
        if lbda is None:
            return self.G_c(x)
        if clss:
            return self.G_c(x), self.G_d(x, lbda)
        return self.G_d(x, lbda)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
