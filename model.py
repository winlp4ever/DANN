import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from modules import GradNet
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import os


class Model(object):
    def __init__(self, device, lr, momentum):
        super(Model,self).__init__()
        self.net = GradNet(init_weight=True).to(device)
        self.writer = SummaryWriter()
        self.optim = optim.SGD(self.net.parameters(), lr=lr, momentum=momentum, weight_decay=1e-5)
        self.lr = lr

    def train_epoch(self, args, device, train_loader, epoch):
        self.net.train()
        c_loss = 0
        d_loss = 0
        for batch_idx, data in enumerate(train_loader):
            percent = batch_idx / len(train_loader)
            p = (epoch + percent) / args.epochs
            self.lr_update(p, args)
            lbda = self.lbda_update(p, args)

            X_t, X_d, y, d_t, d_d = [a.to(device) for a in data]

            self.optim.zero_grad()
            self.net._set_lambda(lbda)
            pred_d_ = self.net(X_t, True)
            pred_c, pred_d = self.net(X_d, True, True)

            c_l = F.nll_loss(pred_c, y.long())
            d_l = F.nll_loss(pred_d, d_d.long()) + F.nll_loss(pred_d_, d_t.long())
            loss = c_l + d_l

            loss.backward()
            self.optim.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tc_loss: {:.4f}\td_loss: {:.4f}'.format(
                    epoch + 1, args.epochs, batch_idx * len(X_t), len(train_loader.dataset),
                           100. * percent, loss.item(), c_l.item(), d_l.item()), end='\r', flush=True)


            c_loss += c_l.item()
            d_loss += d_l.item()

        print('')
        if (epoch + 1) % args.sv_interval == 0:
            print('saving model ...')
            self.save_checkpoint(args,
                                 {
                                     'epoch': epoch + 1,
                                     'state_dict': self.net.state_dict(),
                                     'optimizer': self.optim.state_dict(),
                                 }, epoch)

        c_loss /= len(train_loader.dataset)
        d_loss /= len(train_loader.dataset)
        self.writer.add_scalar('c_loss', c_loss, global_step=epoch)
        self.writer.add_scalar('d_loss', d_loss, global_step=epoch)


    def save_checkpoint(self, args, state, epoch):
        filename = os.path.join(args.ckpt_path, 'checkpoint-{}.pth.tar'.format(epoch))
        torch.save(state, filename)

    def load_checkpoint(self, path):
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)

            self.net.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))
            return checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_path))
            return 0

    def lr_update(self, p, args):
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr / ((1 + args.alpha * p) ** args.beta)

    def lbda_update(self, p, args):
        return 2. / (1 + np.exp(-args.gamma * p)) - 1.

    def test_epoch(self, device, test_loader, epoch=None, msg=''):
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device).long()
                output = self.net(data, classify=True)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\n{} test avg loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(
            msg, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        if epoch is not None:
            self.writer.add_scalar('test_loss', test_loss, global_step=epoch)
            self.writer.add_scalar('test_correct', correct * 100. / len(test_loader.dataset), global_step=epoch)
        return test_loss, correct, len(test_loader.dataset)
