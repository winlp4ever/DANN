# main model, includes all modules and how info flows within, forward and backward
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from modules import GradNet
from dataset import GradData
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
        self.lbda = 0
        self.lr = lr

    def train_epoch(self, args, device, train_loader, epoch):

        self.net.train()
        c_loss = 0
        d_loss = 0
        for batch_idx, data in enumerate(train_loader):
            percent = batch_idx / len(train_loader)
            p = (epoch + percent) / args.epochs
            self.lr_update(p, args)
            self.lbda_update(p, args)

            X_t, X_d, y, d_t, d_d = [a.to(device) for a in data]

            self.optim.zero_grad()
            pred_d_ = self.net(X_t, self.lbda)
            pred_c, pred_d = self.net(X_d, self.lbda, True)

            c_l = F.nll_loss(pred_c, y[:, 0].long())
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

        if (epoch + 1) % args.sv_interval == 0:
            print('\nsaving model ...')
            self.save_checkpoint(args,
                                 {
                                     'epoch': epoch,
                                     # 'arch': args.arch,
                                     'state_dict': self.net.state_dict(),
                                     'optimizer': self.optim.state_dict(),
                                 }, epoch)
            #pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        print('')
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
            # args.start_epoch = checkpoint['epoch']
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
        self.lbda = 2. / (1 + np.exp(-args.gamma * p)) - 1.

    def test_epoch(self, device, test_loader, epoch=None):
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device).long()
                output = self.net(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        if epoch is not None:
            self.writer.add_scalar('test_loss', test_loss, global_step=epoch)
            self.writer.add_scalar('test_correct', correct * 100. / len(test_loader), global_step=epoch)


def main(args):
    # Training settings

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    mnist_fn = './data/mnist/train'
    svhn_fn = './data/svhn/train'
    mnist_test = './data/mnist/test'
    mnist_test_label = './data/mnist/test_label'
    train_data = GradData(mnist_fn, svhn_fn)
    test_data = GradData(mnist_test, None, mnist_only=True, mnist_label_fn=mnist_test_label)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True)
    model = Model(device, args.lr, args.momentum)

    for epoch in range(args.epochs):
        if epoch % args.sv_interval == 0:
            model.test_epoch(device, test_loader, epoch)
        model.train_epoch(args, device, train_loader, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--sv-interval', type=int, default=10)
    parser.add_argument('--ckpt-path', '-c', default='./checkpoints/', nargs='?')
    parser.add_argument('--alpha', nargs='?', type=float, default=10)
    parser.add_argument('--beta', nargs='?', type=float, default=0.75)
    parser.add_argument('--gamma', nargs='?', type=float, default=10)
    parser.add_argument('--resume', type=bool, default=False, const=True, nargs='?')
    args = parser.parse_args()
    main(args)
