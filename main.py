import argparse
from model import Model
from dataset import GradData
import torch

def main(args):
    # Training settings

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    mnist_fn = './data/mnist/train'
    svhn_fn = './data/svhn/train'
    mnist_test = './data/mnist/test'
    mnist_test_label = './data/mnist/test_label'
    svhn_test = './data/svhn/test'
    train_data = GradData(mnist_fn, svhn_fn)
    mnist_test = GradData(mnist_test, None, mnist_only=True, mnist_label_fn=mnist_test_label)
    svhn_test = GradData(None, svhn_test, svhn_only=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=args.test_batch_size, shuffle=True)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_test, batch_size=args.test_batch_size, shuffle=True)
    model = Model(device, args.lr, args.momentum)

    for epoch in range(args.epochs):
        if epoch % args.sv_interval == 0:
            tl, tc, tle = model.test_epoch(device, mnist_test_loader, epoch)
            print('target: mnist: avg loss {:.4f} -- accuracy {}/{} ({:.0f}%)'.format(tl, tc, tle, tc / tle * 100.))
            dl, dc, dle = model.test_epoch(device, svhn_test_loader, epoch)
            print('source: svhn: avg loss {:.4f} -- accuracy {}/{} ({:.0f}%)'.format(dl, dc, dle, dc / dle * 100.))
        model.train_epoch(args, device, train_loader, epoch)
    tl, tc, tle = model.test_epoch(device, mnist_test_loader, epoch)
    print('target: mnist: avg loss {:.4f} -- accuracy {}/{} ({:.0f}%)'.format(tl, tc, tle, tc / tle * 100.))
    dl, dc, dle = model.test_epoch(device, svhn_test_loader, epoch)
    print('source: svhn: avg loss {:.4f} -- accuracy {}/{} ({:.0f}%)'.format(dl, dc, dle, dc / dle * 100.))

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
