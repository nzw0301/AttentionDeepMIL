from __future__ import print_function

import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from dataloader import MnistBags
from model import Attention
from model import GatedAttention

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay (default: 10e-5)')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length (default: 10)')
parser.add_argument('--std_bag_length', type=int, default=2, metavar='VL',
                    help='standard deviation of bag length (default: 2)')
parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                    help='number of bags in training set (default: 200)')
parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
                    help='number of bags in test set (default: 50)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gated', action='store_true', default=False,
                    help='Use gated_attention ')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_loader = data_utils.DataLoader(
    MnistBags(
        target_number=args.target_number,
        mean_bag_length=args.mean_bag_length,
        std_bag_length=args.std_bag_length,
        num_bag=args.num_bags_train,
        seed=args.seed,
        train=True
    ),
    batch_size=1,
    shuffle=True,
    **loader_kwargs
)

test_loader = data_utils.DataLoader(
    MnistBags(
        target_number=args.target_number,
        mean_bag_length=args.mean_bag_length,
        std_bag_length=args.std_bag_length,
        num_bag=args.num_bags_test,
        seed=args.seed,
        train=False
    ),
    batch_size=1,
    shuffle=False,
    **loader_kwargs
)

print('Init Model')
if args.gated:
    model = GatedAttention()
else:
    model = Attention()

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)


def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]
        data, bag_label = data.to(device), bag_label.to(device)

        # reset gradients
        optimizer.zero_grad()

        # forward to get predict_prob, labels, and attention scores
        Y_prob, Y_hat, A = model(data)

        # calculate objective
        loss = model.calculate_objective(Y_prob, bag_label)
        train_loss += loss.item()

        # backward pass
        loss.backward()
        # step
        optimizer.step()

        # calculate error:  1. - acc
        error = model.calculate_classification_error(Y_hat, bag_label)
        train_error += error

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))


def test():
    model.eval()
    test_loss = 0.
    test_error = 0.
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            bag_label = label[0]
            instance_labels = label[1]
            data, bag_label = data.to(device), bag_label.to(device)

            Y_prob, predicted_label, attention_weights = model(data)
            loss = model.calculate_objective(Y_prob, bag_label)
            test_loss += loss.item()

            error = model.calculate_classification_error(predicted_label, bag_label)
            test_error += error

            if batch_idx < 5:  # show bag labels and instance labels for first 5 bags
                bag_level = (bag_label.item(), int(predicted_label.item()))
                instance_level = list(zip(instance_labels.numpy()[0],
                                          np.round(attention_weights.data.numpy()[0], decimals=3)))

                print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                      'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))


if __name__ == '__main__':
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    print('Start Testing')
    test()
