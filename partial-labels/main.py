import torch as T
import torchvision
import numpy as np
import models
import losses
import itertools

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str)
parser.add_argument('-bs', type=int, default=256)
parser.add_argument('-n_epochs', type=int, default=100)
parser.add_argument('-n_pos', type=int)
parser.add_argument('-model', type=str, default='small_cnn')
parser.add_argument('-loss', type=str)
parser.add_argument('-lr', type=float, default=0.0001)
parser.add_argument('-hash', type=int, default=101)
args = parser.parse_args()

device = T.device('cuda')

if args.dataset == 'cifar':
    data_train = torchvision.datasets.CIFAR10('data/cifar', download=True, train=True, transform=torchvision.transforms.ToTensor())
    data_test = torchvision.datasets.CIFAR10('data/cifar', download=True, train=False, transform=torchvision.transforms.ToTensor())
    n_channels = 3
elif args.dataset == 'mnist':
    data_train = torchvision.datasets.MNIST('data/mnist', download=True, train=True, transform=torchvision.transforms.ToTensor())
    data_test = torchvision.datasets.MNIST('data/mnist', download=True, train=False, transform=torchvision.transforms.ToTensor())
    n_channels = 1

if args.model == 'small_cnn':
    model = models.SmallCNN(n_channels).to(device)

loader_train = T.utils.data.DataLoader(data_train, args.bs, shuffle=True, pin_memory=True)
loader_test = T.utils.data.DataLoader(data_test, args.bs, shuffle=False, pin_memory=True)

n_classes = 10

opt = T.optim.Adam(model.parameters(), args.lr)

nll = T.nn.NLLLoss(reduction='sum')
loss = eval(f'losses.{args.loss}_loss')

shifts = T.Tensor(list(itertools.combinations(1 + np.arange(9), n_classes - args.n_pos))).long().to(device)

def make_prior(true_label, hash):
    neg = ( shifts[hash % len(shifts)] + true_label.unsqueeze(1) ) % n_classes
    prior = T.ones((true_label.shape[0], 10)).to(device)
    prior.scatter_(1, neg, T.zeros_like(neg).float())
    return prior / prior.sum(1).unsqueeze(1)

for epoch in range(args.n_epochs):

    losses1 = []
    losses2 = []
    nlls = []
    accs = []

    for x, y in loader_train:
       
        x = x.to(device)
        y = y.to(device)

        q = model(x).log_softmax(1)
        
        hash = (x.mean((1,2,3))*args.hash).long()
        #print(hash % len(shifts))
        prior =  make_prior(y, hash)       

        opt.zero_grad()
        
        l1, l2 = loss(q, prior)
        l = l1.mean() - l2.mean()  
        l.backward()

        opt.step()

        losses1.append(l1.sum().item())
        losses2.append(l2.sum().item())
        nlls.append(nll(q, y).item())
        accs.append((q.argmax(1)==y).float().sum().item())
        
    L = len(data_train)
    print('train nll/acc: ', sum(nlls)/L, sum(accs)/L)

    with T.no_grad():
        nlls = []
        accs = []
        for x, y in loader_test:

            x = x.to(device)
            y = y.to(device)

            q = model(x).log_softmax(1)
            
            nlls.append(nll(q, y).item())
            accs.append((q.argmax(1)==y).float().sum().item())
            
        L = len(data_test)
        print('test nll/acc: ', sum(nlls)/L, sum(accs)/L)




