# ************************************************************
# Author : Bumsoo Kim, 2017
# Github : https://github.com/meliketoy/fine-tuning.pytorch
#
# Korea University, Data-Mining Lab
# Deep Convolutional Network Fine tuning Implementation
#
# Description : main.py
# The main code for training classification networks.
# ***********************************************************

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import config as cf
import torchvision
import time
import copy
import os
import sys
import argparse
import scipy.ndimage as ndi
import csv

from torchvision import datasets, models, transforms
from networks import *
from torch.autograd import Variable
from transform_helper import random_transform_fn


parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, type=int, help='depth of model')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--finetune', '-f', action='store_true', help='Fine tune pretrained model')
parser.add_argument('--addlayer','-a',action='store_true', help='Add additional layer in fine-tuning')
parser.add_argument('--resetClassifier', '-r', action='store_true', help='Reset classifier')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()

# Phase 1 : Data Upload
print('\n[Phase 1] : Data Preperation')
 
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomRotation(180),
        transforms.Resize(299),
        transforms.RandomResizedCrop(224),
        # transforms.Lambda(lambda x: random_transform_fn(x, cf.T)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean, cf.std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean, cf.std)
        # transforms.Resize(384),
        # transforms.FiveCrop((224,224)),
        # transforms.Lambda(
        #     lambda crops: torch.stack([
        #         transforms.Normalize(cf.mean, cf.std)
        #         (transforms.ToTensor()(crop)) for crop in crops])),
    ]),
}

data_dir = cf.aug_base
dataset_dir = cf.data_base.split("/")[-1] + os.sep
print("| Preparing model trained on %s dataset..." %(cf.data_base.split("/")[-1]))
dsets = {
    x : datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}
dset_loaders = {
    x : torch.utils.data.DataLoader(dsets[x], batch_size = cf.batch_size, shuffle=(x=='train'), num_workers=4)
    for x in ['train', 'val']
}

dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes

use_gpu = torch.cuda.is_available()

# Phase 2 : Model setup
print('\n[Phase 2] : Model setup')

def getNetwork(args):
    if (args.net_type == 'alexnet'):
        net = models.alexnet(pretrained=args.finetune)
        file_name = 'alexnet'
    elif (args.net_type == 'vggnet'):
        if(args.depth == 11):
            net = models.vgg11(pretrained=args.finetune)
        elif(args.depth == 13):
            net = models.vgg13(pretrained=args.finetune)
        elif(args.depth == 16):
            net = models.vgg16(pretrained=args.finetune)
        elif(args.depth == 19):
            net = models.vgg19(pretrained=args.finetune)
        else:
            print('Error : VGGnet should have depth of either [11, 13, 16, 19]')
            sys.exit(1)
        file_name = 'vgg-%s' %(args.depth)
    elif (args.net_type == 'resnet'):
        net = resnet(args.finetune, args.depth)
        file_name = 'resnet-%s' %(args.depth)
    else:
        print('Error : Network should be either [alexnet / vggnet / resnet]')
        sys.exit(1)

    return net, file_name

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Test only option
if (args.testOnly):
    print("| Loading checkpoint model for test phase...")
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    model, file_name = getNetwork(args)
    # model.fc = torch.nn.Linear(model.fc.in_features,len(dset_classes))

    if use_gpu:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        # cudnn.benchmark = True

    checkpoint = torch.load('./checkpoint/'+dataset_dir+'/'+file_name+'.t7')
    model.load_state_dict(checkpoint['model'].state_dict())

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    testsets = datasets.ImageFolder(cf.test_dir, data_transforms['val'])

    testloader = torch.utils.data.DataLoader(
        testsets,
        batch_size = 1,
        shuffle = False,
        num_workers=1
    )

    print("\n[Phase 3 : Inference on %s]" %cf.test_dir)
    for batch_idx, (inputs, targets) in enumerate(testloader):#dset_loaders['val']):
        # bs,ncrops,c,h,w = inputs.size()
        # inputs = inputs.view(-1,c,h,w)
        if use_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        # outputs_avg = outputs.view(bs,ncrops,-1).mean(1)

        softmax_res = softmax(outputs.data.cpu().numpy()[0])
        # softmax_res = softmax(outputs_avg.data.cpu().numpy()[0])

        _, predicted = torch.max(outputs.data, 1)
        # predicted = torch.clamp(predicted.cpu().sum(0),min=0,max=1)
        total += targets.size(0)
        correct += predicted.eq(targets.cpu().data).sum()

    acc = 100.*correct/total
    print("| Test Result\tAcc@1 %.2f%%" %(acc))

    sys.exit(0)

# Training model
def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=cf.num_epochs):
    global dataset_dir
    since = time.time()

    best_model, best_acc  = model, 0
    best_score = 0

    print('\n[Phase 3] : Training Model')
    print('| Training Epochs = %d' %num_epochs)
    print('| Initial Learning Rate = %f' %args.lr)
    print('| Optimizer = SGD')
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer, lr = lr_scheduler(optimizer, epoch)
                print('\n=> Training Epoch #%d, LR=%f' %(epoch+1, lr))
                model.train(True)
            else:
                model.train(False)
                model.eval()

            running_loss, running_corrects, tot, running_tp, running_pos = 0.0, 0, 0, 0.0 ,0 

            for batch_idx, (inputs, labels) in enumerate(dset_loaders[phase]):
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                # Forward Propagation
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # Backward Propagation
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Statistics
                running_loss += loss.data[0]
                running_corrects += preds.eq(labels.data).cpu().sum()
                pos_examples = labels.data.eq(1).cpu()
                running_tp += (pos_examples * preds.eq(labels.data).cpu()).sum()
                running_pos += pos_examples.sum()
                tot += labels.size(0)

                if (phase == 'train'):
                    sys.stdout.write('\r')
                    sys.stdout.write('| Epoch [%2d/%2d] Iter [%3d/%3d]\t\tLoss %.4f\tAcc %.2f%%'
                            %(epoch+1, num_epochs, batch_idx+1,
                                (len(dsets[phase])//cf.batch_size)+1, loss.data[0], 100.*running_corrects/tot))
                    sys.stdout.flush()
                    sys.stdout.write('\r')

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc  = running_corrects / dset_sizes[phase]
            epoch_sensitivity = running_tp / running_pos

            if (phase == 'val'):
                print('\n| Validation Epoch #%d\t\t\tLoss %.4f\tAcc %.2f%%\tSensitivity %.2f'
                    %(epoch+1, loss.data[0], 100.*epoch_acc, epoch_sensitivity))

                # if epoch_acc > best_acc:
                save_name = file_name
                if epoch_sensitivity + epoch_acc > best_score:
                    save_name = file_name + '_best'
                    # print('| Saving Best model...\t\t\tTop1 %.2f%%' %(100.*epoch_acc))
                    print('| Saving Best model...\t\t\tTop1 %.2f acc %.2f' % (epoch_sensitivity, epoch_acc))
                    # best_acc = epoch_acc
                    best_score =  epoch_sensitivity + epoch_acc

                save_model = copy.deepcopy(model)
                state = {
                    'model': save_model,
                    'acc':   epoch_acc,
                    'sensitivity': epoch_sensitivity,
                    'epoch':epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                save_point = './checkpoint/'+cf.name+'/'
                if not os.path.isdir(save_point):
                    os.mkdir(save_point)
                torch.save(state, save_point+save_name+'.t7')

                

    time_elapsed = time.time() - since
    print('\nTraining completed in\t{:.0f} min {:.0f} sec'. format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc\t{:.2f}%'.format(best_acc*100))

    return best_model

def exp_lr_scheduler(optimizer, epoch, init_lr=args.lr, weight_decay=args.weight_decay, lr_decay_epoch=10):
    lr = init_lr * (0.5**(epoch // cf.lr_decay_epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = weight_decay

    return optimizer, lr

model_ft, file_name = getNetwork(args)

# discard last layer
model_ft.fc = torch.nn.Linear(model_ft.fc.in_features,len(dset_classes))
ignored_params_id = list(map(id, model_ft.fc.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params_id,
	model_ft.parameters())
fc_params = filter(lambda p: id(p) in ignored_params_id,
	model_ft.parameters())


if(args.resetClassifier):
    print('| Reset final classifier...')
    if(args.addlayer):
        print('| Add features of size %d' %cf.feature_size)
        num_ftrs = model_ft.fc.in_features
        feature_model = list(model_ft.fc.children())
        feature_model.append(nn.Linear(num_ftrs, cf.feature_size))
        feature_model.append(nn.BatchNorm1d(cf.feature_size))
        feature_model.append(nn.ReLU(inplace=True))
        feature_model.append(nn.Linear(cf.feature_size, len(dset_classes)))
        model_ft.fc = nn.Sequential(*feature_model)
    else:
        if(args.net_type == 'alexnet' or args.net_type == 'vggnet'):
            num_ftrs = model_ft.classifier[6].in_features
            feature_model = list(model_ft.classifier.children())
            feature_model.pop()
            feature_model.append(nn.Linear(num_ftrs, len(dset_classes)))
            model_ft.classifier = nn.Sequential(*feature_model)
        elif(args.net_type == 'resnet'):
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, len(dset_classes))

if use_gpu:
    model_ft = model_ft.cuda()
    model_ft = torch.nn.DataParallel(model_ft, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

if cf.resume:
    checkpoint = torch.load(os.path.join('./checkpoint/',cf.name,file_name+'.t7'))
    model_ft.load_state_dict(checkpoint['model'].state_dict())


if __name__ == "__main__":
    if cf.class_weight is None:
        criterion = nn.CrossEntropyLoss()
    else:
        if use_gpu:
            weight = torch.cuda.FloatTensor(cf.class_weight)
        else:
            weight = torch.FloatTensor(cf.class_weight)
        criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer_ft = optim.SGD([
	{'params': base_params},
	{'params': fc_params, 'lr': args.lr}
	], lr=args.lr*0.1, momentum=0.9, weight_decay=args.weight_decay)

    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=cf.num_epochs)
