#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import pdb
from os.path import exists, join, split

import time

import numpy as np
import shutil

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.autograd import grad

import drn
import data_transforms as transforms



no_of_iterations = 150
 

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return self.softmax(y), x

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param


class SegList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        if self.label_list is not None:
            data.append(Image.open(join(self.data_dir, self.label_list[index])))
        data = list(self.transforms(*data))
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255]
    correct = correct.view(-1)
    score = correct.float().sum(0).mul(100.0 / correct.size(0))
    return score.data[0]

def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind])
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def genAdv(args):

    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase
    num_classes = args.classes 
    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
                          pretrained=False)
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = torch.nn.DataParallel(single_model).cuda()

    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'], std=info['std'])            #normalize the data[images]
    test_loader = torch.utils.data.DataLoader(
        SegList(data_dir, phase, transforms.Compose([        
            transforms.ToTensor(),
            normalize,
        ]), out_name=True),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()
    data_time = AverageMeter()
    model.eval()                         #switch to eval mode

    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    for iter, (image, label, name) in enumerate(test_loader):
        
        data_time.update(time.time() - end)
        image_var = Variable(image, requires_grad=False, volatile=True)        
        
        count = 0                  # to keep track of the number of iterations in the while loop 
        flag = 0                   #To check whether the random permutation function has been called. 
        no_of_Cpredicted = 1       # initialize the number of correctly predicted targets 
        
        while count < no_of_iterations or no_of_Cpredicted == 0:  

            final = model(image_var)[0]
            _, pred_labels = torch.max(final, 1)
            
            pred_labels = pred_labels.cpu().data.numpy()
            ground_truth = label.numpy()
                        
            diff = np.subtract(pred_labels[0],ground_truth[0])     #diff between ground truth and predicted labels 
            indices = np.where(diff == 0)                          #check the correctly predicted labels 
            i_index = indices[0]                                   #indices of correctly predicted lables.          
            j_index = indices[1]    

            no_of_Cpredicted = len(i_index)


            final = final.cpu().data.numpy()                       # matrix containing the probablities
            finalshape = final.shape                               # get the shape of the matrix
            width = finalshape[2]                                  # width of image
            height = finalshape[3]                                 # height of image
            
            classifier_gt = np.empty([width, height])              #create matrix to store the probabilities of corresponding ground truth labels
            classifier_ad = np.empty([width, height])              #create matrix to store the probabilities of corresponding adversarial labels
            
            temp = ground_truth[0]                                 # temp matrix to store ground truth labels         
            for i in range(width):                                  
                for j in range(height):    
                    index = temp[i][j]                             # check which label is present             
                    if index > 18:
                        pass
                    else:
                        classifier_gt[i][j] =  final[0,index,i,j]      # select the probability of the selected label 

            for i in i_index:
                for j in j_index:
                    classifier_gt[i][j] = 0


            if flag == 0:                                               
                adversarial_labels = np.random.permutation(ground_truth[0])   # create adversarial label matrix once for an image
                flag = 1    

            temp = adversarial_labels    
            for i in range(width):
                for j in range(height):
                    index = temp[i][j]
                    if index > 18:
                        pass
                    else:
                        classifier_ad[i][j] =  final[0,index,i,j]

            for i in i_index:
                for j in j_index:
                    classifier_ad[i][j] = 0
            # print ('classifier_gt', classifier_gt)
            # print ('classifier_ad', classifier_ad)
            # print(no_of_Cpredicted)

            grad_variables = image_var.cpu().data.numpy()[0]
            gradient_gt = np.zeros((3,width,height))
            gradient_ad = np.zeros((3,width,height))
            
            for k in range(3):        
                for i in range(width):
                    for j in range(height):
                        #print("grad_variables",len(grad_variables[k][0,:]), len(grad_variables[k][:,0]))
                        #print (classifier_gt.shape)
                        gx, gy = np.gradient(classifier_gt,grad_variables[k][:,0],grad_variables[k][0,:])     
                        if np.isinf(gx[i][j]) == True:
                            gx[i][j] = 0
                        if np.isinf(gy[i][j]) == True:
                            gy[i][j] = 0            
                        gradient_gt[k][i][j] = np.sqrt( np.square(gx[i][j]) + np.square(gy[i][j]) )        

            for k in range(3):
                for i in range(width):
                    for j in range(height):
                        gx, gy = np.gradient(classifier_ad,grad_variables[k][:,0],grad_variables[k][0,:])     
                        if np.isinf(gx[i][j]) == True:
                            gx[i][j] = 0
                        if np.isinf(gy[i][j]) == True:
                            gy[i][j] = 0            
                        gradient_ad[k][i][j] = np.sqrt( np.square(gx[i][j]) + np.square(gy[i][j]) )        


            r_m = gradient_ad - gradient_gt

            print (r_m,r_m.shape)
            r_m_normalized = r_m*(gamma/np.linalg.det(r_m.data.numpy()))  
            perturbation = np.add(perturbation, r_m_normalized)
            image_var = image_var.cpu().data.numpy()
            image_var = np.add(image_var, perturbation)
            count = count + 1
            print("perturbation",perturbation)
        adv_image = image_var.cpu().data.numpy()
        image_var = torch.from_numpy(image_var)
        image_var = Variable(image_var,requires_grad=False, volatile=True )
        save_output_images(adv_image, name, 'adv')    

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('-d', '--data-dir', default=None)
    parser.add_argument('-c', '--classes', default=0, type=int)
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--load-release', dest='load_rel', default=None)
    parser.add_argument('--phase', default='val')
    args = parser.parse_args()

    assert args.data_dir is not None
    assert args.classes > 0

    print(' '.join(sys.argv))
    print(args)

    return args


def main():
    args = parse_args()
    if args.cmd == 'test':
        genAdv(args)

if __name__ == '__main__':
    main()            