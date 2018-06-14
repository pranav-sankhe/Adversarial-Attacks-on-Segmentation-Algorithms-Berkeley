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

import drn
import data_transforms as transforms

perturbation = []
alpha = 1                           #step size for the iteration 
epilson_limit = 10                  #all values of perturbation above this will be clipped
sum_loss_gradient_sum = []          #sum of gradients of loss function of all images which will be divided by number of images to find the avg.   
count = 0                           #variable to count number of training images             
n_of_iterations = 200               #number of iterations for the iterative step to calculate the universal perturbations. 
excluded_labels = [24]               #enter the ID[or IDs] of the objects you want to remove from the segmentation results. 24 corresponds to 'person'                    


def genYtarget(array):              #implementing Dynamic target segmentation
    array = np.array(array)        

    allindices_array = []            #define array to store all the indices of the input array    

    for i in range(array.shape[1]):
        for j in range(array.shape[2]):
            allindices_array.append([i,j])


    for excluded_item in excluded_labels:
        exclude_indices = np.argwhere(array==excluded_item)     #check which pixel is labeled as the excluded label
        exclude_indices = exclude_indices[:,[1,2]]              #convert the excluded indices in (x,y) format                             
        included_indices = []                                   #array which stores the indices of the allowed labels                
        
        #prepare the included index list 
        for x in allindices_array:      
            dist = []                                                                                        
            for y in exclude_indices:
                dist.append(np.linalg.norm(x-y))                #check if the indice is equal to the one in the excluded index list
                states = [0]
                mask = np.in1d(dist, states)
            if np.any(mask == True):                                            
                pass
            else:
                included_indices.append(x)  
        
        # check the nearest neighbour for each excluded index and fill in the array with the nearest neighbour
        for x in exclude_indices:
            dist = []                                            #store the distances   
            for y in included_indices:
                dist.append(np.linalg.norm(x-y))                 
            min_index_temp = np.argmin(dist)                       
            min_index = included_indices[min_index_temp]              # find the index of the closest pixel  
            array[0,x[0],x[1]] = array[0, min_index[0], min_index[1]]    


        return array


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



def genAdv(args):

    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size    
    cudnn.benchmark = True
    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)


    single_model = DRNSeg(args.arch, args.classes, None,
                          pretrained=True)    
    if args.pretrained:                                                         #load pretrained model    
        single_model.load_state_dict(torch.load(args.pretrained))
    model = torch.nn.DataParallel(single_model).cuda()
    # array = np.ndarray.tolist(np.append(np.arange(19,34), [-1]))
    criterion = nn.NLLLoss2d(ignore_index=-1)                                  #define loss
    criterion.cuda()

    #load data
    data_dir = args.data_dir

    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])

    val_loader = torch.utils.data.DataLoader(
        SegList(data_dir, 'val', transforms.Compose([
            transforms.RandomCrop(crop_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True
    )    

    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()

    model.eval()                         #switch to eval mode

    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        print (target, type(target))
        target = target.numpy()
        target = genYtarget(target)
        print(target,type(target))
        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()
        input = input.cuda()
        target = target.cuda(async=True)

        input_var = torch.autograd.Variable(input,requires_grad=True)
        target_var = torch.autograd.Variable(target)
        
        output = model(input_var)[0]                 #predicted label 
        loss = criterion(output, target_var)         # calculate the loss pixel by pixel    
        
        loss_gradient = loss.backward()              #calculate the gradient of loss fucntion w.r.t input   
        loss_gradient = loss_gradient.cpu().data.np()  # convert to 2D numpy array

        sum_loss_gradient_sum = np.add(sum_loss_gradient_sum,loss_gradient_sum)         #sum over all the images
        count = count + 1                                                               #count number of images
        # loss_f = loss.cpu().data.numpy()
        # input_img = input.cpu().data.numpy()         


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
        avg_loss_gradient = np.divide(sum_loss_gradient_sum,count)      #calculate avg gradient of loss function 
    
        for i in range(n_of_iterations):
            perturbation = np.click(np.subtract(perturbation, np.multiply(np.sign(loss_gradient_sum), alpha)), None, epilson_limit) 
    
    print ("perturbation", perturbation)        # final adversarial perturbation 

if __name__ == '__main__':
    main()
