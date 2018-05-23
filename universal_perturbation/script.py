#############################################################
import time
import numpy as np
import copy
import math
from PIL import Image
import sys
import random
import pdb

import torch
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data_utils
import torch.utils.model_zoo as model_zoo

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torch.autograd.gradcheck import zero_gradients

import drn

def get_model(model):

    if model =='drn_c_26':
        net = drn.drn_c_26(pretrained=True)
    elif model =='drn_c_42':
        net = drn.drn_c_42(pretrained=True)
    elif model =='drn_c_58':
        net = drn.drn_c_58(pretrained=True)
    elif model =='drn_c_22':
        net = drn.drn_d_22(pretrained=True)
    elif model =='drn_c_38':
        net = drn.drn_c_38(pretrained=True)
    elif model =='drn_c_54':
        net = drn.drn_c_54(pretrained=True)
    elif model =='drn_c_105':
        net = drn.drn_c_105(pretrained=True)    
                            

    
    for params in net.parameters():
        requires_grad = False
    net.eval()                             #switch to eval mode   
    net.cuda()                             #enable processing on GPU using tensors  
    return net

if __name__ == '__main__':
  
    net           = get_model('drn_c_22')       #specify the model you want to attack  
    location_img  = '/home/pranav/Adversarial-Attacks-on-Segmentation-Algorithms/data'  #folder containing the imagefile folder   
    max_iter_uni    = 100                         # iterations for generating universal perturbation 
    Max_iterDAG = 100                         # iterations for DAG algorithm 
    delta         = 0.2
    img_list      = '/home/pranav/Adversarial-Attacks-on-Segmentation-Algorithms/data/val_images.txt'
    xi            = 0.2
    p             = np.inf
    save_dir      = '.'
    num_classes   = 19                          #no.of classes(labels) in the model  
    overshoot     = 0.02
    t_p           = 0.2

    file = open(img_list)
    img_names = []
    for f in file:
        img_names.append(f.split(' ')[0])
    img_names = [location_img + '/' +x for x in img_names]
    start_time = time.time()
    
    batch_size   = 1
    univ_perturbation = drn.univ_pert(img_names, net, xi=xi, delta=delta, max_iter_uni =max_iter_uni,
                                                      p=p, num_classes=num_classes, overshoot=overshoot, 
                                                      Max_iterDAG =Max_iterDAG,init_batch_size = batch_size,t_p = t_p)
            
    print('Total time: ' ,time.time()-start_time)
    univ_perturbation = univ_perturbation.data.cpu()
    torch.save(univ_perturbation,save_dir+'uap_drn_'+args['<model>']+'.pth')
