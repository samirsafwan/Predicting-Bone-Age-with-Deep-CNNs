import csv
from collections import defaultdict
import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn

import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate
import torchvision.models as models
import torch.utils.model_zoo as model_zoo



vgg19_bn = models.vgg19_bn(pretrained=True)
vgg19_bn = nn.Sequential(*list(vgg19_bn.features.children())[:-1])
for params in vgg19_bn:
    params.requires_grad = False
print (vgg19_bn)



############################################################################
def read_csv():
    response_data = defaultdict(str)
    with open('boneage-training-dataset.csv') as f:
        reader = csv.reader(f, delimiter= ',')
        rownum = 0
        for row in reader:
            if rownum > 0:
                if 0 <= float(row[1])/12.0 < 5:
                    response_data[row[0]+'.png'] = "toddler"
                elif 5 <= float(row[1])/12.0 < 13:
                    response_data[row[0]+'.png'] = "child"
                elif 13 <= float(row[1])/12.0 < 20:
                    response_data[row[0]+'.png'] = "young adult"
                else: 
                    response_data[row[0]+'.png'] = "adult"
            rownum += 1
    return response_data

