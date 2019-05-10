import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import hfv_model as model
from anchors import Anchors
import losses
# from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
# from dataloader import *
from hfv_dataloader import *
from torch.utils.data import Dataset, DataLoader

import coco_eval
# import csv_eval
# import hfv_csv_eval as csv_eval
import hfv_crowdhuman_eval as csv_eval

# assert torch.__version__.split('.')[1] == '4'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print('CUDA available: {}'.format(torch.cuda.is_available()))
model_dir = '/home/YC/crowd_retinanet/pytorch-retinanet/model/adam_A_F_csv_retinanet_19_mAP0_dis0.pt'
# model_dir='/home/YC/pytorch-retinanet/all_full_csv_retinanet_36_mAP{0: (0.6896753220630812, 99481.0)}_dis[12.61821368].pt'
def main(args=None):
    print('0')
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', default='csv', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', default='./data/pair_train_all.csv',
                        help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', default='./data/class.csv',
                        help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', default='./data/pair_val_all.csv',
                        help='Path to file containing validation annotations (optional, see readme)')
    # parser.add_argument('--csv_val', default=None,
    # 					help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--model', help='Pretrained model or nothing', type=str, default=model_dir)
    parser.add_argument('--gpu', help='Whether to use gpu', type=bool, default=True)

    parser = parser.parse_args(args)

    # Create the data loaders
    print('1')
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    print('2')

    print('3')
    if not os.path.isfile('./process.pth.tar'):
        print('loading pretrained model {}'.format(parser.model))
        retinanet = torch.load(parser.model)
        torch.save(retinanet.state_dict(), './process.pth.tar')
        start = 0
        print('continue on {}'.format(start))
    else:
        # Create the model
        print('init model resnet{}'.format(parser.depth))
        if parser.depth == 18:
            retinanet = model.resnet18(num_classes=1, pretrained=False)
        elif parser.depth == 34:
            retinanet = model.resnet34(num_classes=1, pretrained=False)
        elif parser.depth == 50:
            retinanet = model.resnet50(num_classes=1, pretrained=False)
        elif parser.depth == 101:
            retinanet = model.resnet101(num_classes=1, pretrained=False)
        elif parser.depth == 152:
            retinanet = model.resnet152(num_classes=1, pretrained=False)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    # use_gpu = True
    print('4')
    retinanet.load_state_dict(torch.load('./process.pth.tar'))
    if parser.gpu:
        retinanet = retinanet.cuda()

    retinanet = torch.nn.DataParallel(retinanet).cuda()

    if dataset_val is not None:
        if parser.dataset == 'coco':
            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv':  # and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, retinanet)
            print(mAP)


if __name__ == '__main__':
    main()
