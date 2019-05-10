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
# import model
import hfv_model as model
from anchors import Anchors
import losses
# from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
# from dataloader import *
from hfv_dataloader import *
from torch.utils.data import Dataset, DataLoader
import json

import coco_eval
import hfv_csv_eval as csv_eval

# assert torch.__version__.split('.')[1] == '4'
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
	print('0')
	parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', default='csv', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--csv_train', default='./data/pair_train_all.csv', help='Path to file containing training annotations (see readme)')
	parser.add_argument('--csv_classes', default='./data/class.csv', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', default='./data/pair_val_all.csv', help='Path to file containing validation annotations (optional, see readme)')
	# parser.add_argument('--csv_val', default=None,
	# 					help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
	parser.add_argument('--epochs', help='Number of epochs', type=int, default=20)
	parser.add_argument('--model', help='Pretrained model or nothing', type=str, default=None)
	parser.add_argument('--gpu', help='Whether to use gpu', type=bool, default=True)

	parser = parser.parse_args(args)

	# Create the data loaders
	print('1')
	if parser.dataset == 'coco':

		if parser.coco_path is None:
			raise ValueError('Must provide --coco_path when training on COCO,')

		dataset_train = CocoDataset(parser.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
		dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))

	elif parser.dataset == 'csv':

		if parser.csv_train is None:
			raise ValueError('Must provide --csv_train when training on COCO,')

		if parser.csv_classes is None:
			raise ValueError('Must provide --csv_classes when training on COCO,')


		dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

		if parser.csv_val is None:
			dataset_val = None
			print('No validation annotations provided.')
		else:
			dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	print('2')
	sampler = AspectRatioBasedSampler(dataset_train, batch_size=8, drop_last=False)  # bacth_size default 2
	dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

	if dataset_val is not None:
		sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
		dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)
		dataloader_val = None

	# Create the model
	# if parser.depth == 18:
	# 	retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
	# elif parser.depth == 34:
	# 	retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
	# elif parser.depth == 50:
	# 	retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
	# elif parser.depth == 101:
	# 	retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
	# elif parser.depth == 152:
	# 	retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
	# else:
	# 	raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
	print('3')

	start = 0
	if parser.model is not None:
		print('loading pretrained model {}'.format(parser.model))
		retinanet = torch.load(parser.model)
		s_b = parser.model.rindex('_')
		s_e = parser.model.rindex('.')
		# start = int(parser.model[s_b + 1:s_e]) + 1
		print('continue on {}'.format(start))
	else:
		# Create the model
		print('init model resnet{}'.format(parser.depth))
		if parser.depth == 18:
			retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
		elif parser.depth == 34:
			retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
		elif parser.depth == 50:
			retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
		elif parser.depth == 101:
			retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
		elif parser.depth == 152:
			retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
		else:
			raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

	# use_gpu = True
	print('4')
	if parser.gpu:
		retinanet = retinanet.cuda()
	
	retinanet = torch.nn.DataParallel(retinanet).cuda()

	retinanet.training = True
	print('5')
	# optimizer = optim.SGD(retinanet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)  # default lr = 1e-5
	optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 16])
	# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

	loss_hist = collections.deque(maxlen=500)
	print('6')
	retinanet.train()
	retinanet.module.freeze_bn()
	logging.basicConfig(level=logging.DEBUG,
						format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
						datefmt='%b %d %H:%M', filename='./model/train.log', filemode='w')
	print('Num training images: {}'.format(len(dataset_train)))
	logging.info('Num training images: {}'.format(len(dataset_train)))

	loss_curve = []
	# for epoch_num in range(parser.epochs):
	for epoch_num in range(start, parser.epochs):
		retinanet.train()
		retinanet.module.freeze_bn()

		epoch_loss = []
		mAP = 0
		dis = 0

		for iter_num, data in enumerate(dataloader_train):
			#break
			try:

				optimizer.zero_grad()

				classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

				classification_loss = classification_loss.mean()
				regression_loss = regression_loss.mean()

				loss = classification_loss + regression_loss

				if bool(loss == 0):
					continue

				loss.backward()

				torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)  # default is open

				optimizer.step()

				loss_hist.append(float(loss))

				epoch_loss.append(float(loss))

				loss_curve.append(float(loss))

				# print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
				print(
					'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
						epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

				logging.info(
					'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
						epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
				del classification_loss
				del regression_loss
			except Exception as e:
				print(e)
				continue

		# if parser.dataset == 'coco':
		if dataset_val is not None and 0:
			if (epoch_num % 5 == 0 and epoch_num != 0) or epoch_num == parser.epochs - 1:
				if parser.dataset == 'coco':
					print('Evaluating dataset')

					coco_eval.evaluate_coco(dataset_val, retinanet)

				elif parser.dataset == 'csv': # and parser.csv_val is not None:

					print('Evaluating dataset')

					mAP, dis = csv_eval.evaluate(dataset_val, retinanet)

		
		# scheduler.step(np.mean(epoch_loss))
		scheduler.step()

		# torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))
		torch.save(retinanet.module, './model/adam_A_F_{}_retinanet_{}_mAP{}_dis{}.pt'.format(parser.dataset, epoch_num, mAP, dis))

	retinanet.eval()

	# torch.save(retinanet, 'model_final.pt'.format(epoch_num))
	torch.save(retinanet, './model/all_full_{}_retinanet_final.pt'.format(parser.dataset))
	with open('loss.json', 'w') as f:
		json.dump(loss_curve, f)


if __name__ == '__main__':
	main()
