# parseで渡せるようにしたい．
from anywhere_4S import SequentialSemiSupervisedSegmentation
import torch_networks as networks
import argparse
import csv
import os
import glob
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from evaluate_selective_annotation import evaluation

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="heart")
parser.add_argument("--gpu_id", type=int, default=-1)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--_lambda", type=float, default=0.0005)
parser.add_argument("--random", type=int, default=0)
parser.add_argument("--M", type=int, default=3)
parser.add_argument("--epoch", type=int, default=3)
parser.add_argument("--scratch", type=int, default=0)
parser.add_argument("--save_dir", type=str, default="")
parser.add_argument("--only_supervise", type=int, default=0)
parser.add_argument("--locally", type=int, default=0)
parser.add_argument("--ita", type=int, default=0)
parser.add_argument("--epoch_decay", type=int, default=0)

args = parser.parse_args()


iou_list = []
for i in range(20):# 症例数だけ繰り返し 
    model = networks.UNet()   
    learn = SequentialSemiSupervisedSegmentation(dataset=args.dataset,
                                                 model=model, 
                                                 random_selection=args.random, 
                                                 gpu_id=args.gpu_id, 
                                                 lr=args.lr, 
                                                 _lambda=args._lambda, 
                                                 M=args.M, 
                                                 epoch=args.epoch, 
                                                 scratch=args.scratch, 
                                                 save_dir=args.save_dir, 
                                                 supervise=args.only_supervise, 
                                                 locally=args.locally, 
                                                 ita=args.ita, 
                                                 epoch_decay=args.epoch_decay)

    learn.training(i)

# 評価
evaluation(args.dataset)