# parseで渡せるようにしたい．
from anywhere_4S_dev import SequentialSemiSupervisedSegmentation
import torch_networks as networks
import argparse
import csv
import os

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="heart")
parser.add_argument("--gpu_id", type=int, default=-1)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--_lambda", type=float, default=0.0005)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--random", type=int, default=0)
parser.add_argument("--n", type=int, default=3)
parser.add_argument("--M", type=int, default=3)
parser.add_argument("--epoch", type=int, default=3)
parser.add_argument("--semi_epoch", type=int, default=0)
parser.add_argument("--batch", type=int, default=3)
parser.add_argument("--dataset_num", type=int, default=0)
parser.add_argument("--scratch", type=int, default=0)
parser.add_argument("--pp", type=int, default=1)
parser.add_argument("--pp_iou", type=int, default=0)
parser.add_argument("--save_dir", type=str, default="")
parser.add_argument("--trained_model", type=str, default="")
parser.add_argument("--only_supervise", type=int, default=0)
parser.add_argument("--reverse", type=int, default=0)
parser.add_argument("--locally", type=int, default=0)
parser.add_argument("--ita", type=int, default=0)


args = parser.parse_args()


iou_list = []
for i in range(20):# 症例数だけ繰り返し 
    model = networks.UNet()   
    learn = SequentialSemiSupervisedSegmentation(dataset=args.dataset, model=model, repeat_num=1, random_selection=args.random, gpu_id=args.gpu_id, lr=args.lr, _lambda=args._lambda, M=args.M, epoch=args.epoch, batch=args.batch, scratch=args.scratch, pp=args.pp, save_dir=args.dataset, supervise=args.only_supervise, reverse=args.reverse, locally=args.locally, ita=args.ita)

    learn.training(i)