"""
実験条件をargparseで受け取る
- データセットのpath
- active selection
- gpu_id
- 
"""

from semisupervise_general_2 import SemisuperviesedLearning
import networks
import DCN_master
import argparse
import evaluation
import evaluation_2
import csv
import os

parser = argparse.ArgumentParser()

parser.add_argument("--gpu_id", type=int, default=-1)
parser.add_argument("--alpha", type=float, default=0.001)
parser.add_argument("--_lambda", type=float, default=0.0005)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--n", type=int, default=3)
parser.add_argument("--M", type=int, default=3)
parser.add_argument("--epoch", type=int, default=3)
parser.add_argument("--semi_epoch", type=int, default=0)
parser.add_argument("--batch", type=int, default=3)
parser.add_argument("--dataset_num", type=int, default=0)
parser.add_argument("--scratch", type=int, default=0)
parser.add_argument("--pp", type=int, default=1)
parser.add_argument("--pp_iou", type=int, default=0)
parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--trained_model", type=str, default="")
parser.add_argument("--only_supervise", type=int, default=0)
parser.add_argument("--reverse", type=int, default=0)
parser.add_argument("--locally", type=int, default=0)