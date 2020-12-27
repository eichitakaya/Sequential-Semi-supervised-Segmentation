# parseで渡せるようにしたい．
from semisupervise_general_2 import SemisuperviesedLearning
import networks
import DCN_master
import chainer
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


args = parser.parse_args()

iou_list = []
for i in range(args.repeat):
    os.makedirs(args.model_path+"/result_img"+str(i+1), exist_ok=True)
    print("repeat " + str(i+1))
    raw_model = networks.UNet
    model = networks.UNet()
    learn = SemisuperviesedLearning(model, i+1, raw_model=raw_model, gpu_id=args.gpu_id, alpha=args.alpha, _lambda=args._lambda, n=args.n, M=args.M, epoch=args.epoch, semi_epoch=args.semi_epoch, batch=args.batch, dataset_num=args.dataset_num, scratch=args.scratch, pp=args.pp, save_dir=args.model_path, supervise=args.only_supervise, reverse=args.reverse, locally=args.locally)
    learn.training()
    iou, ious = evaluation_2.evaluate(repeat_num=i+1, batchsize=args.batch, img_path=args.model_path, dataset=args.dataset_num, n=args.n, postprocess=args.pp_iou, gpu_id=args.gpu_id, reverse=args.reverse)
    iou_list.append(iou) #iouの計算を行い，listにappend
    with open(args.model_path + "/result" + str(i+1) + ".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(ious)

model.to_cpu()
#model_name = args.model_path + "/semisuper" + ".model"
#model_name = args.model_path + "/semisuper_unet" + ".model"
#chainer.serializers.save_npz(model_name, model)
#print("model written")
print(iou_list)

with open(args.model_path + "/result_all.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(iou_list)
print("saved iou_means!")