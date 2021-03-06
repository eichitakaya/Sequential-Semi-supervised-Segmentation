# parseで渡せるようにしたい．
from semisupervise import  SemisuperviesedLearning
import networks
import DCN_master
import chainer
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--gpu_id", type=int, default=-1)
parser.add_argument("--_lambda", type=float, default=0.0005)
parser.add_argument("--n", type=int, default=3)
parser.add_argument("--epoch", type=int, default=3)
parser.add_argument("--batch", type=int, default=3)
parser.add_argument("--dataset_num", type=int, default=0)
parser.add_argument("--model_path", type=str, default="")

args = parser.parse_args()

#model = DCN_master.DCN_modify10()
model = networks.DCN123()
learn = SemisuperviesedLearning(model, gpu_id=args.gpu_id, _lambda=args._lambda, n=args.n, epoch=args.epoch, batch=args.batch, dataset_num=args.dataset_num, save_dir=args.model_path)
learn.training()

model.to_cpu()
model_name = args.model_path + "/semisuper" + ".model"
#model_name = args.model_path + "/semisuper_unet" + ".model"
chainer.serializers.save_npz(model_name, model)
print("model written")
