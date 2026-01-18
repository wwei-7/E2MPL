from experiment_manage import ExperimentServer
from main import main
import argparse, os
import make_args

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='1')
parser.add_argument('--metric', default='R2D2Trans')
# parser.add_argument('--metric', default='Img2Class')
# parser.add_argument('--metric', default='MELDA')
p_args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = p_args.gpu

parameters_dir = {'metric': [p_args.metric],
                  'data_name': ['visda'],
                  'backbone': ['ResNet12'],
                  'shot_num': [1],
                  'epochs': [15],
                  'interim': [1],
                  'source_domain': ['real'],
                  'target_domain': ['clipart'],
                  'loss_weight': [0.5],
                  'DA': [True],
                  'class_temperature': [10],
                  'discri_temperature': [1],
                  'SGD': [False],
                  'iter': [10],
                  'lr': [3e-4],
                  'discri': ['normal'],
                  'init_lambda': [0],
                  'init_gama': [740],
                  'init_trans_scale': [60],
                  'init_beta': [7],
                  'init_eta': [1],
                  'pretrained': [True],
                  'ld_num': [25],
                  'random_target': [False],
                  # for DSN
                  'on_bn': [True],
                  'iswrite': [True],
                  }

exp_server = ExperimentServer(parameters_dir, main)
exp_server.opt.gpu = p_args.gpu
exp_server.run()
