from experiment_manage import ExperimentServer
from main import main
from test import test
import argparse, os
import make_args

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='7')
parser.add_argument('--metric', default='MELDA')
p_args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

parameters_dir = {'metric': [p_args.metric],
                  'data_name': ['visda'],
                  'model_name': ['ViT-B/32'],# \thetaP4
                  'backbone': ['CLIP'],
                  'shot_num': [1],
                  'epochs': [2],
                  'interim': [1],
                  'episode_train_num': [10],
                  'episode_val_num': [5],
                  'episode_test_num': [5],
                  'source_domain': ['sketch'], # Synthetic->Real:55.25 61.34
                  'target_domain': ['real'], #'painting', 'clipart'],# 'painting', 'clipart'], # 57.30, 66.98
                  'loss_weight': [0.5],
                  'loss_lda': [0.01],
                  'adv_loss': [1.0],
                  'threshold': [0.85],
                  'class_temperature': [10.],
                  'discri_temperature': [1],
                  'DA': [False],
                  'bow_DA': [False],
                  'SGD': [False],
                  'iter': [10],
                  'lr': [5e-4],
                  'ilr': [0.00035],
                  'tri_margin': [1.89503],
                  'epsilon': [0.24],
                  'triplet_loss': [0.032],
                  'discri': ['normal'],
                  'init_lambda': [0],
                  'init_gama': [765.42],
                  'init_trans_scale': [60],
                  'adj_weight': [98.],
                  'init_beta': [7.44],
                  'init_eta': [1],
                  'drop_rate': [0.],
                  'pretrained': [True],
                  'ld_num': [25],
                  'random_target': [False],
                  # for baseline++
                  'indim': [128],
                  'outdim': [5],
                  # for DSN
                  'on_bn': [True],
                  'bn_epoch': [10],
                  'iswrite': [True],
                  'path': ['/home/yl/nni-experiments/adjust_parameters/2023-02-24-20:30:08.849127-R2D2Trans-mini-Imagenet/0']
                  }

para_spaces = {
    'lr' : {
        '_type': 'uniform',
        '_value': [1e-5, 1e-3]
    },
    # 'class_temperature' : {
    #     '_type': 'uniform',
    #     '_value': [5.,12.]
    # },
}
# save_path = '/home/whr/pyProject/Pycharm/melda_v64/experiment_results/FSL_BOW'
save_path = '/home/whr/workspace/melda_v64/experiment_results/FSL_BOW'
if not os.path.exists(save_path):
    os.makedirs(save_path)
exp_server = ExperimentServer(parameters_dir, main, False, para_spaces, save_path)
exp_server.opt.gpu = p_args.gpu
exp_server.run()
