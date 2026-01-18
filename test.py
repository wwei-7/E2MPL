from __future__ import print_function
import datetime
import torch.nn.parallel
from model import hyper_model
import torch.utils.data
from PIL import ImageFile
import matplotlib.pyplot as plt
import numpy as np
from data_generator import *

import sys
sys.dont_write_bytecode = True

# ============================ Data & Networks =====================================
from tqdm import tqdm
# ==================================================================================
from experiment_api.print_utils import accumulate_and_output_meters, print_and_record
from experiment_api.out_files import out_put_metadata

ImageFile.LOAD_TRUNCATED_IMAGES = True
seed = 1024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def pretrain(encoder, max_epochs):
    fc_layer = nn.Linear(441*64, 64)
# DEVICE = torch.device('cuda:0')

import json
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./runs/pics/img2class')
import scipy as sp
import scipy.stats
import scipy


def mean_confidence_interval(data, confidence=0.95):
    a = [1.0*np.array(data[i]) for i in range(len(data))]
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h

import time

def test_(opt):
    # define loss function (criterion) and optimizer
    # model.method.eval()
    # optionally resume from a checkpoint
    model = hyper_model.Trainer_new(opt)
    model_dir = os.path.join(opt.path, 'saved_models')
    ckpt = os.path.join(model_dir, 'model_best.pth.tar')
    if os.path.isfile(ckpt):
        print("=> loading checkpoint '{}'".format(ckpt))
        checkpoint = torch.load(ckpt)
        epoch_index = checkpoint['epoch_index']
        best_prec1 = checkpoint['best_prec1']
        keys = checkpoint['state_dict'].keys()
        checkpoint_tmp = {'state_dict': {}}
        for k in keys:
            k_tmp = k.replace('.module', '')
            checkpoint_tmp['state_dict'][k_tmp] = checkpoint['state_dict'][k]
        model.method.load_state_dict(checkpoint_tmp['state_dict'])
        # model.g_optimizer.load_state_dict(checkpoint_tmp['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(ckpt, checkpoint['epoch_index']))
    else:
        print('load pretrained model failure.')

        # model.method.gen.load_state_dict(ckpt['state_dict'])


    model.cuda()
    model.eval()

    repeat_num = 5
    best_prec1 = 0.0
    total_accuracy = 0.0
    total_h = np.zeros(repeat_num)

    total_accuracy_vector = []
    meters_dict = {}
    for r in range(repeat_num):
        data_dir = opt.dataset_dir+'/'+opt.data_name
        testset = DataGenerator(mode='test', datasource=opt.data_name, data_dir=data_dir, imagesize=opt.imageSize,
                      episode=opt.final_test_episode_num, support_num=opt.shot_num, query_num=opt.query_num, split_PATH=opt.split_PATH,
                                source_domain=opt.source_domain, target_domain=opt.target_domain, way_num=opt.way_num)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=opt.testepisodeSize, shuffle=True,
            num_workers=int(opt.workers), drop_last=True, pin_memory=True
        )
        print('Testset: %d' % len(testset))
        print('dataset %s, with 5-way, %d-shot, %s metric.'%(opt.data_name, opt.shot_num, opt.metric))
        # prec1, accuracies = meta_test(test_loader, model, r, best_prec1, opt, train='test')
        prec1 = validate(test_loader, model, 0, best_prec1, opt, meters_dict, mode='Final test', inp_records=(None, None))
        test_accuracy, h = mean_confidence_interval(meters_dict['target_acc'].histories[-1])
        print("Test accuracy", test_accuracy, "h", h)
        total_accuracy += test_accuracy
        total_accuracy_vector.extend(meters_dict['target_acc'].histories[-1])
        total_h[r] = h

    aver_accuracy, _ = mean_confidence_interval(total_accuracy_vector)
    print("Aver_accuracy:", aver_accuracy, "Aver_h", total_h.mean())
    return {"aver_accuracy": aver_accuracy, "aver_h": total_h.mean()}

@torch.no_grad()
def validate(val_loader, model, epoch_index, best_prec1, opt, meters_dict, inp_records, mode='val'):
    # switch to evaluate mode
    model.eval()
    train_test_data, out_traintest_data_path = inp_records
    for episode_index, (query_images, query_targets, query_modal, support_images, support_targets, query_global_targets) in tqdm(enumerate(val_loader)):

        # Convert query and support images
        query_images = torch.squeeze(query_images.type(torch.FloatTensor))
        support_images = support_images.squeeze(0)
        input_var1 = query_images.cuda()
        input_var2 = support_images.cuda()
        query_images = torch.squeeze(query_images.type(torch.FloatTensor))
        support_images = support_images
        input_var1 = query_images.cuda()
        input_var2 = support_images.cuda()
        support_targets = support_targets.cuda()
        # Calculate the output
        query_modal = query_modal.cuda()
        query_targets = query_targets.cuda()
        loss_acc = model(query_x=input_var1, support_x=input_var2, query_y=query_targets, support_y=support_targets, query_m=query_modal, train=mode)

        # Measure accuracy and record loss
        out_line = accumulate_and_output_meters(loss_acc, meters_dict, epoch_index, episode_index, len(val_loader), mode)


        # ============== print the intermediate results ==============#
        if episode_index % opt.print_freq == 0 and episode_index != 0:
            print_and_record(out_line, opt.outf+'/out_line.txt')
            if mode in ['test', 'val']:
                train_test_data[mode] = {k: meters_dict[k].histories for k in meters_dict}
                out_put_metadata(train_test_data, out_traintest_data_path)


    print(' * Best prec1 {best_prec1:.3f}, now prec1 {now:.3f}'.format(best_prec1=best_prec1, now=meters_dict['target_acc'].epoch_avg(-1)))
    return meters_dict['target_acc'].epoch_avg(-1) # , meters_dict['cls loss']


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

# ======================================== Settings of path ============================================


import traceback
import shutil
def test(opt):
    try:
        start_time = time.time()
        test_(opt)
        run_time = time.time() - start_time
        print('running time= {time:.3f}'.format(time=run_time/60))
    except Exception as e:
        print(traceback.print_exc())
        shutil.rmtree(opt.outf, ignore_errors=True)
        father_dir = os.path.join('/', *opt.outf.split('/')[:-1])
        if not any([os.path.isdir(os.path.join(father_dir, _)) for _ in os.listdir(father_dir)]):
            shutil.rmtree(father_dir, ignore_errors=True)
        raise Exception