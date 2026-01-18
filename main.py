# from __future__ import print_function
# import datetime
# import torch.nn.parallel
# from model import hyper_model
# import torch.utils.data
# from PIL import ImageFile
# from data_generator import *

# import sys
# sys.dont_write_bytecode = True

# # ============================ Data & Networks =====================================
# from tqdm import tqdm
# # ==================================================================================
# from experiment_api.print_utils import accumulate_and_output_meters, print_and_record
# from experiment_api.out_files import out_put_metadata

# ImageFile.LOAD_TRUNCATED_IMAGES = True
# # os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# seed = 1024
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

# def pretrain(encoder, max_epochs):
#     fc_layer = nn.Linear(441*64, 64)
# # DEVICE = torch.device('cuda:0')

# import json
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('./runs/pics/img2class')
# import scipy as sp
# import scipy.stats
# import scipy


# def mean_confidence_interval(data, confidence=0.95):
#     a = [1.0*np.array(data[i]) for i in range(len(data))]
#     n = len(a)
#     m, se = np.mean(a), scipy.stats.sem(a)
#     h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
#     return m, h

# def test(opt):
#     # define loss function (criterion) and optimizer
#     # model.method.eval()
#     # optionally resume from a checkpoint
#     model = hyper_model.Trainer_new(opt)
#     model_dir = os.path.join(opt.outf, 'saved_models')
#     ckpt = os.path.join(model_dir, 'model_best.pth.tar')
#     if os.path.isfile(ckpt):
#         print("=> loading checkpoint '{}'".format(ckpt))
#         checkpoint = torch.load(ckpt)
#         epoch_index = checkpoint['epoch_index']
#         best_prec1 = checkpoint['best_prec1']
#         keys = checkpoint['state_dict'].keys()
#         checkpoint_tmp = {'state_dict': {}}
#         for k in keys:
#             k_tmp = k.replace('.module', '')
#             checkpoint_tmp['state_dict'][k_tmp] = checkpoint['state_dict'][k]
#         model.method.load_state_dict(checkpoint_tmp['state_dict'])
#         # model.g_optimizer.load_state_dict(checkpoint_tmp['optimizer'])
#         print("=> loaded checkpoint '{}' (epoch {})".format(ckpt, checkpoint['epoch_index']))
#     else:
#         print("=> no checkpoint found at '{}', use pretrained.".format(model_dir))
#         if opt.metric == 'DWT':
#             root_ = '/home/hsq/Projects/dwt-domain-adaptation/saved_model/conv_weights'
#             if opt.data_name == 'visda':
#                 model_path = os.path.join(root_, 'visda_%s_to_%s_conv64f_best.pth.tar' % (opt.source_domain, opt.target_domain))
#             elif opt.data_name == 'mini-Imagenet':
#                 model_path = os.path.join(root_, 'mini-imagenet_ycm_%s_to_%s_conv64f_best.pth.tar' % (opt.source_domain, opt.target_domain))
#             elif opt.data_name == 'officehome':
#                 model_path = os.path.join(root_, 'officehome_%s_to_%s_conv64f_best.pth.tar' % (opt.source_domain, opt.target_domain))
#             else:
#                 raise ValueError('Error dataset name.')
#             model_dict = model.method.gen.state_dict()
#             ckpt = torch.load(model_path, map_location='cuda:0')
#             pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#             if len(pretrained_dict) == 0:
#                 raise ValueError('Error ckpt.')
#             model_dict.update(pretrained_dict)
#             model.method.gen.load_state_dict(model_dict)
#             print('load pretrained model from %s successly.'%root_)
#         elif opt.metric in ('MCD', 'MCDR2D2'):
#             root_ = '/home/hsq/Projects/IMSE/MCD_pretrained'
#             if opt.data_name == 'visda':
#                 model_path = os.path.join(root_, 'MCD_%s_to_%s_model_conv64f.pt' % (
#                 opt.source_domain, opt.target_domain))
#             elif opt.data_name == 'mini-Imagenet':
#                 model_path = os.path.join(root_, 'mini-imagenet_ycm_MCD_%s_to_%s_model_conv64f.pth.tar' % (
#                 opt.source_domain, opt.target_domain))
#             elif opt.data_name == 'officehome':
#                 model_path = os.path.join(root_, 'officehome_MCD_%s_to_%s_model_conv64f.pth.tar' % (
#                 opt.source_domain, opt.target_domain))
#             else:
#                 raise ValueError('Error dataset name.')
#             model_dict = model.method.gen.state_dict()
#             ckpt = torch.load(model_path, map_location='cuda:0')
#             pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#             if len(pretrained_dict) == 0:
#                 raise ValueError('Error ckpt.')
#             model_dict.update(pretrained_dict)
#             model.method.gen.load_state_dict(model_dict)
#             print('load pretrained model from %s successly.' % root_)
#         elif opt.metric == 'DANN':
#             root_ = '/home/hsq/Projects/IMSE/DANN_pretrained'
#             if opt.data_name == 'visda':
#                 model_path = os.path.join(root_, 'DANN_%s_to_%s_model_conv64f.pth' % (
#                 opt.source_domain, opt.target_domain))
#             elif opt.data_name == 'mini-Imagenet':
#                 model_path = os.path.join(root_, 'mini-imagenet_ycm_DANN_conv64f_%s_to_%s.pth' % (
#                 opt.source_domain, opt.target_domain))
#             elif opt.data_name == 'officehome':
#                 model_path = os.path.join(root_, 'officehome_DANN_conv64f_%s_to_%s.pth' % (
#                 opt.source_domain, opt.target_domain))
#             else:
#                 raise ValueError('Error dataset name.')
#             model_dict = model.method.gen.state_dict()
#             ckpt = torch.load(model_path, map_location='cuda:0')
#             pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#             if len(pretrained_dict) == 0:
#                 raise ValueError('Error ckpt.')
#             model_dict.update(pretrained_dict)
#             model.method.gen.load_state_dict(model_dict)
#             print('load pretrained model from %s successly.' % root_)
#         elif opt.metric == 'ADDA':
#             root_ = '/home/hsq/Projects/IMSE/ADDA_pretrained'
#             if opt.data_name == 'visda':
#                 model_path = os.path.join(root_, 'visda_ADDA_conv64f_%s_to_%s_epoch_19.pth' % (
#                 opt.source_domain, opt.target_domain))
#             elif opt.data_name == 'mini-Imagenet':
#                 model_path = os.path.join(root_, 'mini-imagenet_ycm_ADDA_conv64f_%s_to_%s_epoch_19.pth' % (
#                 opt.source_domain, opt.target_domain))
#             elif opt.data_name == 'officehome':
#                 model_path = os.path.join(root_, 'officehome_ADDA_conv64f_%s_to_%s_epoch_19.pth' % (
#                 opt.source_domain, opt.target_domain))
#             else:
#                 raise ValueError('Error dataset name.')
#             model_dict = model.method.gen.state_dict()
#             ckpt = torch.load(model_path, map_location='cuda:0')
#             pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#             if len(pretrained_dict) == 0:
#                 raise ValueError('Error ckpt.')
#             model_dict.update(pretrained_dict)
#             model.method.gen.load_state_dict(model_dict)
#             print('load pretrained model from %s successly.' % root_)

#         elif opt.rotate:
#             root_ = '/home/hsq/Projects/rotate_pretrained'
#             if opt.data_name == 'visda':
#                 model_path = os.path.join(root_, 'visda_%s_to_%s_rotate_resnet_best.pth.tar' % (
#                 opt.source_domain, opt.target_domain))
#             elif opt.data_name == 'mini-Imagenet':
#                 model_path = os.path.join(root_, 'mini-imagenet_ycm_%s_to_%s_conv64f_best.pth.tar' % (
#                 opt.source_domain, opt.target_domain))
#             elif opt.data_name == 'officehome':
#                 model_path = os.path.join(root_, 'officehome_%s_to_%s_conv64f_best.pth.tar' % (
#                 opt.source_domain, opt.target_domain))
#             else:
#                 raise ValueError('Error dataset name.')
#             model_dict = model.method.gen.state_dict()
#             ckpt = torch.load(model_path, map_location='cuda:0')
#             pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
#             a_dict = {}
#             for k in pretrained_dict:
#                 new_k = k.replace('module.', '')
#                 a_dict[new_k] = pretrained_dict[k]
#                 # pretrained_dict.pop(k)
#             pretrained_dict = a_dict
#             # print(pretrained_dict.keys())
#             # exit()
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#             if len(pretrained_dict) == 0:
#                 raise ValueError('Error ckpt.')
#             model_dict.update(pretrained_dict)
#             model.method.gen.load_state_dict(model_dict)
#             print('load pretrained model from %s successly.' % root_)
#         elif 'Baseline' in opt.metric:
#             root_ = '/home/hsq/Projects/rotate_pretrained'
#             if opt.data_name == 'visda':
#                 model_path = os.path.join(root_, 'visda_%s_to_%s_normal_resnet_best.pth.tar' % (
#                     opt.source_domain, opt.target_domain))
#             elif opt.data_name == 'mini-Imagenet':
#                 model_path = os.path.join(root_, 'mini-imagenet_ycm_%s_to_%s_conv64f_best.pth.tar' % (
#                     opt.source_domain, opt.target_domain))
#             elif opt.data_name == 'officehome':
#                 model_path = os.path.join(root_, 'officehome_%s_to_%s_conv64f_best.pth.tar' % (
#                     opt.source_domain, opt.target_domain))
#             else:
#                 raise ValueError('Error dataset name.')
#             model_dict = model.method.gen.state_dict()
#             ckpt = torch.load(model_path, map_location='cuda:0')
#             pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
#             a_dict = {}
#             for k in pretrained_dict:
#                 new_k = k.replace('module.', '')
#                 a_dict[new_k] = pretrained_dict[k]
#                 # pretrained_dict.pop(k)
#             pretrained_dict = a_dict
#             # print(pretrained_dict.keys())
#             # exit()
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#             if len(pretrained_dict) == 0:
#                 raise ValueError('Error ckpt.')
#             model_dict.update(pretrained_dict)
#             model.method.gen.load_state_dict(model_dict)
#             print('load pretrained model from %s successly.' % root_)

#         elif opt.backbone == 'ResNet12':
#             # ckpt = torch.load('pretrained_model/mini_resnet_model_3pool_epoch161.pth.tar')

#             def load_model(model, dir):
#                 model_dict = model.state_dict()
#                 print('loading model from :', dir)
#                 pretrained_dict = torch.load(dir)['params']
#                 if 'encoder' in list(pretrained_dict.keys())[0]:  # load from a parallel meta-trained model
#                     if 'module' in list(pretrained_dict.keys())[0]:
#                         pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
#                     else:
#                         pretrained_dict = {k: v for k, v in pretrained_dict.items()}
#                 else:
#                     pretrained_dict = { k: v for k, v in
#                                        pretrained_dict.items()}  # load from a pretrained model
#                 pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#                 # print(pretrained_dict)
#                 model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
#                 model.load_state_dict(model_dict)
#             if opt.data_name == 'officehome':
#                 model_dict = model.method.gen.state_dict()
#                 ckpt = torch.load('./pretrained_model/officehome_resnet12_model_3pool_epoch90.pth.tar')
#                 pretrained_dict = ckpt['state_dict']
#                 pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#                 model_dict.update(pretrained_dict)
#                 model.method.gen.load_state_dict(model_dict)
#             if opt.data_name == 'visda':
#                 model_dict = model.method.gen.state_dict()
#                 if opt.combine_uda_fsl:
#                     # e_dir = '/home/hsq/Projects/MCD_DA/classification/pretrained/MCD_%s_to_%s_resnet.pth.tar'%(opt.source_domain, opt.target_domain, )
#                     e_dir = 'ADDA_pretrained/%s2%s-ADDA-target-encoder-final.pt'%(opt.source_domain, opt.target_domain, )
#                     # e_dir = '/home/hsq/Projects/FSUDA_lds_spatial/ADDA_pretrained/%s2%s-ADDA-target-encoder-final.pt'% (opt.source_domain, opt.target_domain, )
#                     ckpt = torch.load(e_dir)
#                     n_ckpt = {}
#                     for k in ckpt.keys():
#                         n_k = k.replace('module.', '')
#                         n_ckpt[n_k] = ckpt[k]
#                     ckpt = {'state_dict': n_ckpt}
#                     print('load from %s' % (e_dir))
#                 else:
#                     ckpt = torch.load('./pretrained_model/visda_%s_resnet12_model_3pool.pth.tar' % opt.source_domain)
#                 pretrained_dict = ckpt['state_dict']
#                 pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#                 model_dict.update(pretrained_dict)
#                 model.method.gen.load_state_dict(model_dict)
#                 if pretrained_dict.__len__() != 0:
#                     print('load pretrained model success.')
#                 else:
#                     print('load pretrained model failed.')
#             elif opt.data_name == 'tiered-Imagenet':
#                 load_model(model.method.gen, 'pretrained_model/tieredImagenet_resnet12.pth')
#             else:
#                 load_model(model.method.gen, 'pretrained_model/epoch-64.pth')
#         else:
#             if opt.data_name == 'visda':
#                 model_path = os.path.join('/home/hsq/Projects/pretrained_models', 'visda_%s_100epoch.pth.tar'%opt.source_domain)
#             elif opt.data_name == 'mini-Imagenet':
#                 model_path = os.path.join('/home/hsq/Projects/pretrained_models', 'mini-Imagenet_%s_50epoch.pth.tar' % opt.source_domain)
#             elif opt.data_name == 'officehome':
#                 model_path = os.path.join('/home/hsq/Projects/pretrained_models', 'officehome_%s_30epoch.pth.tar' % opt.source_domain)
#             else:
#                 raise ValueError('Error dataset name.')
#             model_dict = model.method.gen.state_dict()
#             ckpt = torch.load(model_path)
#             pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#             if len(pretrained_dict) == 0:
#                 raise ValueError('Error ckpt.')
#             model_dict.update(pretrained_dict)
#             model.method.gen.load_state_dict(model_dict)
#             print('load pretrained model success.')

#             # model.method.gen.load_state_dict(ckpt['state_dict'])


#     if len(opt.gpu) > 1:
#         gid = range(opt.gpu.split(',').__len__())
#         model.method.gen = nn.DataParallel(model.method.gen, [int(g) for g in gid],)
#         model.method.imgtoclass = nn.DataParallel(model.method.imgtoclass, [int(g) for g in gid],)
#         model.method.discri = nn.DataParallel(model.method.discri, [int(g) for g in gid],)
#     model.cuda()
#     model.eval()

#     repeat_num = 5
#     best_prec1 = 0.0
#     total_accuracy = 0.0
#     total_h = np.zeros(repeat_num)
#     total_accuracy_vector = []
#     meters_dict = {}
#     for r in range(repeat_num):
#         data_dir = opt.dataset_dir+'/'+opt.data_name
#         testset = DataGenerator(mode='test', datasource=opt.data_name, data_dir=data_dir, imagesize=opt.imageSize,
#                       episode=opt.final_test_episode_num, support_num=opt.shot_num, query_num=opt.query_num, split_PATH=opt.split_PATH,
#                                 source_domain=opt.source_domain, target_domain=opt.target_domain, way_num=opt.way_num)
#         test_loader = torch.utils.data.DataLoader(
#             testset, batch_size=opt.testepisodeSize, shuffle=True,
#             num_workers=int(opt.workers), drop_last=True, pin_memory=True
#         )
#         print('Testset: %d' % len(testset))
#         print('dataset %s, with 5-way, %d-shot, %s metric.'%(opt.data_name, opt.shot_num, opt.metric))
#         # prec1, accuracies = meta_test(test_loader, model, r, best_prec1, opt, train='test')
#         prec1 = validate(test_loader, model, 0, best_prec1, opt, meters_dict, mode='Final test', inp_records=(None, None))
#         test_accuracy, h = mean_confidence_interval(meters_dict['target_acc'].histories[-1])
#         print("Test accuracy", test_accuracy, "h", h)
#         total_accuracy += test_accuracy
#         total_accuracy_vector.extend(meters_dict['target_acc'].histories[-1])
#         total_h[r] = h

#     aver_accuracy, _ = mean_confidence_interval(total_accuracy_vector)
#     print("Aver_accuracy:", aver_accuracy, "Aver_h", total_h.mean())
#     return {"aver_accuracy": aver_accuracy, "aver_h": total_h.mean()}


# def train(train_loader, model, epoch_index, opt, meters_dict, inp_records, mode='train'):
#     train_test_data, out_traintest_data_path = inp_records
#     for episode_index, (query_images, query_targets, query_modal, support_images, support_targets, query_global_targets) in tqdm(enumerate(train_loader)):

#         # Convert query and support images

#         query_images = torch.squeeze(query_images.type(torch.FloatTensor))
#         # input_var2 = [s[0].cuda() for s in support_images]
#         support_images = support_images.squeeze(0)
#         input_var2 = support_images.cuda()

#         input_var1 = query_images.cuda()
#         query_modal = query_modal.cuda()
#         query_targets = query_targets.cuda()
#         query_global_targets = query_global_targets.cuda()
#         loss_acc = model(query_x=input_var1, support_x=input_var2, query_y=query_targets, query_m=query_modal, global_label=query_global_targets)
#         # print(model.method.imgtoclass.module.maps)


#         # Measure accuracy and record loss
#         out_line = accumulate_and_output_meters(loss_acc, meters_dict, epoch_index, episode_index, len(train_loader), mode)

#         # ============== print the intermediate results ==============#
#         if episode_index % opt.print_freq == 0 and episode_index != 0:
#             print_and_record(out_line, opt.outf+'/out_line.txt')
#             train_test_data['train'] = {k: meters_dict[k].histories for k in meters_dict}
#             out_put_metadata(train_test_data, out_traintest_data_path)


# @torch.no_grad()
# def validate(val_loader, model, epoch_index, best_prec1, opt, meters_dict, inp_records, mode='val'):
#     # switch to evaluate mode
#     model.eval()
#     train_test_data, out_traintest_data_path = inp_records
#     for episode_index, (query_images, query_targets, query_modal, support_images, support_targets, query_global_targets) in tqdm(enumerate(val_loader)):

#         # Convert query and support images
#         query_images = torch.squeeze(query_images.type(torch.FloatTensor))
#         support_images = support_images.squeeze(0)
#         input_var1 = query_images.cuda()
#         input_var2 = support_images.cuda()

#         # Calculate the output
#         query_modal = query_modal.cuda()
#         query_targets = query_targets.cuda()
#         loss_acc = model(query_x=input_var1, support_x=input_var2, query_y=query_targets, query_m=query_modal, train=mode)


#         # Measure accuracy and record loss
#         out_line = accumulate_and_output_meters(loss_acc, meters_dict, epoch_index, episode_index, len(val_loader), mode)


#         # ============== print the intermediate results ==============#
#         if episode_index % opt.print_freq == 0 and episode_index != 0:
#             print_and_record(out_line, opt.outf+'/out_line.txt')
#             if mode in ['test', 'val']:
#                 train_test_data[mode] = {k: meters_dict[k].histories for k in meters_dict}
#                 out_put_metadata(train_test_data, out_traintest_data_path)


#     print(' * Best prec1 {best_prec1:.3f}, now prec1 {now:.3f}'.format(best_prec1=best_prec1, now=meters_dict['target_acc'].epoch_avg(-1)))
#     return meters_dict['target_acc'].epoch_avg(-1)


# def save_checkpoint(state, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)


# # ======================================== Settings of path ============================================



# import nni
# # ========================================== Model Config ===============================================
# def main_(opt):
#     print('loss weight is:', opt.loss_weight)
#     if opt.auto_tune:
#         opt.outf = os.path.join(opt.outf, nni.get_trial_id())
#     else:
#         opt.outf = os.path.join(opt.outf, str(opt.exp_id))
#     if not os.path.exists(opt.outf):
#         os.makedirs(opt.outf)
#     print(opt.exp_id, opt.outf)
#     json.dump(vars(opt), open(os.path.join(opt.outf, 'exp_args.json'), 'w'))
    
#     if torch.cuda.is_available() and not opt.cuda:
#         print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#     # save the opt and results to a txt file
#     print(opt)
#     ngpu = int(opt.ngpu)
#     global best_prec1, epoch_index
#     best_prec1 = 0
#     best_test = 0
    
#     meta_data = {
#         'metric': opt.metric,
#         'epochs': opt.epochs,
#         'start_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
#         'end_time': '-1',
#         'progress': 1,
#         'bestacc': 0,
#     }
#     train_meters, test_meters, val_meters = {}, {}, {}
#     train_test_data = {'train': {}, 'test': {}, 'val': {}}
#     out_metadata_path = opt.outf+'/meta_data.json'
#     out_traintest_data_path = opt.outf+'/train_test.json'
    
#     print(out_metadata_path, out_traintest_data_path)
#     out_put_metadata(meta_data, out_metadata_path)
#     out_put_metadata(train_test_data, out_traintest_data_path)
#     model = hyper_model.Trainer_new(opt=opt)
#     model.method.imgtoclass.writer = writer

#     if opt.pretrained and opt.metric not in ('MCD', 'DWT', 'DANN'):
#         if opt.backbone == 'ResNet12':
#             # ckpt = torch.load('pretrained_model/mini_resnet_model_3pool_epoch161.pth.tar')

#             def load_model(model, dir):
#                 model_dict = model.state_dict()
#                 print('loading model from :', dir)
#                 pretrained_dict = torch.load(dir)['params']
#                 if 'encoder' in list(pretrained_dict.keys())[0]:  # load from a parallel meta-trained model
#                     if 'module' in list(pretrained_dict.keys())[0]:
#                         pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
#                     else:
#                         pretrained_dict = {k: v for k, v in pretrained_dict.items()}
#                 else:
#                     pretrained_dict = { k: v for k, v in
#                                        pretrained_dict.items()}  # load from a pretrained model
#                 pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#                 # print(pretrained_dict)
#                 model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
#                 model.load_state_dict(model_dict)
#             if opt.rotate:
#                 root_ = '/home/hsq/Projects/rotate_pretrained'
#                 if opt.data_name == 'visda':
#                     model_path = os.path.join(root_, 'visda_%s_to_%s_rotate_resnet_best.pth.tar' % (
#                     opt.source_domain, opt.target_domain))
#                 elif opt.data_name == 'mini-Imagenet':
#                     model_path = os.path.join(root_, 'mini-imagenet_ycm_%s_to_%s_conv64f_best.pth.tar' % (
#                     opt.source_domain, opt.target_domain))
#                 elif opt.data_name == 'officehome':
#                     model_path = os.path.join(root_, 'officehome_%s_to_%s_conv64f_best.pth.tar' % (
#                     opt.source_domain, opt.target_domain))
#                 else:
#                     raise ValueError('Error dataset name.')
#                 model_dict = model.method.gen.state_dict()
#                 ckpt = torch.load(model_path, map_location='cuda:0')
#                 pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
#                 a_dict = {}
#                 for k in pretrained_dict:
#                     new_k = k.replace('module.', '')
#                     a_dict[new_k] = pretrained_dict[k]
#                     # pretrained_dict.pop(k)
#                 pretrained_dict = a_dict
#                 # print(pretrained_dict.keys())
#                 # exit()
#                 pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#                 if len(pretrained_dict) == 0:
#                     raise ValueError('Error ckpt.')
#                 model_dict.update(pretrained_dict)
#                 model.method.gen.load_state_dict(model_dict)
#                 print('load pretrained rotate model from %s successly.' % root_)
#             elif opt.data_name == 'officehome':
#                 model_dict = model.method.gen.state_dict()
#                 ckpt = torch.load('./pretrained_model/officehome_resnet12_model_3pool_epoch90.pth.tar')
#                 pretrained_dict = ckpt['state_dict']
#                 pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#                 model_dict.update(pretrained_dict)
#                 model.method.gen.load_state_dict(model_dict)
#             elif opt.data_name == 'visda':
#                 model_dict = model.method.gen.state_dict()
#                 if opt.combine_uda_fsl:
#                     # e_dir = '/home/hsq/Projects/MCD_DA/classification/pretrained/MCD_%s_to_%s_resnet.pth.tar'%(opt.source_domain, opt.target_domain, )
#                     e_dir = 'ADDA_pretrained/%s2%s-ADDA-target-encoder-final.pt'%(opt.source_domain, opt.target_domain, )
#                     # e_dir = '/home/hsq/Projects/FSUDA_lds_spatial/ADDA_pretrained/%s2%s-ADDA-target-encoder-final.pt'% (opt.source_domain, opt.target_domain, )
#                     ckpt = torch.load(e_dir)
#                     n_ckpt = {}
#                     for k in ckpt.keys():
#                         n_k = k.replace('module.', '')
#                         n_ckpt[n_k] = ckpt[k]
#                     ckpt = {'state_dict': n_ckpt}
#                     print('load from %s' % (e_dir))            
#                 else:
#                     ckpt = torch.load('/home/hsq/Projects/pretrained_models/visda_resnet_%s_100epoch.pth.tar' % opt.source_domain)
#                 pretrained_dict = ckpt
#                 pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#                 model_dict.update(pretrained_dict)
#                 model.method.gen.load_state_dict(model_dict)
#                 if pretrained_dict.__len__() != 0:
#                     print('load pretrained model success.')
#                 else:
#                     print('load pretrained model failed.')
#             elif opt.data_name == 'tiered-Imagenet':
#                 load_model(model.method.gen, '/home/hsq/NAS/Projects/pretrained_model/tieredImagenet_resnet12.pth')
#             else:
#                 load_model(model.method.gen, '/home/hsq/NAS/Projects/pretrained_model/epoch-64.pth')

#         else:
#             if opt.data_name == 'visda':
#                 model_path = os.path.join('/home/hsq/NAS/Projects/pretrained_models', 'visda_%s_100epoch.pth.tar'%opt.source_domain)
#             elif opt.data_name == 'mini-Imagenet':
#                 model_path = os.path.join('/home/hsq/NAS/Projects/pretrained_models', 'mini-Imagenet_%s_50epoch.pth.tar' % opt.source_domain)
#             elif opt.data_name == 'officehome':
#                 model_path = os.path.join('/home/hsq/NAS/Projects/pretrained_models', 'officehome_%s_30epoch.pth.tar' % opt.source_domain)

#             model_dict = model.method.gen.state_dict()
#             ckpt = torch.load(model_path)
#             pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#             if len(pretrained_dict) == 0:
#                 raise ValueError('Error ckpt.')
#             model_dict.update(pretrained_dict)
#             model.method.gen.load_state_dict(model_dict)
#             print('load pretrained model success.')

#             # model.method.gen.load_state_dict(ckpt['state_dict'])

#     model.cuda()
#     # define loss function (criterion) and optimizer
#     # optionally resume from a checkpoint
#     if len(opt.gpu) > 1 and opt.multigpu:
#         gpuid = range(opt.gpu.split(',').__len__())
#         model.method.gen = nn.DataParallel(model.method.gen, [int(g) for g in gpuid],)
#         model.method.imgtoclass = nn.DataParallel(model.method.imgtoclass, [int(g) for g in gpuid],)
#         model.method.discri = nn.DataParallel(model.method.discri, [int(g) for g in gpuid],)
#         # model.method.cluster = nn.DataParallel(model.method.cluster, [int(g) for g in gpuid])
#     model.eval()

#     # ======================================== Training phase ===============================================
#     print('\n............Start training............\n')
#     for epoch_item in range(opt.pretrain, opt.epochs):
#         if 'Baseline' in opt.metric or opt.metric in ('MCD', 'DWT', 'DANN', 'MCDR2D2', 'ADDA') or opt.finetune:
#             break
#         print('===================================== Epoch %d =====================================' % epoch_item)
#         model.adjust_learning_rate(epoch_item)

#         # ======================================= Folder of Datasets =======================================
#         data_dir = opt.dataset_dir+'/'+opt.data_name
#         trainset = DataGenerator(mode='train', datasource=opt.data_name,  data_dir=data_dir, imagesize=opt.imageSize,
#             episode=opt.episode_train_num, support_num=opt.shot_num, query_num=opt.query_num, split_PATH=opt.split_PATH,
#                                  source_domain=opt.source_domain, target_domain=opt.target_domain, way_num=opt.way_num, random_target=opt.random_target)

#         valset = DataGenerator(mode='val', datasource=opt.data_name, data_dir=data_dir, imagesize=opt.imageSize,
#                       episode=opt.episode_val_num, support_num=opt.shot_num, query_num=opt.query_num, split_PATH=opt.split_PATH,
#                                source_domain=opt.source_domain, target_domain=opt.target_domain, way_num=opt.way_num)

#         testset = DataGenerator(mode='test', datasource=opt.data_name, data_dir=data_dir, imagesize=opt.imageSize,
#                       episode=opt.episode_test_num, support_num=opt.shot_num, query_num=opt.query_num, split_PATH=opt.split_PATH,
#                                 source_domain=opt.source_domain, target_domain=opt.target_domain, way_num=opt.way_num)

#         print('Trainset: %d' % len(trainset))
#         print('Valset: %d' % len(valset))
#         print('Testset: %d' % len(testset))

#         # ========================================== Load Datasets =========================================
#         train_loader = torch.utils.data.DataLoader(
#             trainset, batch_size=opt.episodeSize, shuffle=False,
#             num_workers=int(opt.workers), drop_last=False, pin_memory=False
#         )
#         val_loader = torch.utils.data.DataLoader(
#             valset, batch_size=opt.testepisodeSize, shuffle=False,
#             num_workers=int(opt.workers), drop_last=False, pin_memory=False
#         )
#         test_loader = torch.utils.data.DataLoader(
#             testset, batch_size=opt.testepisodeSize, shuffle=False,
#             num_workers=int(opt.workers), drop_last=False, pin_memory=False
#         )

#         # ============================================ Training ===========================================
#         # Fix the parameters of Batch Normalization after 10000 episodes (1 epoch)
#         if opt.metric in ('DeepEMD', 'MetaBaseline', 'DSN') or (opt.on_bn and epoch_item < 1):
#             model.train()
#         else:
#             model.eval()

#         # Train for 10000 episodes in each epoch
#         train(train_loader, model, epoch_item, opt, train_meters, (train_test_data, out_traintest_data_path))
#         # print('dataset %s, with 5-way, %d-shot, %s metric.'%(opt.data_name, opt.shot_num, opt.metric))
#         model.eval()
#         # =========================================== Evaluation ==========================================
#         print('============ Validation on the val set ============')
#         prec1 = validate(val_loader, model, epoch_item, best_prec1, opt, val_meters, (train_test_data, out_traintest_data_path), mode='val')
#         # record the best prec@1 and save checkpoint
#         # Testing Prase
#         print('============ Testing on the test set ============')
#         prec_test = validate(test_loader, model, epoch_item, best_prec1, opt, test_meters, (train_test_data, out_traintest_data_path), mode='test')
#         if opt.auto_tune: nni.report_intermediate_result(prec_test)
#         is_best = prec1 > best_prec1
#         best_prec1 = max(prec1, best_prec1)
#         best_test = max(prec_test, best_test)

#         # save the checkpoint
#         if not os.path.exists(os.path.join(opt.outf, 'saved_models')):
#             os.mkdir(os.path.join(opt.outf, 'saved_models'))
#         if is_best:
#             save_checkpoint(
#                 {
#                     'epoch_index': epoch_item,
#                     'arch': opt.basemodel,
#                     'state_dict': model.method.state_dict(),
#                     'best_prec1': best_prec1,
#                     'optimizer': model.g_optimizer.state_dict(),
#                 }, os.path.join(opt.outf, 'saved_models', 'model_best.pth.tar'))
#         meta_data['progress'] = int(100 * (epoch_item + 1) / meta_data['epochs'])
#         meta_data['bestacc'] = best_test
#         # train_test_data['train'] = {k:train_meters[k].histories for k in train_meters}
#         # train_test_data['test'] = {k:test_meters[k].histories for k in test_meters}
#         # train_test_data['val'] = {k:val_meters[k].histories for k in val_meters}
#         # out_put_metadata(train_test_data, out_traintest_data_path)
#         out_put_metadata(meta_data, out_metadata_path)
#     print('............Training is end............')
#     # ============================================ Training End ==============================================================

#     print('............Now meta testing............')
#     res = test(opt)
#     res['val_acc'] = best_prec1
#     print(res)
#     torch.cuda.empty_cache()
#     meta_data['final_test_acc'] = res
#     out_put_metadata(meta_data, out_metadata_path)
#     if opt.auto_tune: nni.report_final_result(res['aver_accuracy'])
#     return res

# def main(opt):
#     try:
#         main_(opt)
#     except Exception as e:
#         # remove the files in opt.outf
#         print(opt.outf)
#         print(e)


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

def test(opt):
    # define loss function (criterion) and optimizer
    # model.method.eval()
    # optionally resume from a checkpoint
    model = hyper_model.Trainer(opt)
    model_dir = os.path.join(opt.outf, 'saved_models')
    ckpt = os.path.join(model_dir, 'model_best.pth.tar')
    if os.path.isfile(ckpt):
        print("=> loading checkpoint '{}'".format(ckpt))
        checkpoint = torch.load(ckpt)
        epoch_index = checkpoint['epoch_index']
        best_prec1 = checkpoint['best_prec1']       
        
        model_dict = model.method.state_dict()
        model_dict.update(checkpoint['state_dict'])
        model.method.load_state_dict(model_dict)

        # model.g_optimizer.load_state_dict(checkpoint_tmp['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(ckpt, checkpoint['epoch_index']))
    else:
        print("=> no checkpoint found at '{}', use pretrained.".format(model_dir))
        if opt.metric == 'DWT':
            root_ = '/home/hsq/Projects/dwt-domain-adaptation/saved_model/conv_weights'
            if opt.data_name == 'visda':
                model_path = os.path.join(root_, 'visda_%s_to_%s_conv64f_best.pth.tar' % (opt.source_domain, opt.target_domain))
            elif opt.data_name == 'mini-Imagenet':
                model_path = os.path.join(root_, 'mini-imagenet_ycm_%s_to_%s_conv64f_best.pth.tar' % (opt.source_domain, opt.target_domain))
            elif opt.data_name == 'officehome':
                model_path = os.path.join(root_, 'officehome_%s_to_%s_conv64f_best.pth.tar' % (opt.source_domain, opt.target_domain))
            else:
                raise ValueError('Error dataset name.')
            model_dict = model.method.gen.state_dict()
            ckpt = torch.load(model_path, map_location='cuda:0')
            pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if len(pretrained_dict) == 0:
                raise ValueError('Error ckpt.')
            model_dict.update(pretrained_dict)
            model.method.gen.load_state_dict(model_dict)
            print('load pretrained model from %s successly.'%root_)
        elif opt.metric in ('MCD', 'MCDR2D2'):
            root_ = '/home/hsq/Projects/IMSE/MCD_pretrained'
            if opt.data_name == 'visda':
                model_path = os.path.join(root_, 'MCD_%s_to_%s_model_conv64f.pt' % (
                opt.source_domain, opt.target_domain))
            elif opt.data_name == 'mini-Imagenet':
                model_path = os.path.join(root_, 'mini-imagenet_ycm_MCD_%s_to_%s_model_conv64f.pth.tar' % (
                opt.source_domain, opt.target_domain))
            elif opt.data_name == 'officehome':
                model_path = os.path.join(root_, 'officehome_MCD_%s_to_%s_model_conv64f.pth.tar' % (
                opt.source_domain, opt.target_domain))
            else:
                raise ValueError('Error dataset name.')
            model_dict = model.method.gen.state_dict()
            ckpt = torch.load(model_path, map_location='cuda:0')
            pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if len(pretrained_dict) == 0:
                raise ValueError('Error ckpt.')
            model_dict.update(pretrained_dict)
            model.method.gen.load_state_dict(model_dict)
            print('load pretrained model from %s successly.' % root_)
        elif opt.metric == 'DANN':
            root_ = '/home/hsq/Projects/IMSE/DANN_pretrained'
            if opt.data_name == 'visda':
                model_path = os.path.join(root_, 'DANN_%s_to_%s_model_conv64f.pth' % (
                opt.source_domain, opt.target_domain))
            elif opt.data_name == 'mini-Imagenet':
                model_path = os.path.join(root_, 'mini-imagenet_ycm_DANN_conv64f_%s_to_%s.pth' % (
                opt.source_domain, opt.target_domain))
            elif opt.data_name == 'officehome':
                model_path = os.path.join(root_, 'officehome_DANN_conv64f_%s_to_%s.pth' % (
                opt.source_domain, opt.target_domain))
            else:
                raise ValueError('Error dataset name.')
            model_dict = model.method.gen.state_dict()
            ckpt = torch.load(model_path, map_location='cuda:0')
            pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if len(pretrained_dict) == 0:
                raise ValueError('Error ckpt.')
            model_dict.update(pretrained_dict)
            model.method.gen.load_state_dict(model_dict)
            print('load pretrained model from %s successly.' % root_)
        elif opt.metric == 'ADDA':
            root_ = '/home/hsq/Projects/IMSE/ADDA_pretrained'
            if opt.data_name == 'visda':
                model_path = os.path.join(root_, 'visda_ADDA_conv64f_%s_to_%s_epoch_19.pth' % (
                opt.source_domain, opt.target_domain))
            elif opt.data_name == 'mini-Imagenet':
                model_path = os.path.join(root_, 'mini-imagenet_ycm_ADDA_conv64f_%s_to_%s_epoch_19.pth' % (
                opt.source_domain, opt.target_domain))
            elif opt.data_name == 'officehome':
                model_path = os.path.join(root_, 'officehome_ADDA_conv64f_%s_to_%s_epoch_19.pth' % (
                opt.source_domain, opt.target_domain))
            else:
                raise ValueError('Error dataset name.')
            model_dict = model.method.gen.state_dict()
            ckpt = torch.load(model_path, map_location='cuda:0')
            pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if len(pretrained_dict) == 0:
                raise ValueError('Error ckpt.')
            model_dict.update(pretrained_dict)
            model.method.gen.load_state_dict(model_dict)
            print('load pretrained model from %s successly.' % root_)

        elif opt.rotate:
            root_ = '/home/hsq/Projects/rotate_pretrained'
            if opt.data_name == 'visda':
                model_path = os.path.join(root_, 'visda_%s_to_%s_rotate_resnet_best.pth.tar' % (
                opt.source_domain, opt.target_domain))
            elif opt.data_name == 'mini-Imagenet':
                model_path = os.path.join(root_, 'mini-imagenet_ycm_%s_to_%s_conv64f_best.pth.tar' % (
                opt.source_domain, opt.target_domain))
            elif opt.data_name == 'officehome':
                model_path = os.path.join(root_, 'officehome_%s_to_%s_conv64f_best.pth.tar' % (
                opt.source_domain, opt.target_domain))
            else:
                raise ValueError('Error dataset name.')
            model_dict = model.method.gen.state_dict()
            ckpt = torch.load(model_path, map_location='cuda:0')
            pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            a_dict = {}
            for k in pretrained_dict:
                new_k = k.replace('module.', '')
                a_dict[new_k] = pretrained_dict[k]
                # pretrained_dict.pop(k)
            pretrained_dict = a_dict
            # print(pretrained_dict.keys())
            # exit()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if len(pretrained_dict) == 0:
                raise ValueError('Error ckpt.')
            model_dict.update(pretrained_dict)
            model.method.gen.load_state_dict(model_dict)
            print('load pretrained model from %s successly.' % root_)
        elif 'Baseline' in opt.metric:
            root_ = '/home/hsq/Projects/rotate_pretrained'
            if opt.data_name == 'visda':
                model_path = os.path.join(root_, 'visda_%s_to_%s_normal_resnet_best.pth.tar' % (
                    opt.source_domain, opt.target_domain))
            elif opt.data_name == 'mini-Imagenet':
                model_path = os.path.join(root_, 'mini-imagenet_ycm_%s_to_%s_conv64f_best.pth.tar' % (
                    opt.source_domain, opt.target_domain))
            elif opt.data_name == 'officehome':
                model_path = os.path.join(root_, 'officehome_%s_to_%s_conv64f_best.pth.tar' % (
                    opt.source_domain, opt.target_domain))
            else:
                raise ValueError('Error dataset name.')
            model_dict = model.method.gen.state_dict()
            ckpt = torch.load(model_path, map_location='cuda:0')
            pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            a_dict = {}
            for k in pretrained_dict:
                new_k = k.replace('module.', '')
                a_dict[new_k] = pretrained_dict[k]
                # pretrained_dict.pop(k)
            pretrained_dict = a_dict
            # print(pretrained_dict.keys())
            # exit()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if len(pretrained_dict) == 0:
                raise ValueError('Error ckpt.')
            model_dict.update(pretrained_dict)
            model.method.gen.load_state_dict(model_dict)
            print('load pretrained model from %s successly.' % root_)

        elif opt.backbone == 'ResNet12':
            # ckpt = torch.load('pretrained_model/mini_resnet_model_3pool_epoch161.pth.tar')

            def load_model(model, dir):
                model_dict = model.state_dict()
                print('loading model from :', dir)
                pretrained_dict = torch.load(dir)['params']
                if 'encoder' in list(pretrained_dict.keys())[0]:  # load from a parallel meta-trained model
                    if 'module' in list(pretrained_dict.keys())[0]:
                        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
                    else:
                        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
                else:
                    pretrained_dict = { k: v for k, v in
                                       pretrained_dict.items()}  # load from a pretrained model
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                # print(pretrained_dict)
                model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
                model.load_state_dict(model_dict)
            if opt.data_name == 'officehome':
                model_dict = model.method.gen.state_dict()
                ckpt = torch.load('./pretrained_model/officehome_resnet12_model_3pool_epoch90.pth.tar')
                pretrained_dict = ckpt['state_dict']
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.method.gen.load_state_dict(model_dict)
            if opt.data_name == 'visda':
                model_dict = model.method.gen.state_dict()
                if opt.combine_uda_fsl:
                    # e_dir = '/home/hsq/Projects/MCD_DA/classification/pretrained/MCD_%s_to_%s_resnet.pth.tar'%(opt.source_domain, opt.target_domain, )
                    e_dir = 'ADDA_pretrained/%s2%s-ADDA-target-encoder-final.pt'%(opt.source_domain, opt.target_domain, )
                    # e_dir = '/home/hsq/Projects/FSUDA_lds_spatial/ADDA_pretrained/%s2%s-ADDA-target-encoder-final.pt'% (opt.source_domain, opt.target_domain, )
                    ckpt = torch.load(e_dir)
                    n_ckpt = {}
                    for k in ckpt.keys():
                        n_k = k.replace('module.', '')
                        n_ckpt[n_k] = ckpt[k]
                    ckpt = {'state_dict': n_ckpt}
                    print('load from %s' % (e_dir))
                else:
                    ckpt = torch.load('./pretrained_models/visda_%s_resnet12_model_3pool.pth.tar' % opt.source_domain)
                pretrained_dict = ckpt['state_dict']
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.method.gen.load_state_dict(model_dict)
                if pretrained_dict.__len__() != 0:
                    print('load pretrained model success.')
                else:
                    print('load pretrained model failed.')
            elif opt.data_name == 'tiered-Imagenet':
                load_model(model.method.gen, 'pretrained_model/tieredImagenet_resnet12.pth')
            else:
                load_model(model.method.gen, 'pretrained_model/epoch-64.pth')
        else:
            if opt.data_name == 'visda':
                model_path = os.path.join('/home/hsq/Projects/pretrained_models', 'visda_%s_100epoch.pth.tar'%opt.source_domain)
            elif opt.data_name == 'mini-Imagenet':
                model_path = os.path.join('/home/hsq/Projects/pretrained_models', 'mini-Imagenet_%s_50epoch.pth.tar' % opt.source_domain)
            elif opt.data_name == 'officehome':
                model_path = os.path.join('/home/hsq/Projects/pretrained_models', 'officehome_%s_30epoch.pth.tar' % opt.source_domain)
            else:
                raise ValueError('Error dataset name.')
            model_dict = model.method.gen.state_dict()
            ckpt = torch.load(model_path)
            pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if len(pretrained_dict) == 0:
                raise ValueError('Error ckpt.')
            model_dict.update(pretrained_dict)
            model.method.gen.load_state_dict(model_dict)
            print('load pretrained model success.')

            # model.method.gen.load_state_dict(ckpt['state_dict'])

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
                                preprocess=model.method.preprocess, source_domain=opt.source_domain, target_domain=opt.target_domain, 
                                way_num=opt.way_num)
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


def train(train_loader, model, epoch_index, opt, meters_dict, inp_records, mode='train'):
    train_test_data, out_traintest_data_path = inp_records
    for episode_index, (query_images, query_targets, query_modal, support_images, support_targets, query_global_targets) in tqdm(enumerate(train_loader)):

        # Convert query and support images
        query_images = torch.squeeze(query_images.type(torch.FloatTensor))
        # input_var2 = [s[0].cuda() for s in support_images]
        support_images = support_images.squeeze(0)
        query_images = query_images.type(torch.FloatTensor)
        input_var2 = support_images.cuda()
        input_var1 = query_images.cuda()
        query_modal = query_modal.cuda()
        query_targets = query_targets.cuda()
        support_targets = support_targets.cuda()
        query_global_targets = query_global_targets.cuda()
        loss_acc = model(query_x=input_var1, support_x=input_var2, query_y=query_targets, support_y=support_targets, query_m=query_modal, global_label=query_global_targets)
        # print(model.method.imgtoclass.module.maps)

        # Measure accuracy and record loss
        out_line = accumulate_and_output_meters(loss_acc, meters_dict, epoch_index, episode_index, len(train_loader), mode)

        # ============== print the intermediate results ==============#
        if episode_index % opt.print_freq == 0 and episode_index != 0:
            print_and_record(out_line, opt.outf+'/out_line.txt')
            train_test_data['train'] = {k: meters_dict[k].histories for k in meters_dict}
            out_put_metadata(train_test_data, out_traintest_data_path)
    # return meters_dict['cls loss']

@torch.no_grad()
def validate(val_loader, model, epoch_index, best_prec1, opt, meters_dict, inp_records, mode='val'):
    # switch to evaluate mode
    model.eval()
    train_test_data, out_traintest_data_path = inp_records
    # if mode == 'Final test':
    #     test_acc = []
    for episode_index, (query_images, query_targets, query_modal, support_images, support_targets, query_global_targets) in tqdm(enumerate(val_loader)):

        # Convert query and support images
        query_images = torch.squeeze(query_images.type(torch.FloatTensor))
        support_images = support_images.squeeze(0)
        input_var1 = query_images.cuda()
        input_var2 = support_images.cuda()
        
        support_targets = support_targets.cuda()
        # Calculate the output
        query_modal = query_modal.cuda()
        query_targets = query_targets.cuda()
        loss_acc = model(query_x=input_var1, support_x=input_var2, query_y=query_targets, support_y=support_targets, query_m=query_modal, train=mode)
        # if mode == 'Final test':
        #     test_acc.append(round(loss_acc['target_acc'][0].item(), 2)/100)
        # Measure accuracy and record loss
        out_line = accumulate_and_output_meters(loss_acc, meters_dict, epoch_index, episode_index, len(val_loader), mode)


        # ============== print the intermediate results ==============#
        if episode_index % opt.print_freq == 0 and episode_index != 0:
            print_and_record(out_line, opt.outf+'/out_line.txt')
            if mode in ['test', 'val']:
                train_test_data[mode] = {k: meters_dict[k].histories for k in meters_dict}
                out_put_metadata(train_test_data, out_traintest_data_path)

    # if mode == 'Final test':
        # x_axis_data = range(1, len(test_acc)+1, 6)
        # plt.plot(x_axis_data, test_acc[::6], 'bo-', alpha=0.5, linewidth=1, label='target_acc')
        # plt.legend()
        # plt.title('600_test_tasks')
        # plt.xlabel('test_times')  # x_label
        # plt.ylabel('test_acc')  # y_label
        # plt.grid()
        # plt.savefig('/home/whr/pyProject/Pycharm/melda_clip/fig.png')
        # plt.show()
    print(' * Best prec1 {best_prec1:.3f}, now prec1 {now:.3f}'.format(best_prec1=best_prec1, now=meters_dict['target_acc'].epoch_avg(-1)))
    return meters_dict['target_acc'].epoch_avg(-1)  # , meters_dict['cls loss']


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

# ======================================== Settings of path ============================================

import nni
import collections
import logging
# ========================================== Model Config ===============================================
def main_(opt):
    print('loss weight is:', opt.loss_weight)
    if 'memory_load_path' in vars(opt):
        opt.memory_load_path = opt.memory_load_path.format(opt.source_domain)
    if opt.auto_tune:
        opt.outf = os.path.join(opt.outf, nni.get_trial_id())
    else:
        opt.outf = os.path.join(opt.outf, str(opt.exp_id))
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)
    print(opt.exp_id, opt.outf)
    json.dump(vars(opt), open(os.path.join(opt.outf, 'exp_args.json'), 'w'))
    
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # save the opt and results to a txt file
    print(opt)
    gpu_num = int(opt.ngpu)
    global best_prec1, epoch_index
    best_prec1 = 0
    best_test = 0
    
    meta_data = {
        'metric': opt.metric,
        'epochs': opt.epochs,
        'start_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        'end_time': '-1',
        'progress': 1,
        'bestacc': 0,
    }
    train_meters, test_meters, val_meters = {}, {}, {}
    train_test_data = {'train': {}, 'test': {}, 'val': {}}
    out_metadata_path = opt.outf+'/meta_data.json'
    out_traintest_data_path = opt.outf+'/train_test.json'
    
    print(out_metadata_path, out_traintest_data_path)
    out_put_metadata(meta_data, out_metadata_path)
    out_put_metadata(train_test_data, out_traintest_data_path)

    model = hyper_model.Trainer(opt=opt)

    model.method.imgtoclass.writer = writer

    # if opt.pretrained and opt.metric not in ('MCD', 'DWT', 'DANN'):
    #     if opt.backbone == 'ResNet12':
    #         # ckpt = torch.load('pretrained_model/mini_resnet_model_3pool_epoch161.pth.tar')

    #         def load_model(model, dir):
    #             model_dict = model.state_dict()
    #             print('loading model from :', dir)
    #             pretrained_dict = torch.load(dir)['model_sd']
    #             if 'encoder' in list(pretrained_dict.keys())[0]:  # load from a parallel meta-trained model
    #                 if 'module' in list(pretrained_dict.keys())[0]:
    #                     pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
    #                 else:
    #                     pretrained_dict = {k[8:]: v for k, v in pretrained_dict.items()}
    #             else:
    #                 pretrained_dict = { k: v for k, v in
    #                                    pretrained_dict.items()}  # load from a pretrained model
    #             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #             # print(pretrained_dict)
    #             model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
    #             model.load_state_dict(model_dict)
    #         if opt.rotate:
    #             root_ = '/home/hsq/Projects/rotate_pretrained'
    #             if opt.data_name == 'visda':
    #                 model_path = os.path.join(root_, 'visda_%s_to_%s_rotate_resnet_best.pth.tar' % (
    #                 opt.source_domain, opt.target_domain))
    #             elif opt.data_name == 'mini-Imagenet':
    #                 model_path = os.path.join(root_, 'mini-imagenet_ycm_%s_to_%s_conv64f_best.pth.tar' % (
    #                 opt.source_domain, opt.target_domain))
    #             elif opt.data_name == 'officehome':
    #                 model_path = os.path.join(root_, 'officehome_%s_to_%s_conv64f_best.pth.tar' % (
    #                 opt.source_domain, opt.target_domain))
    #             else:
    #                 raise ValueError('Error dataset name.')
    #             model_dict = model.method.gen.state_dict()
    #             ckpt = torch.load(model_path, map_location='cuda:0')
    #             pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    #             a_dict = {}
    #             for k in pretrained_dict:
    #                 new_k = k.replace('module.', '')
    #                 a_dict[new_k] = pretrained_dict[k]
    #                 # pretrained_dict.pop(k)
    #             pretrained_dict = a_dict
    #             # print(pretrained_dict.keys())
    #             # exit()
    #             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #             if len(pretrained_dict) == 0:
    #                 raise ValueError('Error ckpt.')
    #             model_dict.update(pretrained_dict)
    #             model.method.gen.load_state_dict(model_dict)
    #             print('load pretrained rotate model from %s successly.' % root_)
    #         elif opt.data_name == 'officehome':
    #             model_dict = model.method.gen.state_dict()
    #             ckpt = torch.load('./pretrained_model/officehome_resnet12_model_3pool_epoch90.pth.tar')
    #             pretrained_dict = ckpt['state_dict']
    #             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #             model_dict.update(pretrained_dict)
    #             model.method.gen.load_state_dict(model_dict)
    #         elif opt.data_name == 'visda':
    #             model_dict = model.method.gen.state_dict()
    #             if opt.combine_uda_fsl:
    #                 # e_dir = '/home/hsq/Projects/MCD_DA/classification/pretrained/MCD_%s_to_%s_resnet.pth.tar'%(opt.source_domain, opt.target_domain, )
    #                 e_dir = 'ADDA_pretrained/%s2%s-ADDA-target-encoder-final.pt'%(opt.source_domain, opt.target_domain, )
    #                 # e_dir = '/home/hsq/Projects/FSUDA_lds_spatial/ADDA_pretrained/%s2%s-ADDA-target-encoder-final.pt'% (opt.source_domain, opt.target_domain, )
    #                 ckpt = torch.load(e_dir)
    #                 n_ckpt = {}
    #                 for k in ckpt.keys():
    #                     n_k = k.replace('module.', '')
    #                     n_ckpt[n_k] = ckpt[k]
    #                 ckpt = {'state_dict': n_ckpt}
    #                 print('load from %s' % (e_dir))                    
    #             else:
    #                 ckpt = torch.load('/home/hsq/Projects/pretrained_models/visda_resnet_%s_100epoch.pth.tar' % opt.source_domain)
    #             pretrained_dict = ckpt
    #             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #             model_dict.update(pretrained_dict)
    #             model.method.gen.load_state_dict(model_dict)
    #             if pretrained_dict.__len__() != 0:
    #                 print('load pretrained model success.')
    #             else:
    #                 print('load pretrained model failed.')
    #         elif opt.data_name == 'tiered-Imagenet':
    #             load_model(model.method.gen, '/home/hsq/project/pretrained_models/tieredImagenet_resnet12.pth')
    #         else:
    #             load_model(model.method.gen, '/home/hsq/Projects/pretrained_models/mini_resnet_%s.pth'%opt.source_domain)
    #     elif opt.backbone == 'Conv64F':
    #         if opt.data_name == 'visda':
    #             model_path = os.path.join('/home/hsq/Projects/pretrained_models', 'visda_%s_100epoch.pth.tar'%opt.source_domain)
    #         elif opt.data_name == 'mini-Imagenet':
    #             model_path = os.path.join('/home/hsq/Projects/pretrained_models', 'mini-Imagenet_%s_50epoch.pth.tar' % opt.source_domain)
    #         elif opt.data_name == 'officehome':
    #             model_path = os.path.join('/home/hsq/Projects/pretrained_models', 'officehome_%s_30epoch.pth.tar' % opt.source_domain)
    #         model_dict = model.method.gen.state_dict()
    #         ckpt = torch.load(model_path)
    #         pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    #         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #         if len(pretrained_dict) == 0:
    #             raise ValueError('Error ckpt.')
    #         model_dict.update(pretrained_dict)
    #         model.method.gen.load_state_dict(model_dict)
    #         print('load pretrained model success.')
    #     else:
    #         model_path = os.path.join('/home/hsq/Projects/pretrained_models',
    #                                   'mini_conv512F_%s.pth' % opt.source_domain)

    #         model_dict = model.method.gen.state_dict()
    #         ckpt = torch.load(model_path)
    #         pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    #         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #         if len(pretrained_dict) == 0:
    #             raise ValueError('Error ckpt.')
    #         model_dict.update(pretrained_dict)
    #         model.method.gen.load_state_dict(model_dict)
    #         print('load pretrained model success.')

            # model.method.gen.load_state_dict(ckpt['state_dict'])
    # train_cls_losses = []
    # val_cls_losses = []
    # test_cls_losses = []
    model.eval()

    # ======================================== Training phase ===============================================
    print('\n............Start training............\n')
    test_acc = []
    for epoch_item in range(opt.pretrain, opt.epochs):
        if 'Baseline' in opt.metric or opt.metric in ('MCD', 'DWT', 'DANN', 'MCDR2D2', 'ADDA') or opt.finetune:
            break
        print('===================================== Epoch %d =====================================' % epoch_item)
        model.adjust_learning_rate(epoch_item)

        # ======================================= Folder of Datasets =======================================
        data_dir = opt.dataset_dir+'/'+opt.data_name
        trainset = DataGenerator(mode='train', datasource=opt.data_name,  data_dir=data_dir, imagesize=opt.imageSize,
            episode=opt.episode_train_num, support_num=opt.shot_num, query_num=opt.query_num, split_PATH=opt.split_PATH,
                                 preprocess=model.method.preprocess, source_domain=opt.source_domain, 
                                 target_domain=opt.target_domain, way_num=opt.way_num, random_target=opt.random_target)

        valset = DataGenerator(mode='val', datasource=opt.data_name, data_dir=data_dir, imagesize=opt.imageSize,
                               episode=opt.episode_val_num, support_num=opt.shot_num, query_num=opt.query_num, split_PATH=opt.split_PATH,
                               preprocess=model.method.preprocess, source_domain=opt.source_domain, 
                               target_domain=opt.target_domain, way_num=opt.way_num)

        testset = DataGenerator(mode='test', datasource=opt.data_name, data_dir=data_dir, imagesize=opt.imageSize,
                                episode=opt.episode_test_num, support_num=opt.shot_num, query_num=opt.query_num, split_PATH=opt.split_PATH,
                                preprocess=model.method.preprocess, source_domain=opt.source_domain, 
                                target_domain=opt.target_domain, way_num=opt.way_num)

        # print('Trainset: %d' % len(trainset))
        # print('Valset: %d' % len(valset))
        print('Testset: %d' % len(testset))

        # ========================================== Load Datasets =========================================
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=opt.episodeSize, shuffle=False,
            num_workers=int(opt.workers), drop_last=False, pin_memory=False
        )
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=opt.testepisodeSize, shuffle=False,
            num_workers=int(opt.workers), drop_last=False, pin_memory=False
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=opt.testepisodeSize, shuffle=False,
            num_workers=int(opt.workers), drop_last=False, pin_memory=False
        )

        # ============================================ Training ===========================================
        # Fix the parameters of Batch Normalization after 10000 episodes (1 epoch)
        if opt.metric in ('DeepEMD', 'MetaBaseline', 'DSN') or (opt.on_bn and epoch_item < opt.bn_epoch):
            model.train()
        else:
            model.eval()

        # Train for 10000 episodes in each epoch
        if len(opt.gpu) > 1 and opt.multigpu:
            gpuid = range(opt.gpu.split(',').__len__())
            model.method.gen = nn.DataParallel(model.method.gen, [int(g) for g in gpuid],)
            model.method.imgtoclass = nn.DataParallel(model.method.imgtoclass, [int(g) for g in gpuid],)
            model.method.discri = nn.DataParallel(model.method.discri, [int(g) for g in gpuid],)
                # model.method.cluster = nn.DataParallel(model.method.cluster, [int(g) for g in gpuid])
        #     model.eval()
        train(train_loader, model, epoch_item, opt, train_meters, (train_test_data, out_traintest_data_path))
        # train_cls_losses.append(train_cls_loss.avg)
        print('dataset %s, with 5-way, %d-shot, %s metric.'%(opt.data_name, opt.shot_num, opt.metric))
        model.eval()
        # =========================================== Evaluation ==========================================
        print('============ Validation on the val set ============')
        prec1= validate(val_loader, model, epoch_item, best_prec1, opt, val_meters, (train_test_data, out_traintest_data_path), mode='val')
        # val_cls_losses.append(val_cls_loss.avg)
        # record the best prec@1 and save checkpoint

        # Testing Prase
        print('============ Testing on the test set ============')
        prec_test = validate(test_loader, model, epoch_item, best_prec1, opt, test_meters, (train_test_data, out_traintest_data_path), mode='test')

        # test_cls_losses.append(test_cls_loss.avg)
        if opt.auto_tune: nni.report_intermediate_result(prec_test)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        best_test = max(prec_test, best_test)

        # save the checkpoint
        if not os.path.exists(os.path.join(opt.outf, 'saved_models')):
            os.mkdir(os.path.join(opt.outf, 'saved_models'))
        
        save_params = collections.OrderedDict()
        for key in model.method.state_dict():
            if 'model' not in key:
                save_params[key] = model.method.state_dict()[key]
            if 'model.visual.prompt_embeddings' in key:
                save_params[key] = model.method.state_dict()[key]
            if 'model.visual.prompt_net' in key:
                save_params[key] = model.method.state_dict()[key]
        if is_best:
            save_checkpoint(
                {
                    'epoch_index': epoch_item,
                    'arch': opt.basemodel,
                    'state_dict': save_params,
                    'best_prec1': best_prec1,
                    'optimizer': model.g_optimizer.state_dict(),
                }, os.path.join(opt.outf, 'saved_models', 'model_best.pth.tar'))
        meta_data['progress'] = int(100 * (epoch_item + 1) / meta_data['epochs'])
        meta_data['bestacc'] = best_test
        # train_test_data['train'] = {k:train_meters[k].histories for k in train_meters}
        # train_test_data['test'] = {k:test_meters[k].histories for k in test_meters}
        # train_test_data['val'] = {k:val_meters[k].histories for k in val_meters}
        # out_put_metadata(train_test_data, out_traintest_data_path)
        out_put_metadata(meta_data, out_metadata_path)
    print('............Training is end............')
    # ============================================ Training End ==============================================================
    # =======================plt==================== #
    # print(train_cls_losses)
    # print(test_cls_losses)
    # print(val_cls_losses)
    print('............Now meta testing............')
    starttime = time.time()
    res = test(opt)
    print(time.time()-starttime)
    res['val_acc'] = best_prec1
    torch.cuda.empty_cache()
    meta_data['final_test_acc'] = res
    out_put_metadata(meta_data, out_metadata_path)
    if opt.auto_tune: nni.report_final_result(res['aver_accuracy'])
    return res


import traceback
import shutil
def main(opt):
    try:
        main_(opt)
    except Exception as e:
        print(traceback.print_exc())
        shutil.rmtree(opt.outf, ignore_errors=True)
        father_dir = os.path.join('/', *opt.outf.split('/')[:-1])
        if not any([os.path.isdir(os.path.join(father_dir, _)) for _ in os.listdir(father_dir)]):
            shutil.rmtree(father_dir, ignore_errors=True)
        raise Exception