from model.discriminator import Discriminator
from model.backbone import ResNet12, Conv64F, Conv512F
from model.classifiers import *
from torch import nn, mm
from model.melda import *
from model.IMSE_based import *
import torch
import torch.optim as optim
from torch.autograd import Variable
from utils import cal_cov_loss, init_weights
from model.clip_model import *
from model.baseline import *
import clip

import os

# device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU

# os.environ["CUDA_VISIBLE_DEVICES"] ='4'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def to_var(x, requires_grad=True):
    x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class FSUDA_METHOD(nn.Module):
    def __init__(self, opt):
        super(FSUDA_METHOD, self).__init__()
        # self.discri_name = opt.discri
        self.dper = None
        self.opt = opt
        self.iter = opt.iter
        self.backbone = opt.backbone
        self.construct()

    def construct(self):
        backbone = self.opt.backbone
        if backbone == 'CLIP':
            self.model_name = self.opt.model_name
            self.device = 'cuda'
            self.model, self.preprocess = clip.load(self.model_name, self.device)
        elif backbone == 'ResNet12':
            # if self.opt.ld_num == 25:
            #     last_layer = [False, False, False, False]
            # elif self.opt.ld_num == 441:
            #     last_layer = [False, False, True, True]
            # else:
            #     last_layer = [False, False, False, True]
            # self.gen = ResNet(last_layer=last_layer,
            #                   drop_rate=self.opt.drop_rate)
            # self.preprocess = 0
            self.gen = ResNet12(opt=self.opt,
                                channels=[64, 160, 320, 640]).cuda()
        else:
            raise ValueError(
                "Wrong backbone, no such backbone {}, please check your backbone configuration!".format(backbone))

        # self.discri = MELDA(init_trans_scale=60, epsilon=1e-7, discriminator=True)
        # self.discri = ThetaP(discriminator=True)
        self.clip_adaptor = ClipAdaptor(self.opt)
        nm_space = vars(self.opt)
        kwargs = eval(self.opt.metric).__init__.__code__.co_varnames
        keys_ = {k: nm_space[k] for k in nm_space if k in kwargs}
        self.imgtoclass = eval(self.opt.metric)(**keys_).cuda()
        self.EMD_fun = EMD_dis(opt=self.opt)
        # init_weights(self.gen, init_type='kaiming')
        init_weights(self.clip_adaptor, init_type='kaiming')
        # init_weights(self.discri, init_type='kaiming')
        init_weights(self.imgtoclass, init_type='kaiming')
        print('===' * 10)
        # print('backbone is:%s' % self.gen.__class__.__name__)
        print('adaptor is:%s' % self.clip_adaptor.__class__.__name__)
        # print('discriminator is:%s' % self.discri.__class__.__name__)
        print('metric method is:%s' % self.imgtoclass.__class__.__name__)

    def forward(self, input):
        query, support, support_y, query_y, train = input
        # obtain deep-features
        q, S = self.forward_gen(query, support)

        # query_num, h, w, C = q.size()
        query_num, C = q.size()
        # S = S.view(n_domain, way, n_shot, h_w, C)  # n_support, h, w, C
        # q /= q.norm(dim=-1, keepdim=True)
        # S /= S.norm(dim=-1, keepdim=True)
        # txt_fea /= txt_fea.norm(dim=-1, keepdim=True)
        # query_sim = q @ txt_fea.t()
        # support_sim = S @ txt_fea.t()
        # if 'MELDA' == self.opt.metric:
        #     q = q.view(query_num, -1)
        #     S = S.view(n_domain, way, n_shot, -1)
        # get the domain prediction from the discriminator
        if train == 'train':
            class_predict, loss_regular, loss_lda = self.imgtoclass(q, S, train=True)
        else:
            class_predict, loss_regular, loss_lda = self.imgtoclass(q, S, train=False)
        # model_predict = self.discri(q, S)

        # get the domain prediction from the discriminator
        # model_predict = self.discri(query_sim, support_sim)
        # class_predict = self.imgtoclass(query_sim, support_sim)
        loss_dis, count_yes, count_no = self.EMD_fun.cal_dis(S[0], q, query_y, class_predict, train=True)
        return class_predict, loss_regular, loss_lda, count_yes, count_no#, model_predict  # ["logits"]#, model_predict

    def forward_gen(self, query, support, update=False):
        # return features
        query_num = query.size(0)
        if self.imgtoclass.__class__.__name__ == 'ThetaP' or 'MELDA':
            way_num, shot_num, C, h, w = len(support[0]), *support[0][0].size()
            support = support.view(-1, C, h, w)
            datas = torch.cat([query, support], 0)
            # datas = self.gen(datas).float()#view(datas.shape[0], -1).float() # [160, 3, 224, 224] → [160, 25, 640]
            datas = self.model.encode_image(datas).float()
            datas = self.clip_adaptor(datas)['features']  # [160, 512] → [160, 128]
            # flag = torch.isnan(datas).any()
            # if flag == True:
            #     datas = torch.rand((150 + shot_num * way_num * 2, 128)).float().cuda()
            #     self.count_nan += 1
            #     print(self.count_nan)
            # datas = datas.view(datas.shape[0], -1)
            # text_features = []
            # for text in label_text:
            #     text = clip.tokenize(text).to(datas.device)
            #     text_features.append(self.model.encode_text(text))
            # text_features = torch.cat(text_features, 0).float()
            q, S = datas[:query_num], datas[query_num:].view(2, way_num, shot_num, -1)
            # q, S = datas[:query_num], datas[query_num:].view(2, way_num, shot_num, datas.shape[1], datas.shape[2], datas.shape[3]) # (150, 128) ,(2,5,1,128)
            # q, S= datas[:query_num].view(-1, datas.shape[-1]), datas[query_num:].view(2, way_num, shot_num*datas.shape[1], datas.shape[-1])
        elif self.imgtoclass.__class__.__name__ == 'BaseLinePlusPlus':
            datas = torch.cat([query, support], 0)
            datas = self.model.encode_image(datas).float()
            datas = self.clip_adaptor(datas)['features']
            q, S = datas[:query_num], datas[query_num:]
        else:
            q, S = datas[:query_num], datas[query_num:]
        return q, S

    # def compute_labeled(self, features):
    #     num_classes = 5
    #     samples_per_class = 15  # 75 / 5 = 15
    #     class_indices = [range(i * samples_per_class, (i + 1) * samples_per_class) for i in range(num_classes)]
    #     # 1. 计算每个类的中心（均值），并归一化（余弦距离对方向敏感，归一化更合理）
    #     class_centers = []
    #     for indices in class_indices:
    #         class_samples = features[list(indices)]  # 类内样本 [15, 128]
    #         center = torch.mean(class_samples, dim=0)  # 类中心 [128]
    #         center_norm = nn.functional.normalize(center, p=2, dim=0)  # 归一化到单位向量
    #         class_centers.append(center_norm)
    #     class_centers = torch.stack(class_centers)  # [5, 128]
    #
    #     # 2. 计算类内距离（类内样本与类中心的余弦距离之和）
    #     intra_dist = 0.0
    #     for i, indices in enumerate(class_indices):
    #         class_samples = features[list(indices)]  # [15, 128]
    #         samples_norm = nn.functional.normalize(class_samples, p=2, dim=1)  # 样本归一化
    #         # 计算每个样本与类中心的余弦相似度，再转为距离（1 - 相似度）
    #         cos_sim = torch.sum(samples_norm * class_centers[i], dim=1)  # [15]，内积即余弦相似度（已归一化）
    #         intra_dist += torch.sum(1 - cos_sim)  # 累加类内余弦距离
    #
    #     # 3. 计算类间距离（所有类中心之间的余弦距离之和）
    #     inter_dist = 0.0
    #     num_centers = class_centers.shape[0]
    #     for i in range(num_centers):
    #         for j in range(i + 1, num_centers):  # 只算i<j的对，避免重复
    #             cos_sim = torch.sum(class_centers[i] * class_centers[j])  # 类中心余弦相似度
    #             inter_dist += (1 - cos_sim)  # 累加类间余弦距离
    #
    #     # 4. 构造损失：类内距离 / (类间距离 + 微小值)，最小化该值
    #     eps = 1e-8
    #     loss = intra_dist / (inter_dist + eps)
    #     return loss
    #
    # def compute_unlabel(self, x):
    #     batch_size = x.size(0)
    #     total_intra_loss = 0.0  # 类内损失
    #     total_inter_loss = 0.0  # 类间损失
    #     count_intra = 0  # 类内样本对计数
    #     count_inter = 0  # 类间样本对计数
    #
    #     # 遍历所有样本对
    #     for i in range(batch_size):
    #         for j in range(i + 1, batch_size):
    #             # 计算余弦相似度
    #             sim = self.cosine_sim(x[i].unsqueeze(0), x[j].unsqueeze(0))[0]
    #             cosine_dist = 1 - sim  # 余弦距离（1-相似度）
    #
    #             # 根据阈值判断是否为同类
    #             if sim > self.threshold:
    #                 # 类内：希望距离越小越好（损失为距离本身）
    #                 total_intra_loss += cosine_dist
    #                 count_intra += 1
    #             else:
    #                 # 类间：希望距离越大越好（损失为负距离）
    #                 total_inter_loss -= cosine_dist
    #                 count_inter += 1
    #
    #     # 防止除零（如果没有同类或异类样本对）
    #     if count_intra == 0:
    #         intra_loss = torch.tensor(0.0, device=x.device)
    #     else:
    #         intra_loss = total_intra_loss / count_intra
    #
    #     if count_inter == 0:
    #         inter_loss = torch.tensor(0.0, device=x.device)
    #     else:
    #         inter_loss = total_inter_loss / count_inter
    #
    #     # 总损失：类内损失 + 类间损失（两者都希望最小化）
    #     total_loss = intra_loss + inter_loss
    #     return total_loss

class Trainer(nn.Module):
    def __init__(self, opt):
        super(Trainer, self).__init__()
        self.modallossfn = nn.CrossEntropyLoss()
        self.classlossfn = nn.CrossEntropyLoss()
        self.discri_name = opt.discri
        self.lr = opt.lr
        self.dlr = opt.lr * 1000
        self.ilr = 0.01
        self.globallr = 0.01
        self.method = None
        self.loss_weight = opt.loss_weight
        self.deacy_rate = opt.decay_rate
        self.decay_interim = opt.interim
        self.metric = opt.metric
        self.ld_num = opt.ld_num
        self.opt = opt
        if self.opt.metric == 'DeepEMD' or self.opt.metric == 'MetaBaseline':
            self.opt.DA = False
        if self.method == None:
            self.method = FSUDA_METHOD(opt)
            self.method.cuda()
        param_list = [{'params': self.method.clip_adaptor.parameters(), 'weight_decay': 1e-4}]
        aa = self.method.model.visual.parameters()
        for i, param in enumerate(aa):
            if i == 3:
                param.requires_grad = True
                print(param.shape)
            else:
                param.requires_grad = False
        self.method.model.visual.prompt_net.requires_grad = True
        if self.method.imgtoclass.parameters() != None:
            print('optimize the imagetoclass.')
            param_list.append({'params': self.method.imgtoclass.parameters(), 'lr': self.ilr, 'weight_decay': 1e-4})
            param_list.append(
                {'params': filter(lambda p: p.requires_grad, self.method.model.visual.parameters()), 'lr': self.dlr,
                 'weight_decay': 1e-4})
        if self.method.dper != None:
            print('optimize the dper.')
            param_list.append({'params': self.method.dper.parameters(), 'lr': self.ilr})
        if self.opt.SGD:
            self.g_optimizer = optim.SGD(param_list, lr=self.lr, momentum=0.9, nesterov=True)
        else:
            self.g_optimizer = optim.Adam(param_list, lr=self.lr, betas=(opt.beta1, 0.9), weight_decay=1e-4)
        # self.d_optimizer = optim.Adam(self.method.discri.parameters(), lr=self.dlr, betas=(opt.beta1, 0.9))
        print('===' * 10)
        print('adv weight is:%.4f' % self.loss_weight)
        print('===' * 10)

    def adjust_learning_rate(self, epoch_num):
        """Sets the learning rate to the initial LR decayed by 0.05 every 10 epochs"""
        lr = self.lr * (self.deacy_rate ** (epoch_num // self.decay_interim))
        print("learning rate is: %f" % lr)
        optimizer = self.g_optimizer
        for i, param_group in enumerate(optimizer.param_groups):
            if i == len(optimizer.param_groups) - 1:
                param_group['lr'] = self.ilr * (self.deacy_rate ** (epoch_num // (self.decay_interim)))
            else:
                param_group['lr'] = self.lr * (self.deacy_rate ** (epoch_num // self.decay_interim))
        # optimizer = self.d_optimizer
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = self.dlr * (self.deacy_rate ** (epoch_num // self.decay_interim))

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precaccuracy(output, target, topk=(1,3))ision@k for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def exec_train(self, inp, train):
        query_x, support_x, query_y, support_y, query_m, query_global_label = inp
        # process the label data
        _, n_way, n_shot, _, _, _ = support_x.size()
        query_y = query_y.t().squeeze()
        query_m = query_m.t().squeeze()
        if self.discri_name == 'LD':
            # 'LD' uses all local features, so reproduce the domain label here.
            query_m = query_m.repeat([self.ld_num, 1]).t().reshape([1, -1]).squeeze()
        class_predic, loss_regular, loss_lda, count_yes, count_no = self.method((query_x, support_x, support_y, query_y, train))  # , model_predic
        loss_regular.requires_grad_(True)
        loss_lda.requires_grad_(True)
        if train == 'train':
            # compute the top-1 accuracy.
            class_predic = class_predic.reshape([2, -1, self.opt.way_num])
            query_y = query_y.reshape([2, -1])
            class_acc = self.accuracy(class_predic[0], query_y[0], topk=(1,))
            classify_acc = [class_acc, class_acc]
            acc_specific = 0
        else:
            class_predic = torch.reshape(class_predic, [2, -1, self.opt.way_num])
            query_y = torch.reshape(query_y, [2, -1])
            classify_acc = [self.accuracy(class_predic[0], query_y[0], topk=(1,)),
                            self.accuracy(class_predic[1], query_y[1], topk=(1,))]
            acc_specific = {'pred': [class_predic[0].argmax(-1), class_predic[1].argmax(-1)],
                            'gt': [query_y[0], query_y[1]]}
        # this label used for train the generator in adversarial training strategy.
        confusion_model_label = torch.ones_like(query_m) - query_m
        classify_loss = self.classlossfn(class_predic[0] * self.opt.class_temperature, query_y[0])
        pseudo_acc = round(count_yes / (count_yes + count_no + 1e-6) * 100, 2)
        if pseudo_acc == 0.0:
            pseudo_acc = 100.0
        pseudo_acc = torch.tensor(pseudo_acc).cuda() + 1
        pseudo_count = count_yes + count_no
        pseudo_count = torch.tensor(pseudo_count).cuda()
        if train == 'train' and self.opt.DA:
            # Training phase for the FSUDA.
            g_loss = self.classlossfn(model_predic * self.opt.discri_temperature, confusion_model_label)
            # d_loss = self.modallossfn(model_predic, query_m)
            cgloss = self.loss_weight * classify_loss + (1.0 - self.loss_weight) * g_loss
            # cgloss = self.loss_weight * classify_loss
            # if self.opt.metric != "MELDA":
            #     self.d_optimizer.zero_grad()
            #     d_loss.backward(retain_graph=True)
            self.g_optimizer.zero_grad()
            cgloss.backward()
            grads = torch.cat([_.view(-1) for x in self.g_optimizer.param_groups[0]['params'] for _ in x.grad], -1)
            # avoid the NaN during the updating.
            if True not in torch.isnan(grads):
                # self.d_optimizer.step()
                self.g_optimizer.step()
            # discriminator prediction accuracies.
            modal_acc = self.accuracy(model_predic, query_m, topk=(1,))
        elif self.opt.DA:
            # Testing phase for the FSUDA.
            g_loss = self.classlossfn(model_predic, confusion_model_label)
            d_loss = self.modallossfn(model_predic, query_m)
            modal_acc = self.accuracy(model_predic, query_m, topk=(1,))
        elif train == 'train':
            # Training phase for the FSL.
            cgloss = classify_loss + self.opt.loss_weight * loss_regular + self.opt.loss_lda * loss_lda
            # Computing the L1 term.
            reg_l1 = 0
            if self.method.imgtoclass.parameters():
                for params in self.method.imgtoclass.parameters():
                    reg_l1 += params.abs().sum()
            self.g_optimizer.zero_grad()
            (reg_l1 * 1e-5 + cgloss).backward()
            torch.nn.utils.clip_grad_value_(self.method.clip_adaptor.parameters(), 5)
            self.g_optimizer.step()
            g_loss = d_loss = modal_acc = cgloss.unsqueeze(0)
        else:
            cgloss = classify_loss
            g_loss = d_loss = modal_acc = cgloss.unsqueeze(0)

        return {'class_loss': classify_loss, 'loss_regular': loss_regular, 'loss_lda': loss_lda, 'source_acc': classify_acc[0][0], 'target_acc': pseudo_acc,
                'disc_acc': modal_acc[0],  'pseudo_count': pseudo_count}

    def forward(self, query_x, support_x, query_y, support_y, query_m, train='train', global_label=None):
        query_y = query_y.cuda()
        inp = (query_x, support_x, query_y, support_y, query_m, global_label)
        loss_acc = self.exec_train(inp, train)
        return loss_acc


class ThetaPLoss(nn.Module):
    def __init__(self):
        super(ThetaPLoss, self).__init__()

    def forward(self, source, target):
        # normalize

        source_tmp = source.mean(0)
        target_tmp = target.mean(0)
        thetap_mean_loss = ((source_tmp - target_tmp) ** 2).sum()

        n_s = source.size(0)
        n_t = target.size(0)
        source_tp = source.reshape([n_s, -1])
        target_tp = target.reshape([n_t, -1])
        source_mu = source_tp.mean(0, True)
        target_mu = target_tp.mean(0, True)
        source_tmp = source_tp - source_mu
        target_tmp = target_tp - target_mu
        C_s = mm(source_tmp.permute(1, 0), source_tmp) / (n_s - 1)
        C_t = mm(target_tmp.permute(1, 0), target_tmp) / (n_t - 1)
        thetap_loss = ((C_s - C_t) ** 2).sum()

        return thetap_loss, thetap_mean_loss


class ThetaPLossLD(ThetaPLoss):
    def __init__(self):
        super(ThetaPLossLD, self).__init__()

    def forward(self, source, target):
        # normalize
        source_tmp = source.mean(0)
        target_tmp = target.mean(0)
        thetap_loss = ((source_tmp - target_tmp) ** 2).sum()
        return thetap_loss


class Trainer_new(Trainer):
    def __init__(self, opt):
        super(Trainer_new, self).__init__(opt)

    def exec_train(self, inp, train):
        losses_grads = []
        query_x, support_x, query_y, support_y, query_m, query_global_label = inp
        # process the label data
        query_y = query_y.t().squeeze()
        query_m = query_m.t().squeeze()
        if self.discri_name == 'LD':
            # 'LD' uses all local features, so reproduce the domain label here.
            query_m = query_m.repeat([self.ld_num, 1]).t().reshape([1, -1]).squeeze()
        class_predic = self.method((query_x, support_x, support_y, train)) # , model_predic

        class_predic = class_predic.view(2, -1, class_predic.size(-1))
        query_y = query_y.reshape([2, -1])
        if train == 'train':
            # compute the top-1 accuracy.
            class_acc = self.accuracy(class_predic[0], query_y[0], topk=(1,))
            classify_acc = [class_acc, class_acc]
            acc_specific = 0
        else:
            classify_acc = [self.accuracy(class_predic[0], query_y[0], topk=(1,)),
                            self.accuracy(class_predic[1], query_y[1], topk=(1,))]
            acc_specific = {'pred': [class_predic[0].argmax(-1), class_predic[1].argmax(-1)],
                            'gt': [query_y[0], query_y[1]]}
        # this label used for train the generator in adversarial training strategy.
        confusion_model_label = torch.ones_like(query_m) - query_m
        # target_entopy_loss = self.targetentropy(class_predic[1])
        # theta
        # print(model_predic)
        classify_loss = self.classlossfn(class_predic[0] * self.opt.class_temperature, query_y[0])
        if train == 'train' and self.opt.DA:
            # Training phase for the FSUDA.
            # domain adversary loss
            g_loss = self.classlossfn(model_predic * self.opt.discri_temperature, confusion_model_label)
            # cgloss = self.loss_weight * classify_loss + self.opt.adv_loss * g_loss
            # classify gradients
            # print(classify_loss)
            # print(g_loss*self.loss_weight)
            # exit()
            self.g_optimizer.zero_grad()
            classify_loss.backward(retain_graph=True)
            # losses_grads.append(self.method.clip_adaptor.get_grads())
            # # domain adversary gradients
            # self.g_optimizer.zero_grad()
            g_loss.backward()
            # losses_grads.append(self.method.clip_adaptor.get_grads())
            # classify_loss.backward()
            # self.g_optimizer.zero_grad()
            # avoid the NaN during the updating.
            # new_grads = self.method.clip_adaptor.proj_grad(losses_grads)
            # self.method.clip_adaptor.set_grads(new_grads)
            # new_grads = self.method.clip_adaptor.get_grads()
            # if True not in torch.isnan(new_grads):
            self.g_optimizer.step()
            # else:
            #     print("skip the nan grad.")

            modal_acc = self.accuracy(model_predic, query_m, topk=(1,))
        elif train == 'train':
            # Training phase for the FSL.
            cgloss = classify_loss

            # Computing the L1 term.
            reg_l1 = 0
            if self.method.imgtoclass.parameters():
                for params in self.method.imgtoclass.parameters():
                    reg_l1 += params.abs().sum()
            self.g_optimizer.zero_grad()
            torch.nn.utils.clip_grad_value_(self.method.clip_adaptor.parameters(), 5)
            self.g_optimizer.step()
            g_loss = d_loss = modal_acc = cgloss.unsqueeze(0)
        else:
            cgloss = classify_loss
            g_loss = d_loss = modal_acc = cgloss.unsqueeze(0)
        return {'class_loss': classify_loss, 'g_loss': g_loss, 'source_acc': classify_acc[0][0],
                'target_acc': classify_acc[1][0], 'disacc': modal_acc[0]}