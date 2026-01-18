import torch
from torch import nn, mm
from torch.nn.parameter import Parameter
from model.BaseModel import BaseModelFSL
from torch.autograd import Variable
import numpy as np
from model.similarity_encoder import *
import os
import torch.nn.functional as F
# device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


class MELDAFC(nn.Module):
    def __init__(self, discriminator=False):
        super(MELDAFC, self).__init__()
        self.discriminator = discriminator

    def __getcls__(self, supports):
        if self.discriminator:
            n_domain, n_way, n_shot, h, w, c = supports.size()
            supports = supports.contiguous().view(n_domain, n_way * n_shot, h, w, c)
        else:
            # n_way, n_shot, h, w, c = supports.size()
            supports = supports[0].contiguous()  # .view(n_way, n_shot, h, w, c)
            # support_tmp = support[0].contiguous().view(n_way * n_shot, output_dim)
        n_way, n_shot, h, w, c = supports.size()
        # supports = supports.mean(dim=[-1, -2])
        supports = supports.view(n_way, n_shot, -1)
        supports = supports.detach()
        classifier = nn.Sequential(nn.Linear(h * w * c, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, n_way),
                                   nn.ReLU()
                                   ).to(supports.device)
        labels = torch.arange(start=0, end=n_way).view(-1, 1).repeat(1, n_shot).reshape(-1).to(supports.device)
        CEloss = nn.CrossEntropyLoss().to(supports.device)
        # optim = torch.optim.SGD([{'params': classifier.parameters()}], lr=0.1, momentum=0.9)
        optim = torch.optim.Adam([{'params': classifier.parameters()}], lr=0.01, betas=(0.5, 0.9), weight_decay=1e-4)
        with torch.enable_grad():
            for _ in range(100):
                pred = classifier(supports.view(-1, h * w * c))
                loss = CEloss(pred, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
        return classifier

    def forward(self, query, support):
        # support = torch.stack(support, 0)
        classifier = self.__getcls__(support)
        # query = query.mean([-1, -2])
        query = query.contiguous().view(query.size(0), -1)
        logits = classifier(query)
        return logits


def t_(x):
    return torch.transpose(x, 0, 1)


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def make_float_label(n_way, n_samples, flag=False):
    if flag is False:
        label = torch.FloatTensor(n_way * n_samples, n_way).zero_()
        for i in range(n_way):
            label[n_samples * i:n_samples * (i + 1), i] = 1
        return to_variable(label)
    else:
        label = torch.FloatTensor(n_way * n_samples, n_way).zero_()
        for i in range(n_way):
            k = abs(i - 1)
            label[n_samples * i:n_samples * (i + 1), k] = 1
        return to_variable(label)


def make_target_float_label(target_center):
    center_num = len(target_center)
    target_num = 0
    for c in target_center:
        target_num = target_num + c.size(0)
    label = torch.FloatTensor(target_num, 5).zero_()
    index = 0
    for i in range(center_num):
        label[index:index + target_center[i].size(0), i] = 1
        index = index + target_center[i].size(0)
    return to_variable(label)


def make_long_label(n_way, n_samples, flag=False):
    if flag is False:
        label = torch.LongTensor(n_way * n_samples).zero_()
        for i in range(n_way * n_samples):
            label[i] = i // n_samples
        return to_variable(label)
    else:
        label = torch.LongTensor(n_way * n_samples).zero_()
        for i in range(n_way * n_samples):
            label[i] = abs(i // n_samples - 1)
        return to_variable(label)


class LambdaLayer(nn.Module):
    def __init__(self, learn_lambda=False, init_lambda=1, base=1):
        super().__init__()
        self.l = torch.FloatTensor([init_lambda]).cuda()
        self.base = base
        if learn_lambda:
            self.l = nn.Parameter(self.l)
        else:
            self.l = Variable(self.l)

    def forward(self, x):
        if self.base == 1:
            return x * self.l
        else:
            return x * (self.base ** self.l.to(x.device))


class AdjustLayer(nn.Module):
    def __init__(self, init_scale=1e-4, init_bias=0, base=1):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_scale]).cuda())
        self.bias = nn.Parameter(torch.FloatTensor([init_bias]).cuda())
        self.base = base

    def forward(self, x):
        if self.base == 1:
            return x * self.scale + self.bias
        else:
            return x * (self.base ** self.scale) + self.base ** self.bias - 1


class MELDA(BaseModelFSL):
    def __init__(self, init_trans_scale, epsilon, init_lambda=5, init_adj_scale=1e-4, lambda_base=2, adj_base=2,
                 threshold=0.55, n_augment=1, linsys=False, discriminator=False):
        super(MELDA, self).__init__()
        self.lambda_rr = LambdaLayer(learn_lambda=True, init_lambda=init_lambda, base=lambda_base)
        self.adjust = AdjustLayer(init_scale=init_adj_scale, base=adj_base)
        self.transforms = ThetaP()
        self.trans_scale = nn.Parameter(init_trans_scale * torch.ones([1]))
        self.iter = 1
        self.epsilon = epsilon
        self.n_augment = n_augment
        self.linsys = linsys
        self.discriminator = discriminator

        self.threshold = nn.Parameter(threshold * torch.ones([1]))
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def select_q_target(self, logits):
        target_logits, target_label = torch.max(logits, dim=1)
        selected_index = []
        selected_label = []
        selected_logit = []
        for i, l in enumerate(target_logits):
            if l >= self.epsilon:
                if target_label[i] in selected_label:
                    out_tmp = selected_label.index(target_label[i])
                    if l > selected_logit[out_tmp]:
                        selected_index[out_tmp] = i + 75
                        selected_logit[out_tmp] = target_logits[i]
                    else:
                        pass
                else:
                    selected_index.append(i + 75)
                    selected_label.append(target_label[i])
                    selected_logit.append(target_logits[i])
        return selected_index, selected_label

    def get_target_center(self, query, support, query_index, query_label):
        bins = [[], [], [], [], []]
        class_center = []
        for i, idx in enumerate(query_index):
            bins[query_label[i]].append(query[idx])
        for i, bin in enumerate(bins):
            if bin:
                # bin.append(support[i])
                class_center.append(torch.cat(bin, dim=0))
            else:
                class_center.append(support[i].squeeze())
        return class_center

    def get_target_pattern(self, target_logits, query_emb_high, support_emb_high):
        target_ep, target_ep_label = self.select_q_target(target_logits)

        # print(target_ep)
        # print(target_ep_label)
        # exit()
        target_ep_center = self.get_target_center(query_emb_high, support_emb_high, target_ep, target_ep_label)
        # target_ep_samples = self.get_target_ep_samples(target_ep, query_emb_high)
        # print(target_ep_center.size())
        num_target_center = 0
        for c in target_ep_center:
            num_target_center = num_target_center + c.size(0)
        # num_target_center = 5
        # ones = Variable(torch.unsqueeze(torch.ones(support_tmp.size(0)).to(query.device), 1))
        # ones = Variable(torch.unsqueeze(torch.ones(num_target_center).to(target_logits.device), dim=1))
        num_target_center = support_emb_high.size(0)
        ones = Variable(torch.unsqueeze(torch.ones(num_target_center).to(target_logits.device), dim=1))
        I = Variable(torch.eye(num_target_center).to(target_logits.device))

        # bb = torch.cat(target_ep_center, dim=0)
        # for i in range(len(target_ep_center)):
        #     target_ep_center[i].reshape(-1, 1)
        # bb = torch.cat(target_ep_center, dim=0)
        # bb = torch.stack(target_ep_center)
        # aa = (bb, ones)
        target_ep_center = torch.stack(target_ep_center)
        # y_inner = make_target_float_label(target_ep_center)

        # bb = bb.reshape(-1, 1)
        xx = torch.cat((target_ep_center, ones), dim=1)
        n_way = 5
        n_shot = 1
        y_inner = make_float_label(n_way, n_shot) / np.sqrt(n_way * n_shot)
        # thetaW  = self.rr_woodbury_cs(xx, num_target_center, I, y_inner, self.linsys)
        thetaW = self.rr_woodbury(xx, n_way, n_shot, I, y_inner, self.linsys)
        thetaP = self.transforms(query_emb_high, target_ep_center)  # torch.cat(target_ep_center, dim=0))
        return {"thetaW": thetaW, "thetaP": thetaP, "target_ep": target_ep, "target_ep_label": target_ep_label}

    def iter_select_target(self, logits, selected_index, selected_label):
        target_logits, target_label = torch.max(logits, dim=1)

        for i, l in enumerate(target_logits):
            if l > self.epsilon and i not in selected_index:
                # if target_label[i] in selected_label:
                #     out_tmp = selected_label.index(target_label[i])
                #     if l > target_logits[i]:
                #         selected_index[out_tmp] = i+75

                selected_index.append(i)
                selected_label.append(target_label[i])

        return selected_index, selected_label

    def iter_get_target_pattern(self, target_logits, query_emb_high, support_emb_high, target_ep, target_ep_label):
        target_ep, target_ep_label = self.iter_select_target(target_logits, target_ep, target_ep_label)
        target_ep_center = self.get_target_center(query_emb_high, support_emb_high, target_ep, target_ep_label)
        # print(target_ep_center.size())
        # target_ep_samples = self.get_target_ep_samples(target_ep, query_emb_high)
        # print(target_ep_center.size())
        num_target_center = 0
        for c in target_ep_center:
            num_target_center = num_target_center + c.size(0)
        for i in range(len(target_ep_center)):
            if target_ep_center[i].size(0) == 128:
                pass
            else:
                target_ep_center[i] = target_ep_center[i][:128]
        target_ep_center = torch.stack(target_ep_center)
        # y_inner = make_target_float_label(target_ep_center)

        # bb = bb.reshape(-1, 1)

        num_target_center = 5
        ones = Variable(torch.unsqueeze(torch.ones(num_target_center).to(target_logits.device), dim=1))
        I = Variable(torch.eye(num_target_center).to(target_logits.device))
        # y_inner = make_target_float_label(target_ep_center)
        xx = torch.cat((target_ep_center, ones), dim=1)
        n_way = 5
        n_shot = 1
        y_inner = make_float_label(n_way, n_shot) / np.sqrt(n_way * n_shot)
        # thetaW  = self.rr_woodbury_cs(torch.cat(xx, num_target_center, I, y_inner, self.linsys))
        thetaW = self.rr_woodbury(xx, n_way, n_shot, I, y_inner, self.linsys)
        thetaP = self.transforms(query_emb_high, target_ep_center)
        return {"thetaW": thetaW, "thetaP": thetaP, "target_ep": target_ep, "target_ep_label": target_ep_label}




    def compute_labeled(self, features):
        num_classes = 5
        samples_per_class = 15  # 75 / 5 = 15
        class_indices = [range(i * samples_per_class, (i + 1) * samples_per_class) for i in range(num_classes)]
        # 1. 计算每个类的中心（均值），并归一化（余弦距离对方向敏感，归一化更合理）
        class_centers = []
        for indices in class_indices:
            class_samples = features[list(indices)]  # 类内样本 [15, 128]
            center = torch.mean(class_samples, dim=0)  # 类中心 [128]
            center_norm = nn.functional.normalize(center, p=2, dim=0)  # 归一化到单位向量
            class_centers.append(center_norm)
        class_centers = torch.stack(class_centers)  # [5, 128]

        # 2. 计算类内距离（类内样本与类中心的余弦距离之和）
        intra_dist = 0.0
        for i, indices in enumerate(class_indices):
            class_samples = features[list(indices)]  # [15, 128]
            samples_norm = nn.functional.normalize(class_samples, p=2, dim=1)  # 样本归一化
            # 计算每个样本与类中心的余弦相似度，再转为距离（1 - 相似度）
            cos_sim = torch.sum(samples_norm * class_centers[i], dim=1)  # [15]，内积即余弦相似度（已归一化）
            intra_dist += torch.sum(1 - cos_sim)  # 累加类内余弦距离

        # 3. 计算类间距离（所有类中心之间的余弦距离之和）
        inter_dist = 0.0
        num_centers = class_centers.shape[0]
        for i in range(num_centers):
            for j in range(i + 1, num_centers):  # 只算i<j的对，避免重复
                cos_sim = torch.sum(class_centers[i] * class_centers[j])  # 类中心余弦相似度
                inter_dist += (1 - cos_sim)  # 累加类间余弦距离

        # 4. 构造损失：类内距离 / (类间距离 + 微小值)，最小化该值
        eps = 1e-8
        loss = intra_dist / (inter_dist + eps)
        return loss



    def compute_unlabel(self, x):
        batch_size = x.size(0)
        total_intra_loss = 0.0  # 类内损失
        total_inter_loss = 0.0  # 类间损失
        count_intra = 0  # 类内样本对计数
        count_inter = 0  # 类间样本对计数

        # 遍历所有样本对
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                # 计算余弦相似度
                sim = self.cosine_sim(x[i].unsqueeze(0), x[j].unsqueeze(0))[0]
                cosine_dist = 1 - sim  # 余弦距离（1-相似度）

                # 根据阈值判断是否为同类
                if sim > self.threshold:
                    # 类内：希望距离越小越好（损失为距离本身）
                    total_intra_loss += cosine_dist
                    count_intra += 1
                else:
                    # 类间：希望距离越大越好（损失为负距离）
                    total_inter_loss -= cosine_dist
                    count_inter += 1

        # 防止除零（如果没有同类或异类样本对）
        if count_intra == 0:
            intra_loss = torch.tensor(0.0, device=x.device)
        else:
            intra_loss = total_intra_loss / count_intra

        if count_inter == 0:
            inter_loss = torch.tensor(0.0, device=x.device)
        else:
            inter_loss = total_inter_loss / count_inter

        # 总损失：类内损失 + 类间损失（两者都希望最小化）
        total_loss = intra_loss + inter_loss
        return total_loss

    def forward(self, query, support, train=True):
        # way, nshot*k, C, h, w
        query_num, c = query.size()
        output_dim = c
        _, n_way, n_shot, _ = support.size()
        if self.discriminator:
            support_tmp = support.contiguous().view(-1, output_dim)
            n_shot = n_way * n_shot
            n_way = 2
        else:
            support_tmp = support[0].contiguous().view(n_way * n_shot, output_dim)
        if n_way * n_shot > output_dim + 1:
            rr_type = 'standard'
            I = Variable(torch.eye(output_dim + 1).to(query.device))
        else:
            rr_type = 'woodbury'
            I = Variable(torch.eye(support_tmp.size(0)).to(query.device))
        y_inner = make_float_label(n_way, n_shot) / np.sqrt(n_way * n_shot)
        ones = Variable(torch.unsqueeze(torch.ones(support_tmp.size(0)).to(query.device), 1))
        # rr_type = 'standard'
        if rr_type == 'woodbury':
            aa = torch.cat((support_tmp, ones), 1)
            wb = self.rr_woodbury(aa, n_way, n_shot, I, y_inner, self.linsys)
        else:
            wb = self.rr_standard(torch.cat((support_tmp, ones), 1), n_way, n_shot, I, y_inner, self.linsys)

        W = wb.narrow(0, 0, output_dim)
        B = wb.narrow(0, output_dim, 1)
        query_s = query[:int(query_num / 2), :]
        query_t = query[int(query_num / 2):, :]
        # source_out = mm(query_s, W) + B

        # # if not train:
        # target_out = mm(query_t, W.detach()) + B.detach()
        # results = self.get_target_pattern(target_out, query,
        #                                   support[0].contiguous().view(n_way, n_shot, output_dim))
        # thetaW, thetaP, target_ep, target_ep_label = results["thetaW"], results["thetaP"], results["target_ep"], \
        #                                              results["target_ep_label"]
        #
        # W = thetaW.narrow(0, 0, output_dim)
        # B = thetaW.narrow(0, output_dim, 1)
        # target_out = mm(self.trans_scale * mm(query_t, thetaP), W.detach()) + B.detach()
        # for i in range(self.iter):
        #     results = self.iter_get_target_pattern(target_out, query, support_tmp, target_ep, target_ep_label)
        #     thetaW, thetaP, target_ep, target_ep_label = results["thetaW"], results["thetaP"], results["target_ep"], \
        #                                                  results["target_ep_label"]
        #     W = thetaW.narrow(0, 0, output_dim)
        #     B = thetaW.narrow(0, output_dim, 1)
        #     target_out = mm(self.trans_scale * mm(query_t, thetaP), W.detach()) + B.detach()
        # # else:
        # #     target_out = mm(query_t, W) + B
        # out = torch.cat([source_out, target_out], dim=0)

        # if train==True:
        #     w = wb.narrow(0, 0, output_dim)  # (dimension=0, start=0, length=self.output_dim)
        #     b = wb.narrow(0, output_dim, 1)  # (dimension=0, start=self.output_dim, length=1)
        #     out = mm(query, w) + b
        # else:
        #     theta_p = self.transforms(query, support.contiguous().view(2, n_way * n_shot, output_dim))
        #     w = wb.narrow(0, 0, output_dim)  # (dimension=0, start=0, length=self.output_dim)
        #     b = wb.narrow(0, output_dim, 1)  # (dimension=0, start=self.output_dim, length=1)
        #     w = mm(theta_p, w)
        #     out = mm(query, w)+b\
        # if self.discriminator:
        #     source_out = mm(query_s, W.detach()) + B.detach()
        #     target_out = mm(query_t, W.detach()) + B.detach()
        #     out = torch.cat([source_out, target_out], dim=0)
        # else:
        la_loss = self.compute_labeled(query_s)
        un_loss = self.compute_unlabel(query_t)
        loss_lda = la_loss + un_loss

        # loss_lda = torch.tensor(0.0)


        if not train:
            source_out = mm(query_s, W.detach()) + B.detach()
            thetaP = self.transforms(query_t, support.contiguous().view(2, n_way * n_shot, output_dim))
            # u, s, v = torch.svd(thetaP.double(), some=False)
            target_out = mm(self.trans_scale * mm(query_t, thetaP), W.detach()) + B.detach()
            # target_out = mm(query_t, W.detach()) + B.detach()
            out = torch.cat([source_out, target_out], dim=0)
            probs = F.softmax(target_out, dim=1)
            loss_div = torch.mean(torch.sum(-probs * torch.log(probs + 1e-8), dim=1))
        else:
            thetaP = self.transforms(query, support.contiguous().view(2, n_way * n_shot, output_dim))
            # target_out = mm(query, W) + B
            # u, s, v = torch.svd(thetaP.double(), some=False)
            out = mm(self.trans_scale * mm(query, thetaP), W) + B
            probs = F.softmax(out[75:], dim=1)
            loss_div = torch.mean(torch.sum(-probs * torch.log(probs + 1e-8), dim=1))
            # out = mm(query, W) + B
        # out = torch.cat([source_out, target_out], dim=0)
        y_hat = self.adjust(out)
        y_hat = nn.functional.normalize(y_hat, dim=1, p=1, eps=1e-12)
        return y_hat, loss_div, loss_lda

    def rr_standard(self, x, n_way, n_shot, I, yrr_binary, linsys):
        x /= np.sqrt(n_way * n_shot * self.n_augment)

        if not linsys:
            aa = mm(x, torch.transpose(x, 0, 1), )
            bb = self.lambda_rr(I)
            cc = torch.inverse(aa + bb)
            w = mm(mm(torch.transpose(x, 0, 1), cc), yrr_binary)
        else:
            A = mm(t_(x), x) + self.lambda_rr * I
            v = mm(t_(x), yrr_binary)
            w, _ = torch.solve(v, A)
        return w

    def rr_woodbury(self, x, n_way, n_shot, I, yrr_binary, linsys):
        # x /= np.sqrt(n_way * n_shot * self.n_augment)
        if not linsys:
            a1 = torch.transpose(x, 0, 1)
            a2 = self.lambda_rr(I)
            a3 = torch.inverse(mm(x, torch.transpose(x, 0, 1)) + a2)
            a5 = mm(a1, a3)
            w = mm(a5, yrr_binary)
        else:
            A = mm(x, t_(x)) + self.lambda_rr * I
            v = yrr_binary
            w_, _ = torch.solve(v, A)
            w = mm(t_(x), w_)
        return w

    def rr_woodbury_cs(self, x, nnum, I, yrr_binary, linsys):
        x /= np.sqrt(nnum * self.n_augment)

        if not linsys:
            w = mm(mm(torch.transpose(x, 0, 1), torch.inverse(mm(x, torch.transpose(x, 0, 1)) + self.lambda_rr(I))),
                   yrr_binary)
        else:
            A = mm(x, t_(x)) + self.lambda_rr * I
            v = yrr_binary
            w_, _ = torch.solve(v, A)
            w = mm(t_(x), w_)
        return w


class MCDR2D2(MELDA):
    def __init__(self, init_lambda=5, init_adj_scale=1e-4, lambda_base=2, adj_base=2, n_augment=1, linsys=False,
                 discri=False):
        super(MCDR2D2, self).__init__(init_lambda, init_adj_scale, lambda_base, adj_base, n_augment, linsys, False)

    def forward(self, query, support):
        # way, nshot*k, C, h, w
        support = torch.stack(support, 0)
        way, _, C, h, w = support.size()
        support = support.view(way, 1, -1, C, h, w).permute(1, 0, 2, 3, 4, 5)[0]
        support = support.view(*support.size()[:2], -1)
        query = query.view(query.size(0), -1)
        output_dim = support.size(-1)
        n_way, n_shot = support.size()[:2]
        support = support.reshape(-1, output_dim)
        n_query = query.size(0)
        if n_way * n_shot > output_dim + 1:
            rr_type = 'standard'
            I = Variable(torch.eye(output_dim + 1).to(query.device))
        else:
            rr_type = 'woodbury'
            I = Variable(torch.eye(n_way * n_shot).to(query.device))
        y_inner = make_float_label(n_way, n_shot) / np.sqrt(n_way * n_shot)

        ones = Variable(torch.unsqueeze(torch.ones(support.size(0)).to(query.device), 1))
        # print(support.size(), ones.size())
        # exit()
        if rr_type == 'woodbury':
            wb = self.rr_woodbury(torch.cat((support, ones), 1), n_way, n_shot, I, y_inner, self.linsys)
        else:
            wb = self.rr_standard(torch.cat((support, ones), 1), n_way, n_shot, I, y_inner, self.linsys)
        w = wb.narrow(0, 0, output_dim)  # (dimension=0, start=0, length=self.output_dim)
        b = wb.narrow(0, output_dim, 1)  # (dimension=0, start=self.output_dim, length=1)
        out = mm(query, w) + b
        y_hat = self.adjust(out)
        return y_hat


class ThetaP(nn.Module):
    def __init__(self, iternum=10, init_lambda=1, init_gama=750, init_beta=1, init_eta=1, init_adj_scale=1e-4,
                 lambda_base=2, adj_base=2, n_augment=1, linsys=False, discriminator=False):
        super(ThetaP, self).__init__()
        self.lambda_rr = init_lambda
        self.gama = LambdaLayer(learn_lambda=True, init_lambda=init_gama)
        self.beta = init_beta
        self.eta = init_eta
        self.n_augment = n_augment
        self.linsys = linsys
        self.iternum = iternum
        self.discriminator = discriminator

    def euc(self, target, source):
        n_target, C = target.size()
        n_source, C = source.size()
        target = target.view(n_target, -1, C)
        # target = target.unsqueeze(1)
        source = source.view(1, n_source, C)
        # source = source.unsqueeze(0)
        dist_mat = (target.expand(n_target, n_source, C) - source.expand(n_target, n_source, C)).pow(2).sum(-1)
        return dist_mat

    def cos(self, target, source):
        n_target, C = target.size()
        n_source, C = source.size()

        target = target.view(n_target, C)
        source = source.view(n_source, C)
        dist_mat = 1 - (target / target.norm(dim=-1, keepdim=True)) @ (source / source.norm(dim=-1, keepdim=True)).t()
        return dist_mat

    def sinkhorn(self, A, epsilon=1e-7):
        while True:
            A_r = A / A.sum(1, keepdim=True)
            A_c = A_r / A_r.sum(0, keepdim=True)
            if ((torch.max(A_c) - torch.min(A)) - (torch.max(A) - torch.min(A_c))).abs() < epsilon:
                break
            A = A_c
        return A_c

    def sinkhornv2(self, A, epsilon=4e-7):
        while True:
            A_r = A / A.sum(1, keepdim=True)
            A_c = A_r / A_r.sum(0, keepdim=True)
            now = (A - A_c).abs().sum()
            if (A - A_c).abs().sum() < epsilon:
                break
            pre = now
            A = A_c
        return A_c

    def get_Av1(self, source, target, A_k):
        dist_mat = -self.euc(target, source)
        dist_mat = (80 * nn.functional.normalize(dist_mat, dim=0, p=2)).exp()
        A = self.sinkhorn(dist_mat)
        _, index = torch.topk(A, k=A_k, dim=0, sorted=False, largest=True)
        A_s2t = torch.zeros_like(dist_mat).to(dist_mat.device)
        A = A_s2t.scatter_(0, index, _)
        # A = A + map_ones
        return A * self.eta

    def get_Av2(self, source, target, A_k):
        dist_mat = -self.euc(target, source)
        dist_mat = 10 * nn.functional.normalize(dist_mat, dim=0, p=2)
        _, index = torch.topk(dist_mat, k=A_k, dim=0, sorted=False, largest=True)
        A_s2t = torch.zeros_like(dist_mat).to(dist_mat.device)
        A = A_s2t.scatter_(0, index, _.exp())
        # A = A + map_ones
        return A / A.sum(0, keepdim=True) * self.eta

    def get_Av3(self, source, target):
        dist_mat = -self.euc(target, source)
        A = (80 * nn.functional.normalize(dist_mat, dim=0, p=2)).exp()
        return A / A.sum(1, keepdim=True) * self.eta

    def get_Av4(self, source, target):

        dist_mat = -self.euc(target, source[0])  # source[0].squeeze())
        #dist_mat = -self.cos(target, source[0].squeeze())
        A = (98 * nn.functional.normalize(dist_mat, dim=0, p=2)).exp()
        A = self.sinkhorn(A)
        return A * self.eta

    def get_Av5(self, source, target, A_k):
        dist_mat = -self.euc(target, source)
        A = (80 * nn.functional.normalize(dist_mat, dim=0, p=2)).exp()
        A = self.sinkhorn(A)
        _, index = torch.topk(A, k=A_k, dim=0, sorted=False, largest=True)
        A = torch.zeros_like(A).to(source.device)
        A = A.scatter_(0, index, _)
        return A * self.eta

    def get_Av6(self, source, target):
        dist_mat = 1 - self.cos(target, source[0])
        A = (75 * nn.functional.normalize(dist_mat, dim=0, p=2)).exp()
        A = self.sinkhorn(A)
        return A * self.eta

    def get_Av7(self, source, target):
        dist_mat = 1 - self.cos(target, source)
        A = (75 * nn.functional.normalize(dist_mat, dim=0, p=2)).exp()
        return A * self.eta

    def KNN(self, source, target):
        # S->T
        A = self.get_Av6(source, target)
        # T->T
        S = torch.ones_like(A)
        return A, S

    def forward(self, query, support):
        # way, nshot*k, C, h, w
        # cal knn adj
        # n_query, outputdim = query.size()
        # if self.discriminator:
        #     support_tmp = support.contiguous().view(-1, output_dim)
        #     n_shot = n_way * n_shot
        #     n_way = 2
        # else:
        #     support_tmp = support[0].contiguous().view(n_way * n_shot, output_dim)
        # _, n_way, n_shot, _ = support.size()
        # support = torch.mean(support, dim=2, keepdim=True)
        # n_way = support.shape[1]
        # n_shot = support.shape[2]
        # if self.discriminator:
        #     support_tmp = support.contiguous().view(2, n_way, n_shot, support.shape[3])
        #     n_shot = n_way * n_shot
        #     n_way = 2
        # else:
        #     support_tmp = support.contiguous().view(2, n_way, n_shot, support.shape[3])
        # y_inner = make_float_label(n_way, n_shot) / np.sqrt(n_way * n_shot)
        A, S = self.KNN(support, query)
        A = A.to(query.device)
        S = S.to(query.device)
        wb = self.rr_woodbury2(support[0], query, A, S)

        # w = wb.narrow(0, 0, 128)  # (dimension=0, start=0, length=self.output_dim)
        # b = wb.narrow(0, 128, 1)  # (dimension=0, start=self.output_dim, length=1)
        # out = mm(query, w) + b
        return wb  # out

    def rr_standard(self, x, n_way, n_shot, I, yrr_binary, linsys):
        x /= np.sqrt(n_way * n_shot * self.n_augment)
        if not linsys:
            w = mm(mm(torch.inverse(mm(torch.transpose(x, 0, 1), x) + self.lambda_rr * I), torch.transpose(x, 0, 1)),
                   yrr_binary)
        else:
            A = mm(t_(x), x) + self.lambda_rr * I
            v = mm(t_(x), yrr_binary)
            w, _ = torch.solve(v, A)
        return w

    def rr_woodbury(self, support, query_t, A, S, R, n_way, n_shot, I, linsys, train):
        support /= np.sqrt(n_way * n_shot * self.n_augment)
        support_s = support[0].reshape([n_way * n_shot, 1600])
        support_t = support[1].reshape([n_way * n_shot, 1600])
        M = torch.diag(torch.sum(A, dim=0)) + 2 * self.lambda_rr * torch.diag(
            torch.sum(S, dim=1)) - 2 * self.lambda_rr * torch.diag(torch.sum(S, dim=1)) - 2 * self.lambda_rr * S
        M = M.to(support.device)
        w = mm(mm(mm(torch.transpose(support_t, 0, 1),
                     torch.inverse(self.gama * I + mm(M, mm(support_t, torch.transpose(support_t, 0, 1))))),
                  torch.transpose(A, 0, 1)), support_s)
        return w

    def rr_woodbury2(self, support_s, query_t, A, S):
        M = torch.diag(torch.sum(A, dim=1))
        M = M.to(support_s.device)
        A = A + torch.ones_like(A).to(query_t.device) * self.beta  # (150, 5)
        M = M + torch.ones_like(M).to(query_t.device) * self.beta  # (150, 150)
        I = torch.eye(query_t.size(0)).to(support_s.device)
        # ones = Variable(torch.unsqueeze(torch.ones(query_t.size(0)).to(query_t.device), 1))
        # x = torch.cat((query_t, ones), 1)
        # a1 = torch.transpose(x, 0, 1) # (129, 150)
        # a2 = self.gama(I) # (150, 150)
        # a3 = mm(M, mm(x, a1)) # (150, 150)
        # a4 = torch.inverse(a2 + a3) #(150, 150)
        # a5 = mm(mm(a1, a4), A) #(129, 5)
        # # ones = Variable(torch.unsqueeze(torch.ones(support_s.size(0)).to(support_s.device), 1))
        # a6 = torch.cat((support_s, ones), 1)
        # w = mm(a5, mm(a6, a6.T))
        support_ss = support_s.squeeze()
        a = mm(torch.transpose(query_t, 0, 1),
               torch.inverse(I + mm(M, mm(query_t, torch.transpose(query_t, 0, 1)))))
        w = mm(mm(mm(torch.transpose(query_t, 0, 1),
                     torch.inverse(self.gama(I) + mm(M, mm(query_t, torch.transpose(query_t, 0, 1))))), A), support_ss) # self.gama(I)
        return w


class R2D2Trans(MELDA):
    def __init__(self, init_trans_scale=60, init_eta=1, init_lambda=1, init_gama=750, init_beta=1,
                 iteration=1, epsilon=4e-7, tri_margin=1, init_lambda_r2d2=5, init_adj_scale=1e-4, lambda_base=2,
                 adj_base=2, n_augment=1, linsys=False, discriminator=False):
        super(R2D2Trans, self).__init__(init_lambda_r2d2, init_adj_scale, lambda_base, adj_base, n_augment, linsys,
                                        False)
        self.transform = ThetaP(init_eta, init_lambda, init_gama, init_beta, iteration)
        self.trans_scale = nn.Parameter(init_trans_scale * torch.ones([1]))
        self.iter = 1
        self.epsilon = epsilon
        self.tri_margin = tri_margin
        self.discriminator = discriminator

    def get_triplet_loss(self, similarity):
        triplet_loss = torch.tensor(0.).to(similarity.device)
        poses, _ = torch.max(similarity, dim=1)
        most_sim = similarity.topk(k=2, dim=1, largest=True, sorted=True)[0][:, 1]
        # poses = poses.unsqueeze(1).repeat(1, 4)
        losses = most_sim - poses + self.tri_margin
        for loss in losses:
            if loss > 0.:
                triplet_loss += loss
        return triplet_loss

    def select_q_target(self, logits):
        target_logits, target_label = torch.max(logits, dim=1)
        selected_index = []
        selected_label = []
        selected_logit = []
        for i, l in enumerate(target_logits):
            if l >= self.epsilon:
                if target_label[i] in selected_label:
                    out_tmp = selected_label.index(target_label[i])
                    if l > selected_logit[out_tmp]:
                        selected_index[out_tmp] = i + 75
                        selected_logit[out_tmp] = target_logits[i]
                    else:
                        pass
                else:
                    selected_index.append(i + 75)
                    selected_label.append(target_label[i])
                    selected_logit.append(target_logits[i])
                # if i+75 in selected_index:
                #     out_tmp = selected_index.index(i+75)
                #     if l > selected_logit[out_tmp]
        return selected_index, selected_label

    def select_q_target_topk(self, logits):
        target_logits, target_label = torch.max(logits, dim=1)
        selected_index = []
        selected_label = []
        target_logits_topk, topk_index = torch.topk(target_logits, k=10)
        for i, index in enumerate(topk_index):
            selected_index.append(index + 75)
            selected_label.append(target_label[index])
        return selected_index, selected_label

    def get_target_center(self, query, support, query_index, query_label):
        bins = [[], [], [], [], []]
        class_center = []
        for i, idx in enumerate(query_index):
            bins[query_label[i]].append(query[idx])
        for i, bin in enumerate(bins):
            if bin:
                # bin.append(support[i])
                class_center.append(torch.cat(bin, dim=0))
            else:
                class_center.append(support[i].squeeze())
        return class_center

    def get_target_ep_samples(self, target_ep, query_emb_high):
        selected_vector = []
        for i in target_ep:
            selected_vector.append(query_emb_high[i])
        return torch.stack(selected_vector)

    def get_target_pattern(self, target_logits, query_emb_high, support_emb_high):
        target_ep, target_ep_label = self.select_q_target(target_logits)

        # print(target_ep)
        # print(target_ep_label)
        # exit()
        target_ep_center = self.get_target_center(query_emb_high, support_emb_high, target_ep, target_ep_label)
        # target_ep_samples = self.get_target_ep_samples(target_ep, query_emb_high)
        # print(target_ep_center.size())
        num_target_center = 0
        for c in target_ep_center:
            num_target_center = num_target_center + c.size(0)

        ones = Variable(torch.unsqueeze(torch.ones(num_target_center).to(target_logits.device), dim=1))
        I = Variable(torch.eye(num_target_center).to(target_logits.device))
        y_inner = make_target_float_label(target_ep_center)
        thetaW = self.rr_woodbury_cs(torch.cat((torch.cat(target_ep_center, dim=0), ones), 1), num_target_center, I,
                                     y_inner, self.linsys)
        thetaP = self.transform(query_emb_high[75:], torch.cat(target_ep_center, dim=0))
        return {"thetaW": thetaW, "thetaP": thetaP, "target_ep": target_ep, "target_ep_label": target_ep_label}

    def iter_select_target(self, logits, selected_index, selected_label):
        target_logits, target_label = torch.max(logits, dim=1)
        for i, l in enumerate(target_logits):
            if l > self.epsilon and i not in selected_index:
                selected_index.append(i)
                selected_label.append(target_label[i])
        return selected_index, selected_label

    def iter_get_target_pattern(self, target_logits, query_emb_high, support_emb_high, target_ep, target_ep_label):
        target_ep, target_ep_label = self.iter_select_target(target_logits, target_ep, target_ep_label)
        target_ep_center = self.get_target_center(query_emb_high, support_emb_high, target_ep, target_ep_label)
        # print(target_ep_center.size())
        # target_ep_samples = self.get_target_ep_samples(target_ep, query_emb_high)
        # print(target_ep_center.size())
        num_target_center = 0
        for c in target_ep_center:
            num_target_center = num_target_center + c.size(0)
        ones = Variable(torch.unsqueeze(torch.ones(num_target_center).to(target_logits.device), dim=1))
        I = Variable(torch.eye(num_target_center).to(target_logits.device))
        y_inner = make_target_float_label(target_ep_center)
        thetaW = self.rr_woodbury_cs(torch.cat((torch.cat(target_ep_center, dim=0), ones), 1), num_target_center, I,
                                     y_inner, self.linsys)
        thetaP = self.transform(query_emb_high[75:], torch.cat(target_ep_center, dim=0))
        return {"thetaW": thetaW, "thetaP": thetaP, "target_ep": target_ep, "target_ep_label": target_ep_label}

    def get_source_center(self, query, support):
        source_center = []
        for i, sup in enumerate(support):
            source_center.append(torch.cat([sup.unsqueeze(0), query[15 * i: 15 * (i + 1)]], dim=0).mean(dim=0))
        return torch.stack(source_center)

    def get_multi_shot_center(self, target_samples):
        prototypes = torch.stack(target_samples[0:self.shot]).contiguous().view(5, -1)
        bin = [[p] for p in target_samples[0:self.shot]]
        others = torch.stack(target_samples[self.shot:]).contiguous().view(len(target_samples) - 5, -1)
        _, sim = torch.mm(others, prototypes.transpose(1, 0)).max(dim=1)
        for i in range(others.size(0)):
            bin[sim[i]].append(target_samples[self.shot:][i])
        for i, b in enumerate(bin):
            bin[i] = torch.stack(b).mean(dim=0)
        return torch.stack(bin).view(-1, bin[0].size()[-1])

    def lack_shot(self, support, target_samples):
        n = len(target_samples)
        support = support.view(self.shot, -1, support.size()[-1])
        for i in range(self.shot - n):
            target_samples.append(support.mean(0))
        return torch.stack(target_samples).view(-1, support.size()[-1])

    def forward(self, query, support, train=True):
        # support = torch.stack(support, 0)
        # print(query.size())
        # print(support.size())
        # exit()
        query_num, C = query.size()
        query_s = query[:int(query_num / 2), :]
        query_t = query[int(query_num / 2):, :]
        output_dim = C
        _, n_way, n_shot, c = support.size()
        if self.discriminator:
            support_tmp = support.contiguous().view(-1, output_dim)
            n_shot = n_way * n_shot
            n_way = 2
        else:
            support_tmp = support[0].contiguous().view(n_way * n_shot, output_dim)
        I = Variable(torch.eye(support_tmp.size(0)).to(query.device))
        y_inner = make_float_label(n_way, n_shot) / np.sqrt(n_way * n_shot)
        ones = Variable(torch.unsqueeze(torch.ones(support_tmp.size(0)).to(query.device), 1))
        wb = self.rr_woodbury(torch.cat((support_tmp, ones), 1), n_way, n_shot, I, y_inner, self.linsys)
        W = wb.narrow(0, 0, output_dim)  # (dimension=0, start=0, length=self.output_dim)
        B = wb.narrow(0, output_dim, 1)  # (dimension=0, start=self.output_dim, length=1)
        source_out = mm(query_s, W) + B

        if not train:
            # transform = self.transform(query_t, support_tmp)
            target_out = mm(query_t, W.detach()) + B.detach()
            # print(target_out.max(dim=1), target_out)
            # exit()
            # y_hat = self.adjust(target_out)
            # y_hat = nn.functional.normalize(y_hat, dim=1, p=1, eps=1e-12)
            # target_out = mm(query_t, W) + B
            # select query T'
            results = self.get_target_pattern(target_out, query,
                                              support[0].contiguous().view(n_way, n_shot, output_dim))
            thetaW, thetaP, target_ep, target_ep_label = results["thetaW"], results["thetaP"], results["target_ep"], \
                                                         results["target_ep_label"]

            W = thetaW.narrow(0, 0, output_dim)
            B = thetaW.narrow(0, output_dim, 1)
            target_out = mm(self.trans_scale * mm(query_t, thetaP), W.detach()) + B.detach()
            # print(target_out.max(dim=1), target_out)
            # exit()
            # target_out= mm(self.trans_scale * mm(query_t, thetaP), W.detach()) + B.detach()
            for i in range(self.iter):
                results = self.iter_get_target_pattern(target_out, query, support_tmp, target_ep, target_ep_label)
                thetaW, thetaP, target_ep, target_ep_label = results["thetaW"], results["thetaP"], results["target_ep"], \
                                                             results["target_ep_label"]
                W = thetaW.narrow(0, 0, output_dim)
                B = thetaW.narrow(0, output_dim, 1)
                target_out = mm(self.trans_scale * mm(query_t, thetaP), W.detach()) + B.detach()
        else:
            target_out = mm(query_t, W) + B
        out = torch.cat([source_out, target_out], dim=0)
        y_hat = self.adjust(out)
        y_hat = nn.functional.normalize(y_hat, dim=1, p=1, eps=1e-12)
        # triplet_loss = self.get_triplet_loss(target_out_final)
        return {'logits': y_hat}


class MELDA_DN4(R2D2Trans):
    def __init__(self, init_trans_scale, init_eta, init_lambda, init_gama, init_beta,
                 iteration, init_lambda_r2d2=5, init_adj_scale=1e-4, lambda_base=2, adj_base=2, n_augment=1,
                 linsys=False):
        super(MELDA_DN4, self).__init__(init_lambda_r2d2, init_adj_scale, lambda_base, adj_base, n_augment, linsys,
                                        False)
        self.encoder = ImgtoClass_Metric(neighbor_k=3)

    def forward(self, query, support, train=True):
        # support = torch.stack(support, 0)
        # print(query.size())
        # print(support.size())
        # exit()

        query_num, h_w, C = query.size()
        query_s = query[:int(query_num / 2), :, :]
        query_t = query[int(query_num / 2):, :, :].contiguous().view(int(query_num / 2), -1)
        output_dim = C
        _, n_way, n_shot, _, c = support.size()
        support_tmp = support[0].contiguous().view(-1, h_w, output_dim)
        query_t_tmp = query[int(query_num / 2):, :, :]
        # if not train:
        #     support_s = support[0].contiguous().view(n_way * n_shot, -1)
        #     # query_t_tmp = query_t.contiguous().view(n_way * n_shot, -1)
        #     transform = self.transform(query_t, support_s)
        #     query_t_tmp = mm(query_t, transform).view(int(query_num/2), h_w, -1)
        query_tmp = torch.cat([query_s, query_t_tmp], dim=0)
        encoder_res = self.encoder(query_tmp, support_tmp)

        # target_out = torch.topk(target_out, dim=1, k=3)[0].sum(1)
        # source_out = torch.topk(source_out, dim=1, k=3)[0].sum(1)

        out = encoder_res['logits']
        y_hat = self.adjust(out)

        y_hat = nn.functional.normalize(y_hat, dim=1, p=1, eps=1e-12)
        return {'logits': y_hat}


class ImgtoClass_Metric(BaseModelFSL):
    def __init__(self, neighbor_k=3):
        super(ImgtoClass_Metric, self).__init__()
        self.neighbor_k = neighbor_k

    def cal_cosinesimilarity(self, input1, input2):
        B, h_w, C = input1.size()
        # input2_tmp = input2.contiguous().view(-1, h_w, C)
        input2_tmp = input2.permute(0, 2, 1)
        support_norm = torch.norm(input2_tmp, 2, 1, True)
        query_norm = torch.norm(input1, 2, 2, True)
        Similarity_matrix = torch.einsum('ijk, bkc->ibjc', input1, input2_tmp)
        Similarity_matrix_norm = torch.einsum('ijk,bkc->ibjc', query_norm, support_norm)
        similar_matrix, _ = torch.topk((Similarity_matrix / Similarity_matrix_norm), self.neighbor_k, -1)
        return similar_matrix.sum(-1).sum(-1)

    def forward_(self, query, support):
        logits = self.cal_cosinesimilarity(query, support)
        return {'logits': logits}


class MELDA_IMSE(R2D2Trans):
    def __init__(self, init_trans_scale, init_eta, init_lambda, init_gama, init_beta,
                 iteration, init_lambda_r2d2=5, init_adj_scale=1e-4, lambda_base=2, adj_base=2, n_augment=1,
                 linsys=False):
        super(MELDA_IMSE, self).__init__(init_lambda_r2d2, init_adj_scale, lambda_base, adj_base, n_augment, linsys,
                                         False)
        self.encoder = SimilarityEncoderV3(1, 0.8, 2)

    def get_patterns(self, query_emb, support_emb):
        patterns = torch.einsum('ijk,lpk->ijlp', query_emb / (query_emb.norm(dim=-1, keepdim=True) + 1e-10),
                                support_emb / (support_emb.norm(dim=-1, keepdim=True) + 1e-10))
        topk_v, topk_i = patterns.topk(3, sorted=False)
        patterns = torch.zeros_like(patterns).scatter(-1, topk_i, topk_v)
        return patterns

    def forward(self, query, support, train=True):
        # support = torch.stack(support, 0)
        # print(query.size())
        # print(support.size())
        # exit()

        query_num, h_w, C = query.size()
        query_s = query[:int(query_num / 2), :, :]
        query_t = query[int(query_num / 2):, :, :].contiguous().view(int(query_num / 2), -1)
        output_dim = C
        _, n_way, n_shot, _, c = support.size()
        support_tmp = support[0].contiguous().view(-1, h_w, output_dim)
        query_t_tmp = query[int(query_num / 2):, :, :]
        # if not train:
        #     support_s = support[0].contiguous().view(n_way * n_shot, -1)
        #     # query_t_tmp = query_t.contiguous().view(n_way * n_shot, -1)
        #     transform = self.transform(query_t, support_s)
        #     query_t_tmp = mm(query_t, transform).view(int(query_num/2), h_w, -1)
        query_tmp = torch.cat([query_s, query_t_tmp], dim=0)
        patterns = self.get_patterns(query_tmp, support_tmp)
        encoder_res = self.encoder(patterns)

        # target_out = torch.topk(target_out, dim=1, k=3)[0].sum(1)
        # source_out = torch.topk(source_out, dim=1, k=3)[0].sum(1)

        out = encoder_res['logits']
        # y_hat = self.adjust(out)

        # y_hat = nn.functional.normalize(y_hat, dim=1, p=1, eps=1e-12)
        return {'logits': out}


class MELDA_SVM(BaseModelFSL):
    def __init__(self, kernel='linear', max_iter=100):
        super(MELDA_SVM, self).__init__()
        self.SVMs = [svm() for i in range(5)]

    def get_predict(self, query):
        logits = []
        for i in range(5):
            logit = self.SVMs[i].get_logit(query)
            logits.append(logit)
        return torch.cat(logits, dim=0)

    def forward(self, query, support, labels):
        _, n_way, n_shot, h_w, c = support.size()
        support_tmp = support[0].contiguous().view(n_way * n_shot, -1)
        labels = labels[0]
        query = query.contiguous().view(-1, h_w * c)
        for i in range(5):
            labels_tmp = torch.zeros(labels.size(0)).cuda()
            for j in range(labels.size(0)):
                if labels[j] == i:
                    labels_tmp[j] = 1
                else:
                    labels_tmp[j] = -1
            self.SVMs[i].fit(support_tmp, labels_tmp)

        logits = self.get_predict(query)
        return {'logits': logits}


class svm:
    def __init__(self, max_iter=1, kernel='linear'):
        '''
        input:max_iter(int):最大训练轮数
              kernel(str):核函数，等于'linear'表示线性，等于'poly'表示多项式
        '''
        self.max_iter = max_iter
        self._kernel = kernel

    # 初始化模型
    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels.unsqueeze(dim=1)
        self.b = torch.tensor(0., dtype=float).cuda()
        # 将Ei保存在一个列表里
        self.alpha = torch.ones(self.m).cuda().unsqueeze(dim=0)
        self.E = torch.mm(torch.mm(self.alpha, torch.mm(self.X, torch.transpose(self.X))), self.Y)
        self.Y = labels
        # 错误惩罚参数
        self.C = 1.0

    # kkt条件
    def _KKT(self, i):
        y_g = self._g(i) * self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1

    # g(x)预测值，输入xi（X[i]）
    def _g(self, i):
        r = self.b
        for j in range(self.m):
            r += self.alpha[j] * self.Y[j] * self.kernel(self.X[i], self.X[j])
        return r

    # 核函数,多项式添加二次项即可
    def kernel(self, x1, x2):
        if self._kernel == 'linear':
            return torch.mul(x1, x2)
        elif self._kernel == 'poly':
            return (torch.mul(x1, x2) + 1) ** 2
        return 0

    # E（x）为g(x)对输入x的预测值和y的差
    def _E(self, i):
        return self._g(i) - self.Y[i]

    # 初始alpha
    def _init_alpha(self):
        # 外层循环首先遍历所有满足0<a<C的样本点，检验是否满足KKT
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)
        for i in index_list:
            if self._KKT(i):
                continue
            E1 = self.E[i]
            # 如果E2是+，选择最小的；如果E2是负的，选择最大的
            if E1 >= 0:
                j = min(range(self.m), key=lambda x: self.E[x])
            else:
                j = max(range(self.m), key=lambda x: self.E[x])
            return i, j

    # 选择alpha参数
    def _compare(self, _alpha, L, H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha

    # 训练
    def fit(self, features, labels):
        '''
        input:features(ndarray):特征
              label(ndarray):标签
        '''
        self.init_args(features, labels)
        for t in range(self.max_iter):
            i1, i2 = self._init_alpha()
            # 边界
            if self.Y[i1] == self.Y[i2]:
                L = torch.max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = torch.min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = torch.max(0, self.alpha[i2] - self.alpha[i1])
                H = torch.min(self.C, self.C + self.alpha[i2] - self.alpha[i1])
            E1 = self.E[i1]
            E2 = self.E[i2]
            # eta=K11+K22-2K12
            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(self.X[i2], self.X[i2]) - 2 * self.kernel(
                self.X[i1], self.X[i2])
            if eta <= 0:
                continue
            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E2 - E1) / eta
            alpha2_new = self._compare(alpha2_new_unc, L, H)
            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2] - alpha2_new)
            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (alpha1_new - self.alpha[i1]) - self.Y[
                i2] * self.kernel(self.X[i2], self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1]) - self.Y[
                i2] * self.kernel(self.X[i2], self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b
            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2
            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new
            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)

    def predict(self, X_data):
        '''
        input:data(ndarray):单个样本
        output:预测为正样本返回+1，负样本返回-1
        '''
        y_pred = []
        for data in X_data:
            r = self.b
            for i in range(self.m):
                r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])
            y_pred.append(torch.sign(r).item())
        return torch.tensor(y_pred, dtype=torch.float).cuda()

    def get_logit(self, X_data, y_data):
        y_pred = []
        for data in X_data:
            r = self.b
            for i in range(self.m):
                r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])
            y_pred.append(r.item())
        return torch.tensor(y_pred, dtype=torch.float).cuda()


class EMD_dis:
    def __init__(self, opt):
        super(EMD_dis, self).__init__()
        self.opt = opt
        self.source_prototype = dict()
        self.target_prototype = dict()
        self.source_prototype_list = []
        self.target_prototype_list = []
        self.source_cov = []
        self.target_cov = []
        for n in range(5):
            self.source_prototype[int(n)] = []
        for n in range(5):
            self.target_prototype[int(n)] = []

    def cal_dis(self, support, query, query_y, class_predic, train):
        class_pred = class_predic.clone().detach().cuda()
        class_pred = class_pred.reshape([2, -1, self.opt.way_num])
        source_predic, target_predic = class_pred[0], class_pred[1]
        # target_predic = F.softmax(target_predic, dim=1)
        query_label = query_y.clone().detach().cuda()
        query_label = torch.reshape(query_label, [2, -1])
        target_predic = F.softmax(target_predic, dim=1)
        max_probs, targets_u = torch.max(target_predic.detach(), dim=-1)
        mask = max_probs.ge(0.2).float()

        feature_cl = query.clone().detach().cuda()
        feature_cl_tmp = torch.reshape(feature_cl, [2, -1, feature_cl.size(-1)])
        source_query, target_query = feature_cl_tmp[0], feature_cl_tmp[1]
        for j in range(5):
            self.source_prototype[j] = (support[j])
        if train:
            for i in range(len(source_query)):
                if self.source_prototype[int(query_label[0][i])] == []:
                    self.source_prototype[int(query_label[0][i])] = (source_query[i].unsqueeze(0))
                else:
                    self.source_prototype[int(query_label[0][i])] = torch.cat((self.source_prototype[int(query_label[0][i])], source_query[i].unsqueeze(0)), dim=0)
        else:
            pass
        self.source_prototype_list.clear()
        for i in range(5):
            self.source_prototype_list.append(self.source_prototype[i])

        source_prototype_tensor = torch.stack(self.source_prototype_list, dim=0)
        source_mean = source_prototype_tensor.mean(dim=1).detach()
        source_var = torch.sqrt(source_prototype_tensor.var(unbiased=False, dim=1) + 1e-6).detach()
        self.source_cov.clear()
        for i in range(source_prototype_tensor.size(0)):
            self.source_cov.append(self.get_cov(source_prototype_tensor[i]))
        count_yes, count_no = 0, 0

        for i in range(len(target_query)):
            if self.target_prototype[int(targets_u[i])] == [] and int(mask[i]):
                self.target_prototype[int(targets_u[i])] = (target_query[i].unsqueeze(0))
                if i < 15 and int(targets_u[i]) == 0:
                    count_yes += 1
                elif 14 < i and i < 30 and int(targets_u[i]) == 1:
                    count_yes += 1
                elif 29 < i and i < 45 and int(targets_u[i]) == 2:
                    count_yes += 1
                elif 44 < i and i < 60 and int(targets_u[i]) == 3:
                    count_yes += 1
                elif 59 < i and i < 75 and int(targets_u[i]) == 4:
                    count_yes += 1
                else:
                    count_no += 1
            elif int(mask[i]):
                self.target_prototype[int(targets_u[i])] = torch.cat((self.target_prototype[int(targets_u[i])], target_query[i].unsqueeze(0)), dim=0)
                if i < 15 and int(targets_u[i]) == 0:
                    count_yes += 1
                elif 14 < i and i < 30 and int(targets_u[i]) == 1:
                    count_yes += 1
                elif 29 < i and i < 45 and int(targets_u[i]) == 2:
                    count_yes += 1
                elif 44 < i and i < 60 and int(targets_u[i]) == 3:
                    count_yes += 1
                elif 59 < i and i < 75 and int(targets_u[i]) == 4:
                    count_yes += 1
                else:
                    count_no += 1
            else:
                pass
        self.target_prototype_list.clear()
        for i in range(5):
            self.target_prototype_list.append(self.target_prototype[i])

        flag = False
        for isture in self.target_prototype_list:
            if len(isture) < 2:
                flag = False
                break
            else:
                flag = True
        if flag:
            target_mean = []
            target_var = []
            for i in range(len(self.target_prototype_list)):
                target_mean.append(self.target_prototype_list[i].mean(dim=0).detach())
                target_var.append(torch.sqrt(self.target_prototype_list[i].var(unbiased=False, dim=0) + 1e-6).detach())
                self.target_cov.append(self.get_cov(self.target_prototype_list[i]))
            target_mean = torch.stack(target_mean, dim=0)
            target_var = torch.stack(target_var, dim=0)
            # spa_loss = 0.0
            # for i in range(len(self.target_prototype_list)):
            #     spa_loss += (self.source_cov[i] - self.target_cov[i]).pow(2).sum()
            # spa_loss /= len(self.target_prototype_list)
            loss_dis = ((source_mean - target_mean).norm() + (source_var - target_var).norm()) / 5
            # loss_dis = self.KL_cov_loss(source_prototype_tensor, self.target_prototype_list)
        else:
            loss_dis = 0.0
            loss_dis = torch.tensor(loss_dis).cuda()

        return loss_dis, count_yes, count_no

    def EMD_loss(self, p_mean, p_var, q_mean, q_var, n_class):
        return ((p_mean - q_mean).norm() + (p_var - q_var).norm()) / n_class

    def KL_cov_loss(self, source_prototype, target_prototype): # p_mean = source_mean(真实分布) q_mean=target_mean(理论分布)
        # return F.kl_div(q_mean.softmax(dim=-1).log(), p_mean.softmax(dim=-1), reduction='sum')
        # return np.sum(p * np.log(p/q))
        loss_list = []
        qm = source_prototype.size(0)
        for class_idx in range(qm):
            mean_s = source_prototype[class_idx].mean(0, True)
            mean_q = target_prototype[class_idx].mean(0, True)
            sub_sq = mean_s - mean_q
            cov_s = self.get_cov(source_prototype[class_idx])
            cov_t = self.get_cov(target_prototype[class_idx])
            loss_list.append(0.5 * (self.KL_loss(cov_s, cov_t) + (sub_sq ** 2).sum()))
        loss = torch.stack(loss_list, 0).mean()
        return loss

    def KL_loss(self, cov_source, cov_target):
        loss = (cov_source - cov_target).pow(2).sum()
        return loss

    def JS_loss(self, p_mean, q_mean):
        p = p_mean  # np.array(p_mean)
        q = q_mean  # np.array(q_mean)
        M = (p + q) / 2
        return 0.5 * np.sum(p*np.log(p/M)) + 0.5 * np.sum(q*np.log(q/M))

    def MMD_loss(self, p_mean, q_mean, n_class):
        return (p_mean-q_mean).norm() / n_class

    def get_cov(self, x):
        mean = x.mean(0, True)
        va = x - mean
        cov = torch.matmul(va.t(), va) / (x.size(1) - 1) + 0.01 * torch.eye(n=x.size(-1)).type(x.dtype).cuda(x.device)
        return cov