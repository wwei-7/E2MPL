from model.BaseModel import BaseModelFSL
from torch import nn
import torch
import math

class IMSE(BaseModelFSL):
    def __init__(self, neighbor_k=3, kernel_size=1, sigma=0.8):
        super().__init__()
        self.neighbor_k = neighbor_k
        self.kernel_size = kernel_size
        self.sigma = sigma

    def get_gauss_kernel(self, k, sigma, device):
        row = 2 * k + 1
        col = 2 * k + 1
        A = []
        sigma = torch.tensor(sigma, dtype=torch.float64)
        for i in range(row):
            r = []
            for j in range(col):
                fenzi = (i + 1 - k - 1) ** 2 + (j + 1 - k - 1) ** 2
                r.append(torch.exp(-fenzi / (2 * sigma)) / (2 * torch.pi * sigma))
            A.append(torch.stack(r, 0))
        A = torch.stack(A, 0)
        A = A / A.sum()
        gauss_kernel = A.view(1, 1, 1, 2 * k + 1, 2 * k + 1).type(torch.float32).to(device)
        return gauss_kernel

    def cal_cosinesimilarity(self, query_x, support_x):
        self.gauss_kernel = self.get_gauss_kernel(k=self.kernel_size, sigma=self.sigma, device=query_x.device)
        Similarity_list = []
        query_x = query_x.permute(0, 2, 3, 1)
        qm, qh, qw, C = query_x.size()
        q_feature_size = qh * qw
        query_x = torch.reshape(query_x, [qm, q_feature_size, C])
        cov_list = []
        for i in range(len(support_x)):
            # prototype_x = torch.nn.functional.max_pool2d(support_x[i], 2, 2)
            prototype_x = support_x[i].permute(0, 2, 3, 1)
            shot, ph, pw, C = prototype_x.size()
            p_feature_size = ph * pw
            prototype_x = torch.reshape(prototype_x, [shot, p_feature_size, C])
            res_q2s = self.get_lds(prototype_x, query_x)
            topk = self.neighbor_k
            q2s = self.compose_ld(res_q2s, topk, q_feature_size, p_feature_size, qm, shot).permute(0, 1, 3, 2)
            res = torch.reshape(q2s, [qm * shot, 1, qh * qw, ph, pw])
            pool_size = 2
            vol_pool_size = 1
            for i in range(8):
                # print(res.size())
                size = res.size()[-1]
                if size == 1:
                    break
                # res = self.gauss_kernel(res)
                res = torch.nn.functional.conv3d(res, self.gauss_kernel.detach(), stride=[1, 1, 1],
                                                 padding=0)
                size = res.size()[-1]
                vol_size = res.size()[-3]
                if size == 1:
                    break
                elif pool_size > size:
                    pool_size = size
                if vol_size == 1:
                    vol_pool_size = vol_size
                res = torch.nn.functional.max_pool3d(res, kernel_size=[vol_pool_size, pool_size, pool_size],
                                                     stride=[1, pool_size, pool_size])
            res = res.reshape([-1, 1, qh, qw])
            cov_list.append(res.view(-1, shot, ph*pw))
            score_vec = res.reshape([qm, -1]).sum(-1)
            Similarity_list.append(score_vec)
        Similarity_list = torch.stack(Similarity_list, 0).t()
        self.compute_spa_loss(torch.cat(cov_list, 1).permute(1, 0, 2))
        return {"logits": Similarity_list}

    def compute_spa_loss(self, cov_list):
        def cov(x):
            x_mean = x.mean(0, True)
            covars_ = (x - x_mean).t() @ (x - x_mean) / (x.size(0) - 1)
            return covars_
        spa_loss = 0.0
        for cls in cov_list:
            s, t = cls.view(2, -1, cls.size(-1))
            spa_loss += (cov(s) - cov(t)).pow(2).sum()
        spa_loss /= cov_list.size(0)
        self.set_output('spa_loss', spa_loss)

    def get_cov(self, x):
        mean = x.mean(0)
        va = x - mean.expand_as(x)
        cov = torch.matmul(va.t(), va)/(x.size(0)-1)
        return cov

    def get_lds(self, p, q):
        innerproduct = torch.einsum('ijk,spk->isjp', q, p)

        q2 = torch.sqrt(torch.einsum('ijk,ijk->ij', q, q))
        p2 = torch.sqrt(torch.einsum('ijk,ijk->ij', p, p))

        q2p2 = torch.einsum('ij,sp->isjp', q2, p2)

        res = innerproduct / q2p2

        return res

    def compose_ld(self, qlds, topk, q_feature_size, p_feature_size, qm, pm, ):
        qm, pm, q_feature_size, p_feature_size = qlds.size()
        qlds = qlds.permute(0, 2, 1, 3)
        qlds = qlds.reshape([qm, q_feature_size, -1])
        q2s_topk, q2s_idx = torch.topk(qlds, k=topk,  dim=-1)
        q2s_idx, idxs = self.handel_topidx(q2s_idx, qm, pm, q_feature_size, topk)
        q2s_topk = torch.squeeze(torch.reshape(q2s_topk, [1, -1])).cuda()
        q2s = torch.sparse.FloatTensor(q2s_idx.t(), q2s_topk,
                                       torch.Size((qm, q_feature_size, pm * p_feature_size))).to_dense()
        q2s = q2s.reshape([qm, q_feature_size, pm, p_feature_size])
        q2s = q2s.permute(0, 2, 1, 3)

        return q2s

    def handel_topidx(self, idxs, qm, pm, m, k):
        all_num = qm * m * k
        idxs = torch.reshape(idxs, [qm, m, k])
        idxs = torch.reshape(idxs, [qm * m * k])

        q_idx = torch.tensor(list(range(qm)), dtype=torch.int64).cuda()
        q_idx = torch.reshape(q_idx, [qm, 1])
        q_idx = torch.squeeze(torch.reshape(q_idx.repeat([1, int(all_num / qm)]), [1, all_num]))

        m_idx = torch.tensor(list(range(m)), dtype=torch.int64).cuda()
        m_idx = torch.reshape(m_idx, [m, 1])
        m_idx = m_idx.repeat([1, k])
        m_idx = torch.reshape(m_idx, [1, int(m * k)])
        m_idx = m_idx.repeat([1, qm])
        m_idx = torch.squeeze(torch.reshape(m_idx, [all_num, 1]))
        sparse_idx = torch.reshape(torch.transpose(torch.stack([q_idx, m_idx, idxs], dim=0), dim0=1, dim1=0),
                                   [all_num, -1])
        return sparse_idx, idxs

    def forward_(self, x1, x2, is_train):
        Similarity_list = self.cal_cosinesimilarity(x1, x2)
        return Similarity_list


class GaussianLayer(BaseModelFSL):
    def __init__(self, kernel_size=1, template_num=5):
        super().__init__()
        self.k = kernel_size
        self.sigma_parameters = nn.Parameter(torch.tensor(torch.rand([template_num])))
        self.gaussian_kernels = None
        self.get_params()
        
    def get_gauss_kernel(self):
        kernel_list = []
        for s in self.sigma_parameters:
            sigma = 2 * (torch.sigmoid(s/self.sigma_parameters.norm()) + 1e-5)
            gauss_kernel = ((-self.mat.to(sigma.device))/sigma).exp()/(torch.pi * sigma)
            gauss_kernel = gauss_kernel/gauss_kernel.sum()
            kernel_list.append(gauss_kernel)
        kernel_list = torch.cat(kernel_list, 0).type(torch.float32)
        self.gaussian_kernels = kernel_list
        
    def get_params(self):
        row = 2 * self.k + 1
        col = 2 * self.k + 1
        mat = []
        for i in range(row):
            r = []
            for j in range(col):
                fenzi = (i + 1 - self.k - 1) ** 2 + (j + 1 - self.k - 1) ** 2
                r.append(fenzi)
            mat.append(r)
        self.mat = torch.tensor(mat, dtype=torch.float64, requires_grad=False).view(1, 1, 1, 2 * self.k + 1, 2 * self.k + 1)

    def forward_(self, x):
        res = torch.nn.functional.conv3d(x, self.gaussian_kernels, stride=1, padding=[0, self.k, self.k])
        return {'res': res}



class IMSE_ASP(IMSE):
    def __init__(self, neighbor_k=3, kernel_size=1, sigma=0.8, template_num=5, mask_scale=10):
        super(IMSE_ASP, self).__init__(neighbor_k, kernel_size, sigma)
        self.template_num_list = [5, 5, 5]
        self.gaussian_layers = nn.ModuleList([GaussianLayer(kernel_size=1, template_num=template_num),
                                              GaussianLayer(template_num=template_num), 
                                              GaussianLayer(template_num=template_num)])
        self.wq = nn.Linear(640, 640)
        self.ws = nn.Linear(640, 320)
        self.Inor = nn.BatchNorm2d(640)
        self.Inor2 = nn.BatchNorm1d(640)
        self.W_phi_q = nn.Linear(640, 640)
        self.W_phi_k = nn.Linear(640, 640)
        self.W_phi_v = nn.Linear(640, 640)
        self.mask_scale = mask_scale
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                a = m.weight.clone()
                torch.nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def Normalize(self, query, support, bn_module):
        qm, C, qh, qw = query.size()
        n_way, n_shot, _, _, _ = support.size()
        support = support.view(-1, C, qh, qw)
        tmp = bn_module(torch.cat([query, support], 0))
        query, support = tmp[:qm], tmp[qm:].view(n_way, n_shot, C, qh, qw)
        return query, support

    def innerproduct(self, q, k):
        res = torch.einsum('ik,lpk->ilp', q, k)
        return res

    def mask_normalize(self, mask):
        return self.mask_scale*nn.functional.normalize(mask, p=2, dim=-1)

    def fusion_block(self, res, i):
        return res.max(1, True)[0]

    def compose_ld(self, qlds, topk, q_feature_size, p_feature_size, qm, pm, ):
        V, I = qlds.topk(k=topk, dim=-1, sorted=False, largest=True)
        tmp = torch.zeros_like(qlds)
        return tmp.scatter(-1, I, V)
    
    def att(self, query):
        Q = self.W_phi_q(query)
        K = self.W_phi_k(query)
        V = self.W_phi_v(query)
        att_map = (torch.einsum("ijk,ilk->ijl", Q, K)/math.sqrt(640)).softmax(-1)
        res = torch.einsum("ilj,ijk->ilk", att_map, V) + V
        res = self.Inor2(res.permute(0, 2, 1)).permute(0, 2, 1)
        return res
    
    def compute_msk_info_loss(self, mask_list_, logits, query_num=15):
        def mse(x, y):
            return ((x-y)**2).sum([-1, -2])
        def cov(x):
            tmp = x - x.mean(0, True)
            cov_mat = 100 * tmp.t() @ tmp / (tmp.size(0) - 1)
            return cov_mat
        class_masks = []
        diff_class_masks = []
        mask_entropy = 0
        
        mask_list = torch.stack([m[:m.size(0) // 2] for m in mask_list_], 0)
        target_mask_list = torch.stack([m[m.size(0) // 2:] for m in mask_list_], 0)
        for c in range(len(mask_list)):
            # tmp_mask = mask_list[c][c*query_num:(c+1)*query_num].view(query_num, -1) * logits[c*query_num:(c+1)*query_num][:, c].view(query_num, 1).softmax(0)
            tmp_mask = mask_list[c][c*query_num:(c+1)*query_num].view(query_num, -1)
            y_ = logits[c*query_num:(c+1)*query_num].softmax(-1)[:, c].unsqueeze(-1)
            mask_entropy += ((-torch.log(tmp_mask)*y_).mean())/len(mask_list)
            # print(tmp_mask.size(), y_.size(), mask_entropy, y_); exit()
            class_masks.append(cov(tmp_mask))
            diff_class_masks.append(cov(torch.cat([mask_list[c][:c*query_num], mask_list[c][(c+1)*query_num:]], 0).view(-1, tmp_mask.size(-1))))
        class_masks = torch.stack(class_masks, 0)
        diff_class_masks = torch.stack(diff_class_masks, 0)
        target_mask_list_cov = torch.stack([cov(t.view(-1, tmp_mask.size(-1))) for t in target_mask_list], 0)
        msk_info_loss = mask_entropy*0.5 + (-(target_mask_list).log()).mean() * 0.1 + mse(class_masks, diff_class_masks).sum() + mse(class_masks, target_mask_list_cov).sum()
        self.set_output('msk_info_loss', msk_info_loss)
    
    def cal_cosinesimilarity(self, query_x, support_x):
        Similarity_list = []
        # query_x = torch.nn.functional.max_pool2d(query_x, 2, 2)
        att_query, att_support = self.Normalize(query_x, support_x, self.Inor)
        query_x = query_x.permute(0, 2, 3, 1)
        qm, qh, qw, C = query_x.size()
        q_feature_size = qh * qw
        query_x = torch.reshape(query_x, [qm, q_feature_size, C])
        n_way, n_shot, C, ph, pw = support_x.size()
        support_x = support_x.view(n_way, n_shot, C, -1).permute(0, 1, 3, 2).contiguous().view(n_way, -1, C)
        simi_mat = self.get_lds(support_x, query_x)
        att_query = att_query.permute(0, 2, 3, 1).view(qm, -1, C)
        att_support = att_support.view(n_way, n_shot, C, -1).permute(0, 1, 3, 2).contiguous().view(n_way, -1, C)
        trans_query = self.wq(self.att(att_query).mean(1, True))
        # print(self.att(att_query).size(), self.att(att_query))
        # support_weight = self.get_support_weight(support_x)
        cov_list = []
        H, W = qh, qw
        mask_list = []
        pattern_list = []
        trans_support = self.wq(self.att(att_support).contiguous().view(-1, C)).view(n_way, n_shot * ph * pw, -1)
        # query_mask_classes = (10*nn.functional.normalize(self.innerproduct(trans_query.mean(1), trans_support).view(qm, n_way*n_shot*ph*pw), p=2, dim=-1)).softmax(-1).view(qm, -1, ph, pw)
        query_mask_classes = self.mask_normalize(self.innerproduct(trans_query.mean(1), trans_support).view(qm, n_way*n_shot*ph*pw)).softmax(-1).view(qm, -1, ph, pw)
        for k in range(len(self.gaussian_layers)):
            self.gaussian_layers[k].get_gauss_kernel()
        for i in range(n_way):
            p_feature_size = ph * pw
            q2s = self.compose_ld(simi_mat[:, i].unsqueeze(1), self.neighbor_k, q_feature_size, p_feature_size, qm, n_shot).permute(0, 1, 3, 2)
            res = torch.reshape(q2s, [qm, 1, n_shot * ph * pw, qh, qw])
            pool_size = 2
            H, W = qh, qw
            for j in range(3):
                res = self.gaussian_layers[j](res)['res']
                # print(res.size()); exit()
                _, pm, psize_, H_, W_ = res.size()
                # res = res.max(1, True)[0]
                res = self.fusion_block(res, j)
                res = res.view(qm, 1, -1, H_, W_)
                res = torch.nn.functional.max_pool3d(res, kernel_size=[1, pool_size, pool_size],
                                                     stride=[1, pool_size, pool_size])
            res = res.view(qm, -1, ph, pw)
            mask_list.append(query_mask_classes[:, i].unsqueeze(1))
            res = res*query_mask_classes[:, i].unsqueeze(1) + res
            res = res.view(qm, -1, ph, pw)
            cov_list.append(res.view(-1, n_shot, ph * pw))
            score_vec = (res).reshape([qm, -1]).sum(-1)
            Similarity_list.append(score_vec)
        logits = torch.stack(Similarity_list, 1)
        self.compute_spa_loss(torch.cat(cov_list, 1).permute(1, 0, 2))
        self.compute_msk_info_loss(mask_list, logits[:logits.size(0) // 2])
        return {"logits": logits}

