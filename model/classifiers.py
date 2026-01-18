import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.weight_norm import WeightNorm
import numpy as np


class distLinear(nn.Module):
    def __init__(self, indim, outdim, pp=False):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = pp  #See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist)
        return scores


class BaselinePP(nn.Module):
    def __init__(self, iter=100):
        super(BaselinePP, self).__init__()
        self.iter = iter

    def get_cls(self, C, h, w, way):
        return distLinear(C * h * w, way)

    def forward(self, query, support):
        # way, shot, C, h, w
        support = torch.stack(support, 0)
        way, shot, C, h, w = support.size()
        support = support.view(way*shot, -1)
        support_y = torch.arange(0, way, dtype=torch.long).unsqueeze(0).repeat(shot, 1).t().reshape([-1]).to(support.device)
        query = query.view(query.size(0), -1)
        classifier = self.get_cls(C, h, w, way).to(support.device)
        set_optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
                                        weight_decay=0.001)
        loss_function = nn.CrossEntropyLoss().to(support.device)
        loss_function = loss_function.to(support.device)
        batch_size = 4
        support_size = way * shot
        classifier.train()
        with torch.enable_grad():
            for epoch in range(self.iter):
                rand_id = np.random.permutation(support_size)
                for i in range(0, support_size, batch_size):
                    set_optimizer.zero_grad()
                    selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                    z_batch = support[selected_id].detach()
                    y_batch = support_y[selected_id]
                    scores = classifier(z_batch)
                    loss = loss_function(scores, y_batch)
                    set_optimizer.zero_grad()
                    loss.backward()
                    set_optimizer.step()
        classifier.eval()
        scores = classifier(query)
        return scores

class Baseline(BaselinePP):
    def __init__(self):
        super(Baseline, self).__init__()

    def get_cls(self, C, h, w, way):
        return distLinear(C * h * w, way, pp=True)