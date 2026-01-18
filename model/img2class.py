import torch
import torch.nn as nn

class ImgtoClass_Metric(nn.Module):
	def __init__(self, neighbor_k=3, discriminator=False):
		super(ImgtoClass_Metric, self).__init__()
		self.neighbor_k = neighbor_k
		self.discriminator = discriminator

	# Calculate the k-Nearest Neighbor of each local descriptor
	def cal_cosinesimilarity(self, input1, input2):
		B, h_w, C = input1.size()
		n_domain, n_way, n_shot, _,  _ = input2.size()
		
		if self.discriminator:
			input2_tmp = input2.contiguous().view(-1, h_w, C)
			n_shot = n_way * n_shot
			n_way = 2
		else:
			input2_tmp = input2[0].contiguous().view(n_way * n_shot, h_w, C)
		input2_tmp = input2_tmp.permute(0, 2, 1)
		support_norm = torch.norm(input2_tmp, 2, 1, True)
		query_norm = torch.norm(input1, 2, 2, True)
		Similarity_matrix = torch.einsum('ijk, bkc->ibjc', input1, input2_tmp)
		Similarity_matrix_norm = torch.einsum('ijk,bkc->ibjc', query_norm, support_norm)
		Similarity_list = (Similarity_matrix / Similarity_matrix_norm).sum(dim=-1).sum(dim=-1)
        
		if self.discriminator:
			Similarity_list = Similarity_list.contiguous().view(B, 2, -1).sum(dim=2)

		return Similarity_list

	def forward(self, x1, x2):

		Similarity_list = self.cal_cosinesimilarity(x1, x2)

		return Similarity_list