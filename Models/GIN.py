import torch
import torch_geometric
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import  GINConv, global_add_pool
import torch.nn.functional as F
	


class GIN(torch.nn.Module):
	def __init__(self):
		super(GIN, self).__init__()
		fc_gin1=Sequential(Linear(373, 256), ReLU(), Linear(256, 256))
		self.gin1 = GINConv(fc_gin1)
		self.bn1 = BatchNorm1d(256)
		fc_gin2=Sequential(Linear(256, 128), ReLU(), Linear(128, 128))
		self.gin2 = GINConv(fc_gin2)
		self.bn2 = BatchNorm1d(128)
		fc_gin3=Sequential(Linear(128, 64), ReLU(), Linear(64, 64))
		self.gin3 = GINConv(fc_gin3)
		self.bn3 = BatchNorm1d(64)
		self.fc1 = Linear(64, 16)
		self.fc2 = Linear(16, 1)
		 
	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		x = F.relu(self.gin1(x, edge_index))
		x = self.bn1(x)
		x = F.relu(self.gin2(x, edge_index))
		x = self.bn2(x)
		x = F.relu(self.gin3(x, edge_index))
		x = self.bn3(x)
		x = global_add_pool(x, data.batch)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, p=0.2, training=self.training)
		x = self.fc2(x)
		x = F.relu(x).view(-1)
		return x  
