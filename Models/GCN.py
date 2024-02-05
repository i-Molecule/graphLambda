import torch
import torch_geometric
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import  GCNConv, global_add_pool
import torch.nn.functional as F

class GCN(torch.nn.Module):
	def __init__(self):
		super(GCN, self).__init__()
		self.conv1 = GCNConv(373, 256, cached=False )
		self.bn1 = BatchNorm1d(256)
		self.conv2 = GCNConv(256, 128, cached=False )
		self.bn2 = BatchNorm1d(128)
		self.conv3 = GCNConv(128, 128, cached=False)
		self.bn3 = BatchNorm1d(128)
		self.fc1 = Linear(128, 64)
		self.bn4 = BatchNorm1d(64)
		self.fc2 = Linear(64, 64)
		self.fc3 = Linear(64, 1)
	 
	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		x = F.relu(self.conv1(x, edge_index))
		x = self.bn1(x)
		x = F.relu(self.conv2(x, edge_index))
		x = self.bn2(x)
		x = F.relu(self.conv3(x, edge_index))
		x = self.bn3(x)
		x = global_add_pool(x, data.batch)
		x = F.relu(self.fc1(x))
		x = self.bn4(x)
		x = F.relu(self.fc2(x))
		x = F.dropout(x, p=0.2, training=self.training)
		x = self.fc3(x)
		x = F.relu(x).view(-1)
		return x 


