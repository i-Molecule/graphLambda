import torch
import torch_geometric
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import  GATConv, global_add_pool
import torch.nn.functional as F


class GAT(torch.nn.Module):
	def __init__(self):
		super(GAT, self).__init__()
		self.gat1 = GATConv(373, 256,heads=3)
		self.bn1 = BatchNorm1d(256*3)
		self.gat2 = GATConv(256*3, 128,heads=3)
		self.bn2 = BatchNorm1d(128*3)
		self.gat3 = GATConv(128*3, 128,heads=3)
		self.bn3 = BatchNorm1d(128*3)
		self.fc1 = Linear(128*3, 64)
		self.bn4 = BatchNorm1d(64)
		self.fc2 = Linear(64, 64)
		self.fc3 = Linear(64, 1)
		 
	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		x = F.relu(self.gat1(x, edge_index))
		x = self.bn1(x)
		x = F.relu(self.gat2(x, edge_index))
		x = self.bn2(x)
		x = F.relu(self.gat3(x, edge_index))
		x = self.bn3(x)
		x = global_add_pool(x, data.batch)
		x = F.relu(self.fc1(x))
		x = self.bn4(x)
		x = F.relu(self.fc2(x))
		x = F.dropout(x, p=0.2, training=self.training)
		x = self.fc3(x)
		x = F.relu(x).view(-1)
		return x
