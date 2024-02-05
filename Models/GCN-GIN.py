import torch
import torch_geometric
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
import torch.nn.functional as F
from torch_geometric.nn import  GCNConv,GINConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU


class GCN_GIN(torch.nn.Module):
	def __init__(self):
		super(GCN_GIN, self).__init__()
		#GCN-network
		self.conv1 = GCNConv(373, 256, cached=False )
		self.bn01 = BatchNorm1d(256)
		self.conv2 = GCNConv(256, 128, cached=False )
		self.bn02 = BatchNorm1d(128)
		self.conv3 = GCNConv(128, 128, cached=False)
		self.bn03 = BatchNorm1d(128)
		#GIN-network
		fc_gin1=Sequential(Linear(373, 256), ReLU(), Linear(256, 256))
		self.gin1 = GINConv(fc_gin1)
		self.bn21 = BatchNorm1d(256)
		fc_gin2=Sequential(Linear(256, 128), ReLU(), Linear(128, 128))
		self.gin2 = GINConv(fc_gin2)
		self.bn22 = BatchNorm1d(128)
		fc_gin3=Sequential(Linear(128, 64), ReLU(), Linear(64, 64))
		self.gin3 = GINConv(fc_gin3)
		self.bn23 = BatchNorm1d(64)
		#FCN 
		self.fc1=Linear(128 + 64, 64)
		self.fc2=Linear(64, 64)
		self.fc3=Linear(64, 1)
	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		z=x
		#GCN-representation
		x = F.relu(self.conv1(x, edge_index))
		x = self.bn01(x)
		x = F.relu(self.conv2(x, edge_index))
		x = self.bn02(x)
		x = F.relu(self.conv3(x, edge_index))
		x = self.bn03(x)
		x = global_add_pool(x, data.batch)
		#GIN-representation
		z = F.relu(self.gin1(z, edge_index))
		z = self.bn21(z)
		z = F.relu(self.gin2(z, edge_index))
		z = self.bn22(z)
		z = F.relu(self.gin3(z, edge_index))
		z = self.bn23(z)
		z = global_add_pool(z, data.batch)
		#Concatinating_representations
		cr=torch.cat((x,z),1)
		cr = F.relu(self.fc1(cr))
		cr = F.dropout(cr, p=0.2, training=self.training)
		cr = F.relu(self.fc2(cr))
		cr = F.dropout(cr, p=0.2, training=self.training)
		cr = self.fc3(cr)
		cr = F.relu(cr).view(-1)
		return cr  
