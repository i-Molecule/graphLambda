import torch
import torch_geometric
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
import torch.nn.functional as F
from torch_geometric.nn import  GCNConv, global_add_pool,GATConv
from torch.nn import Sequential, Linear, ReLU


class GCN_GAT(torch.nn.Module):
	def __init__(self):
		super(GCN_GAT, self).__init__()
		#GCN-network
		self.conv1 = GCNConv(373, 256, cached=False )
		self.bn01 = BatchNorm1d(256)
		self.conv2 = GCNConv(256, 128, cached=False )
		self.bn02 = BatchNorm1d(128)
		self.conv3 = GCNConv(128, 128, cached=False)
		self.bn03 = BatchNorm1d(128)
		#GAT-network
		self.gat1 = GATConv(373, 256,heads=3)
		self.bn11 = BatchNorm1d(256*3)
		self.gat2 = GATConv(256*3, 128,heads=3)
		self.bn12 = BatchNorm1d(128*3)
		self.gat3 = GATConv(128*3, 128,heads=3)
		self.bn13 = BatchNorm1d(128*3)
		#FCN
		self.fc1=Linear(128*4 , 256)
		self.fc2=Linear(256, 64)
		self.fc3=Linear(64, 1)
	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		y=x
		#GCN-representation
		x = F.relu(self.conv1(x, edge_index))
		x = self.bn01(x)
		x = F.relu(self.conv2(x, edge_index))
		x = self.bn02(x)
		x = F.relu(self.conv3(x, edge_index))
		x = self.bn03(x)
		x = global_add_pool(x, data.batch)
		#GAT-representation
		y = F.relu(self.gat1(y, edge_index))
		y = self.bn11(y)
		y = F.relu(self.gat2(y, edge_index))
		y = self.bn12(y)
		y = F.relu(self.gat3(y, edge_index))
		y = self.bn13(y)
		y = global_add_pool(y, data.batch)
		#Concatinating_representations
		cr=torch.cat((x,y),1)
		cr = F.relu(self.fc1(cr))
		cr = F.dropout(cr, p=0.2, training=self.training)
		cr = F.relu(self.fc2(cr))
		cr = F.dropout(cr, p=0.2, training=self.training)
		cr = self.fc3(cr)
		cr = F.relu(cr).view(-1)
		return cr  
