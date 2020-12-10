import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Basic residual block from Kaimeng He's Resnet Paper:

class MLPModel(nn.Module):
  def __init__(self,k):
    super(MLPModel,self).__init__()
    self.seq1 = nn.Sequential(nn.Linear(k*k,k//2*k//2),nn.BatchNorm1d(k//2*k//2),nn.ReLU(),
                              nn.Linear(k//2*k//2,k//2*k//2),nn.BatchNorm1d(k//2*k//2),nn.ReLU(),
                              nn.Linear(k//2*k//2,k*k),nn.ReLU(),nn.Linear(k*k,k*k))
    self.k = k

  def forward(self,left,right):
    mult = left * right
    mult = mult.reshape(-1,self.k*self.k)
    
    #left, right = left.flatten(), right.flatten()
    feat = self.seq1(mult)
    cost = F.softmax(-feat)
    pred = cost.reshape(self.k,self.k)

    return pred