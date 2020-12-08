import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Basic residual block from Kaimeng He's Resnet Paper:

class residual_block(nn.Module):
  def __init__(self,in_channel,out_channel,kernel,stride,padding,dilation):
    super(residual_block, self).__init__()

    self.in_channels = in_channel
    self.out_channels = out_channel
    self.stride = stride
    self.dilation = dilation

    layers = [nn.Conv2d(in_channel, out_channel, kernel, stride, padding, dilation = dilation, bias = False), nn.BatchNorm2d(out_channel),nn.ReLU(inplace=True),
              nn.Conv2d(out_channel, out_channel, kernel, 1, padding,dilation=dilation, bias = False), nn.BatchNorm2d(out_channel)]

    self.layers = nn.Sequential(*layers)


  def forward(self,x):
    if self.in_channels == self.out_channels: # if-else block deals with the condition when number of channels change across residual/skip connection
      input = x
    else:
      diff = self.out_channels - self.in_channels
      x1 = torch.zeros(x.shape[0],diff,x.shape[2],x.shape[3]).to(device)
      input = torch.cat((x,x1),dim=1)


    if self.stride == 2: # Downsample input tensor when number of channels change across residual connections and stride is 2.
      input = F.max_pool2d(input,kernel_size = 2,stride = 2)

    output = self.layers(x)
    output += input
    output = F.relu(output)
    return output


# CNN module (Feature Extractor):

class CNN(nn.Module):
  def __init__(self):
    super(CNN,self).__init__()

    layers_1 = [
              nn.Conv2d(in_channels = 1,out_channels = 32, kernel_size = 3, stride = 2, padding = 1, bias = False), nn.BatchNorm2d(32),nn.ReLU(inplace=True), # H/2 x W/2 x 32 Conv0_1
              nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm2d(32),nn.ReLU(inplace=True), # H/2 x W/2 x 32 Conv0_2
              nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm2d(32),nn.ReLU(inplace=True), # H/2 x W/2 x 32 Conv0_3
              residual_block(32,32,3,1,1,1), residual_block(32,32,3,1,1,1), residual_block(32,32,3,1,1,1), # H/2 x W/2 x 32 Conv1_ 1-3
              residual_block(32,64,3,2,1,1), residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), # Conv2_ 1-4
              residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), # Conv2_ 5-8
              residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), # Conv2_ 9-12
              residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1) # H/4 x W/4 x 64 Conv2_ 13-16
              ]
              
    layers_2 = [      
              residual_block(64,128,3,1,2,2), residual_block(128,128,3,1,2,2), residual_block(128,128,3,1,2,2), # H/4 x W/4 x 64 Conv3_ 1-3
              residual_block(128,128,3,1,4,4), residual_block(128,128,3,1,4,4), residual_block(128,128,3,1,4,4) # H/4 x W/4 x 64 Conv4_ 1-3 
              ]

    # Split sequential block into two lists to extract intermediate outputs for concatentaion in subsequent modules:

    self.layers_1 = nn.Sequential(*layers_1)
    self.layers_2 = nn.Sequential(*layers_2)

  def forward(self,x):
    cnn_concat1 = self.layers_1(x)
    cnn_concat2 = self.layers_2(cnn_concat1)
    return cnn_concat1,cnn_concat2 # Return two outputs(used further in concat operation)


# SPP module (Spatial Pyramidal Pooling):

class SPP(nn.Module):
  def __init__(self):
    super(SPP,self).__init__()

    self.cnn_module = CNN()

    self.branch1 = nn.Sequential(
                  nn.AvgPool2d(16,stride=16),
                  nn.Conv2d(in_channels = 128,out_channels = 32, kernel_size = 1, stride = 1, padding = 0, dilation = 1, bias = False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
                  ) # branch_1

    self.branch2 = nn.Sequential(
                  nn.AvgPool2d(8,stride=8),
                  nn.Conv2d(in_channels = 128,out_channels = 32, kernel_size = 1, stride = 1, padding = 0, dilation = 1, bias = False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
                  ) # branch_2

    self.branch3 = nn.Sequential(
                  nn.AvgPool2d(4,stride=4),
                  nn.Conv2d(in_channels = 128,out_channels = 32, kernel_size = 1, stride = 1, padding = 0, dilation = 1, bias = False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
                  ) # branch_3

    self.branch4 = nn.Sequential(
                  nn.AvgPool2d(2,stride=2),
                  nn.Conv2d(in_channels = 128,out_channels = 32, kernel_size = 1, stride = 1, padding = 0, bias = False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
                  ) # branch_4

    self.final_conv = nn.Sequential(
                    nn.Conv2d(in_channels = 320, out_channels =128, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 1, stride = 1, bias = False)
                    ) # fusion

  def forward(self,x):
    conv2_16,conv4_3 = self.cnn_module(x)
    branch1 = self.branch1(conv4_3)
    branch2 = self.branch2(conv4_3)
    branch3 = self.branch3(conv4_3)
    branch4 = self.branch4(conv4_3)
    branch1 = F.interpolate(branch1, (conv4_3.shape[2],conv4_3.shape[3]),mode='bilinear')
    branch2 = F.interpolate(branch2, (conv4_3.shape[2],conv4_3.shape[3]),mode='bilinear')
    branch3 = F.interpolate(branch3, (conv4_3.shape[2],conv4_3.shape[3]),mode='bilinear')
    branch4 = F.interpolate(branch4, (conv4_3.shape[2],conv4_3.shape[3]),mode='bilinear')
    concat_tensor = torch.cat((conv2_16,conv4_3,branch1,branch2,branch3,branch4), dim = 1) # Concat all pooled components (Concat block of SPP)
    spp_output = self.final_conv(concat_tensor)
    return spp_output


# Custom class for disparity regression (Reference - https://github.com/JiaRenChang/PSMNet):

class disp_reg(nn.Module):
  def __init__(self, maxdisp):
    super(disp_reg, self).__init__()
    self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(), requires_grad=False)

  def forward(self, x):
    disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
    out  = torch.sum(x*disp,1)
    return out

# Class for defining the entire network:
# Takes a rectified stereo pair as input: 

class PSMNET(nn.Module):
  def __init__(self,max_disp):
    super(PSMNET,self).__init__()

    self.max_disp = max_disp

    self.disp = self.max_disp // 4

    self.spp = SPP()

    self.cnn3d_1 = nn.Sequential(nn.Conv3d(in_channels = 64,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                                 nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32), nn.ReLU(inplace=True)) # 3DConv_1
    
    self.cnn3d_2 = nn.Sequential(nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                                 nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32)) # 3DConv_2

    self.cnn3d_3 = nn.Sequential(nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                                 nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32)) # 3DConv_3 

    self.cnn3d_4 = nn.Sequential(nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                                 nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32)) # 3DConv_4

    self.cnn3d_5 = nn.Sequential(nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                                 nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32)) # 3DConv_5
    
    self.cnn3d_last = nn.Sequential(nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                                    nn.Conv3d(in_channels = 32,out_channels = 1, kernel_size = 3, stride = 1, padding = 1, bias = False)) # 3DConv_final

    self.disp_reg = disp_reg(self.max_disp)


  def forward(self,left,right):
    left_feat  = self.spp(left) # Extract feature map from left image
    right_feat = self.spp(right) # Extract feature map from right image

    ## Code for 3D stacked cost volume (Reference https://github.com/JiaRenChang/PSMNet):
    
    cost = torch.zeros(left_feat.shape[0], left_feat.shape[1] * 2, self.disp,  left_feat.shape[2], left_feat.shape[3]).float().cuda()

    for i in range(0, self.disp):
      if i > 0:
        cost[:, left_feat.shape[1]:, i, :, i:]   = left_feat[:,:,:,i:]
        cost[:, :left_feat.shape[1], i, :, i:]   = right_feat[:,:,:,:-i]
      else:
        cost[:, left_feat.shape[1]:, i, :, i:]   = left_feat
        cost[:, :left_feat.shape[1], i, :, i:]   = right_feat

    ##

    cost = self.cnn3d_1(cost)
    cost = F.relu(self.cnn3d_2(cost) + cost) # Additions represent skip connections across 3D Conv layers
    cost = F.relu(self.cnn3d_3(cost) + cost)
    cost = F.relu(self.cnn3d_4(cost) + cost)
    cost = F.relu(self.cnn3d_5(cost) + cost)
    cost = self.cnn3d_last(cost)

    # Trilinear upsampling to restore image to original size:
    cost = F.interpolate(cost, (self.max_disp,left.shape[2],left.shape[3]), mode = 'trilinear')

    # Eliminate num_channels dimension(it's 1 anyway):
    cost = torch.squeeze(cost,1)
    
    # Compute softmax across all disparity values:
    cost = F.softmax(-cost,dim=1)

    #Perform disparity regression as defined in paper:
    pred = self.disp_reg(cost)

    return pred


